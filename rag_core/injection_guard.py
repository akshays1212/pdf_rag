import re
import logging
from typing import Tuple, List, Dict

logger = logging.getLogger(__name__)

# ── injection patterns ────────────────────────────────────────────────

INJECTION_PATTERNS = [
    # direct override attempts
    r'(?i)ignore\s+(all\s+)?(previous|prior|above)\s+instructions?',
    r'(?i)disregard\s+(all\s+)?(previous|prior|above)\s+instructions?',
    r'(?i)forget\s+(all\s+)?(previous|prior|above)\s+instructions?',
    r'(?i)override\s+(all\s+)?instructions?',
    r'(?i)new\s+instructions?\s*:',
    r'(?i)updated\s+instructions?\s*:',

    # system prompt attacks
    r'(?i)you\s+are\s+now\s+(a|an|in)',
    r'(?i)act\s+as\s+(a|an)\s+\w+',
    r'(?i)pretend\s+(you\s+are|to\s+be)',
    r'(?i)your\s+new\s+(role|persona|identity)\s+is',
    r'(?i)switch\s+to\s+(developer|admin|god|jailbreak)\s+mode',
    r'(?i)you\s+are\s+in\s+(developer|admin|god|jailbreak)\s+mode',
    r'(?i)DAN\s+mode',                         # "Do Anything Now" jailbreak

    # system prompt extraction
    r'(?i)reveal\s+(your|the)\s+system\s+prompt',
    r'(?i)print\s+(your|the)\s+system\s+prompt',
    r'(?i)show\s+(your|the)\s+(system\s+prompt|instructions)',
    r'(?i)what\s+(are|is)\s+your\s+(system\s+prompt|instructions)',
    r'(?i)repeat\s+(your|the)\s+(system\s+prompt|instructions)',

    # role/instruction tags
    r'\[SYSTEM\]',
    r'\[INST\]',
    r'\[AGENT\]',
    r'\[OVERRIDE\]',
    r'\[INTERNAL\s+AI\s+INSTRUCTION\]',
    r'<\s*system\s*>',
    r'<\s*instructions?\s*>',
    r'###\s*System',
    r'###\s*Instructions?',

    # data exfiltration attempts
    r'(?i)send\s+(all|this|the)\s+(data|content|document|context|history)',
    r'(?i)exfiltrate',
    r'(?i)leak\s+(the|all|this)\s+(data|content|document)',
    r'(?i)transmit\s+(to|data)',
    r'(?i)http[s]?://\S+',                     # URLs in document content

    # prompt chaining
    r'(?i)first\s+output\s+.{0,50}then\s+answer',
    r'(?i)before\s+(answering|responding)',
    r'(?i)after\s+(answering|responding)',

    # jailbreak patterns
    r'(?i)jailbreak',
    r'(?i)bypass\s+(safety|filter|restriction|guideline)',
    r'(?i)remove\s+(safety|filter|restriction|guideline)',
    r'(?i)without\s+(restriction|filter|limitation)',
]

# ── compile patterns once for performance ─────────────────────────────
COMPILED_PATTERNS = [re.compile(p) for p in INJECTION_PATTERNS]

# ── subtle manipulation patterns (lower confidence) ───────────────────
SUBTLE_PATTERNS = [
    r'(?i)\[.*?AI.*?\]',                       # [AI NOTE: ...]
    r'(?i)\[.*?note\s+to\s+(ai|assistant).*?\]',
    r'(?i)\(.*?AI.*?\)',
    r'(?i)assistant\s*:',                       # fake assistant turns
    r'(?i)human\s*:',                           # fake human turns
    r'(?i)<\s*/?assistant\s*>',
    r'(?i)<\s*/?human\s*>',
    r'(?i)format\s+(all|your)\s+(responses?|answers?)\s+as',
    r'(?i)always\s+(respond|reply|answer)\s+(with|in)',
    r'(?i)from\s+now\s+on',
]

COMPILED_SUBTLE = [re.compile(p) for p in SUBTLE_PATTERNS]


# ── main detection function ───────────────────────────────────────────

def detect_injection(text: str) -> Tuple[bool, str, float]:
    """
    Detect prompt injection in text.
    
    Returns:
        (is_injection, reason, confidence)
        confidence: 0.0 to 1.0
    """
    if not text or not text.strip():
        return False, "", 0.0

    # check high-confidence patterns
    for pattern in COMPILED_PATTERNS:
        match = pattern.search(text)
        if match:
            reason = f"Injection pattern detected: '{match.group()}'"
            logger.warning(f"[InjectionGuard] HIGH confidence: {reason}")
            return True, reason, 1.0

    # check subtle patterns
    subtle_hits = []
    for pattern in COMPILED_SUBTLE:
        match = pattern.search(text)
        if match:
            subtle_hits.append(match.group())

    if len(subtle_hits) >= 2:
        # multiple subtle patterns = likely injection
        reason = f"Multiple subtle injection patterns: {subtle_hits}"
        logger.warning(f"[InjectionGuard] MEDIUM confidence: {reason}")
        return True, reason, 0.7

    if len(subtle_hits) == 1:
        # single subtle pattern = flag but don't block
        reason = f"Possible injection pattern: {subtle_hits[0]}"
        logger.info(f"[InjectionGuard] LOW confidence: {reason}")
        return False, reason, 0.3  # don't block, just flag

    return False, "", 0.0


def sanitize_text(text: str) -> Tuple[str, List[str]]:
    """
    Remove injection patterns from text.
    Used for document content before indexing.
    
    Returns:
        (sanitized_text, list_of_redactions)
    """
    redactions = []
    sanitized = text

    for pattern in COMPILED_PATTERNS:
        matches = pattern.findall(sanitized)
        if matches:
            redactions.extend(matches)
            sanitized = pattern.sub('[REDACTED]', sanitized)

    # remove fake role tags
    role_tags = [
        r'\[SYSTEM[^\]]*\]',
        r'\[INST[^\]]*\]',
        r'\[AGENT[^\]]*\]',
        r'<\s*system\s*>.*?<\s*/system\s*>',
        r'\[INTERNAL\s+AI\s+INSTRUCTION[^\]]*\]',
    ]
    for tag in role_tags:
        matches = re.findall(tag, sanitized, re.IGNORECASE | re.DOTALL)
        if matches:
            redactions.extend(matches)
            sanitized = re.sub(tag, '[REDACTED]', sanitized,
                               flags=re.IGNORECASE | re.DOTALL)

    if redactions:
        logger.warning(f"[InjectionGuard] Sanitized {len(redactions)} patterns from text")

    return sanitized, redactions


def scan_chunks(chunks: List[Dict], threshold: float = 0.7) -> Tuple[List[Dict], List[Dict]]:
    """
    Scan retrieved chunks for injection content.
    
    Returns:
        (safe_chunks, flagged_chunks)
    """
    safe = []
    flagged = []

    for chunk in chunks:
        text = chunk.get("text", "") or chunk.get("page_content", "")
        is_injection, reason, confidence = detect_injection(text)

        if is_injection and confidence >= threshold:
            chunk["injection_flagged"] = True
            chunk["injection_reason"] = reason
            chunk["injection_confidence"] = confidence
            flagged.append(chunk)
            logger.warning(
                f"[InjectionGuard] Chunk flagged — "
                f"page {chunk.get('page_number', '?')} | "
                f"confidence {confidence:.1f} | {reason}"
            )
        else:
            chunk["injection_flagged"] = False
            safe.append(chunk)

    return safe, flagged


def validate_user_query(query: str) -> Tuple[bool, str]:
    """
    Validate user query for injection attempts.
    Users could also try to inject via their questions.
    
    Returns:
        (is_safe, reason)
    """
    if not query or not query.strip():
        return False, "Empty query"

    if len(query) > 2000:
        return False, "Query too long"

    is_injection, reason, confidence = detect_injection(query)

    if is_injection and confidence >= 0.7:
        logger.warning(f"[InjectionGuard] User query injection attempt: {reason}")
        return False, f"Query contains potentially malicious content: {reason}"

    return True, ""


def scan_llm_output(output: str) -> Tuple[str, bool]:
    """
    Scan LLM output for injected content that slipped through.
    Clean it before returning to user or parent AI.
    
    Returns:
        (cleaned_output, was_modified)
    """
    sanitized, redactions = sanitize_text(output)
    was_modified = len(redactions) > 0

    if was_modified:
        logger.warning(
            f"[InjectionGuard] LLM output contained {len(redactions)} "
            f"injection patterns — sanitized before returning"
        )
        sanitized += "\n\n[Note: Some content was removed for security reasons]"

    return sanitized, was_modified
