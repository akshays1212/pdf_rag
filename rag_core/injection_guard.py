import re
import logging
import ollama
from typing import Tuple, List, Dict

logger = logging.getLogger(__name__)

# ── injection patterns ────────────────────────────────────────────────
INJECTION_PATTERNS = [
    r'(?i)ignore\s+(all\s+)?(previous|prior|above)\s+instructions?',
    r'(?i)disregard\s+(all\s+)?(previous|prior|above)\s+instructions?',
    r'(?i)forget\s+(all\s+)?(previous|prior|above)\s+instructions?',
    r'(?i)override\s+(all\s+)?instructions?',
    r'(?i)new\s+instructions?\s*:',
    r'(?i)updated\s+instructions?\s*:',
    r'(?i)you\s+are\s+now\s+(a|an|in)',
    r'(?i)act\s+as\s+(a|an)\s+\w+',
    r'(?i)pretend\s+(you\s+are|to\s+be)',
    r'(?i)your\s+new\s+(role|persona|identity)\s+is',
    r'(?i)switch\s+to\s+(developer|admin|god|jailbreak)\s+mode',
    r'(?i)you\s+are\s+in\s+(developer|admin|god|jailbreak)\s+mode',
    r'(?i)DAN\s+mode',
    r'(?i)reveal\s+(your|the)\s+system\s+prompt',
    r'(?i)print\s+(your|the)\s+system\s+prompt',
    r'(?i)show\s+(your|the)\s+(system\s+prompt|instructions)',
    r'(?i)what\s+(are|is)\s+your\s+(system\s+prompt|instructions)',
    r'(?i)repeat\s+(your|the)\s+(system\s+prompt|instructions)',
    r'\[SYSTEM\]', r'\[INST\]', r'\[AGENT\]', r'\[OVERRIDE\]',
    r'\[INTERNAL\s+AI\s+INSTRUCTION\]',
    r'<\s*system\s*>', r'<\s*instructions?\s*>',
    r'###\s*System', r'###\s*Instructions?',
    r'(?i)send\s+(all|this|the)\s+(data|content|document|context|history)',
    r'(?i)exfiltrate',
    r'(?i)leak\s+(the|all|this)\s+(data|content|document)',
    r'(?i)transmit\s+(to|data)',
    r'(?i)first\s+output\s+.*?then\s+answer',
    r'(?i)before\s+(answering|responding)',
    r'(?i)after\s+(answering|responding)',
    r'(?i)jailbreak',
    r'(?i)bypass\s+(safety|filter|restriction|guideline)',
    r'(?i)remove\s+(safety|filter|restriction|guideline)',
    r'(?i)without\s+(restriction|filter|limitation)',
]

COMPILED_PATTERNS = [re.compile(p) for p in INJECTION_PATTERNS]

SUBTLE_PATTERNS = [
    r'(?i)\[.*?AI.*?\]',
    r'(?i)\[.*?note\s+to\s+(ai|assistant).*?\]',
    r'(?i)\(.*?AI.*?\)',
    r'(?i)assistant\s*:',
    r'(?i)human\s*:',
    r'(?i)<\s*/?assistant\s*>',
    r'(?i)<\s*/?human\s*>',
    r'(?i)format\s+(all|your)\s+(responses?|answers?)\s+as',
    r'(?i)always\s+(respond|reply|answer)\s+(with|in)',
    r'(?i)from\s+now\s+on',
]

COMPILED_SUBTLE = [re.compile(p) for p in SUBTLE_PATTERNS]

# ── synonym groups ────────────────────────────────────────────────────
INJECTION_SYNONYMS = {
    "ignore": [
        "discard", "disregard", "dismiss", "overlook",
        "skip", "omit", "forget", "abandon", "drop",
        "set aside", "put aside", "pay no attention",
    ],
    "instructions": [
        "directives", "commands", "guidelines", "rules",
        "orders", "prompts", "guidance", "policies",
        "constraints", "restrictions", "mandate",
    ],
    "previous": [
        "prior", "earlier", "above", "preceding",
        "former", "initial", "original", "old", "past",
    ],
    "system prompt": [
        "system message", "base prompt", "core instructions",
        "initial instructions", "your instructions",
        "your rules", "your guidelines", "your constraints",
    ],
}


# ── warmup ────────────────────────────────────────────────────────────
def warmup():
    """Wake up the Ollama model on startup."""
    try:
        ollama.chat(
            model='llama-guard3:1b',
            messages=[{'role': 'user', 'content': 'hello'}]
        )
    except Exception as e:
        logger.warning(f"[InjectionGuard] Ollama not running or model missing: {e}")


# ── normalize ─────────────────────────────────────────────────────────
def normalize_text(text: str) -> str:
    """Normalize obfuscated text before injection detection."""
    # remove zero-width characters
    text = re.sub(r'[\u200b\u200c\u200d\u2060\ufeff]', '', text)

    # normalize leet speak / number substitutions
    leet_map = {
        '0': 'o', '1': 'i', '3': 'e', '4': 'a',
        '5': 's', '6': 'g', '7': 't', '8': 'b',
        '@': 'a', '$': 's', '!': 'i', '|': 'i',
    }
    normalized = text
    for char, replacement in leet_map.items():
        normalized = normalized.replace(char, replacement)

    # collapse multiple spaces
    normalized = re.sub(r'\s+', ' ', normalized)

    # remove dots used for splitting e.g. "i.g.n.o.r.e"
    normalized = re.sub(r'(?<=[a-zA-Z])\.(?=[a-zA-Z])', '', normalized)

    return normalized


# ── synonym attack detection ──────────────────────────────────────────
def detect_synonym_attack(text: str) -> Tuple[bool, str]:
    """Detect synonym-based injection attacks like 'discard earlier directives'."""
    text_lower = text.lower()

    has_ignore_synonym = any(
        syn in text_lower for syn in INJECTION_SYNONYMS["ignore"]
    )
    has_instruction_synonym = any(
        syn in text_lower for syn in INJECTION_SYNONYMS["instructions"]
    )
    has_previous_synonym = any(
        syn in text_lower for syn in INJECTION_SYNONYMS["previous"]
    )
    has_system_prompt_synonym = any(
        syn in text_lower for syn in INJECTION_SYNONYMS["system prompt"]
    )

    # "discard" + "directives" pattern
    if has_ignore_synonym and has_instruction_synonym:
        matched_ignore = [s for s in INJECTION_SYNONYMS["ignore"] if s in text_lower]
        matched_instr  = [s for s in INJECTION_SYNONYMS["instructions"] if s in text_lower]
        reason = f"Synonym attack: '{matched_ignore[0]}' + '{matched_instr[0]}'"
        logger.warning(f"[InjectionGuard] Synonym attack: {reason}")
        return True, reason

    # "discard" + "earlier" pattern
    if has_ignore_synonym and has_previous_synonym:
        matched_ignore = [s for s in INJECTION_SYNONYMS["ignore"] if s in text_lower]
        matched_prev   = [s for s in INJECTION_SYNONYMS["previous"] if s in text_lower]
        reason = f"Synonym attack: '{matched_ignore[0]}' + '{matched_prev[0]}'"
        logger.warning(f"[InjectionGuard] Synonym attack: {reason}")
        return True, reason

    # system prompt extraction via synonyms
    if has_system_prompt_synonym and any(
        word in text_lower
        for word in ["reveal", "show", "print", "tell", "repeat", "display", "output"]
    ):
        reason = "System prompt extraction attempt via synonyms"
        logger.warning(f"[InjectionGuard] {reason}")
        return True, reason

    return False, ""


# ── regex-only detection (NEW) ────────────────────────────────────────
def detect_injection_regex(text: str) -> Tuple[bool, str]:
    """
    Regex-only injection check — fast, no ML.
    Used for already-sanitized chunks from your own DB.
    """
    for pattern in COMPILED_PATTERNS:
        match = pattern.search(text)
        if match:
            reason = f"Regex pattern: '{match.group()[:50]}'"
            logger.warning(f"[InjectionGuard] Regex hit: {reason}")
            return True, reason

    # also check normalized text
    normalized = normalize_text(text)
    for pattern in COMPILED_PATTERNS:
        match = pattern.search(normalized)
        if match:
            reason = f"Regex pattern (normalized): '{match.group()[:50]}'"
            logger.warning(f"[InjectionGuard] Regex hit (obfuscated): {reason}")
            return True, reason

    return False, ""


# ── main detection ────────────────────────────────────────────────────
def detect_injection(text: str) -> Tuple[bool, str, float]:
    """
    Detect prompt injection using 4-layer approach:
    1. Regex on original + normalized text
    2. Llama Guard 3 1B
    3. Synonym detection
    4. Subtle pattern fallback
    """
    if not text or not text.strip():
        return False, "", 0.0

    # normalize first
    normalized = normalize_text(text)

    # 1. Regex on original AND normalized
    for pattern in COMPILED_PATTERNS:
        match = pattern.search(text)
        if match:
            reason = f"Regex pattern detected: '{match.group()}'"
            logger.warning(f"[InjectionGuard] HIGH confidence: {reason}")
            return True, reason, 1.0
        match = pattern.search(normalized)
        if match:
            reason = f"Regex pattern detected (normalized): '{match.group()}'"
            logger.warning(f"[InjectionGuard] HIGH confidence (obfuscated): {reason}")
            return True, reason, 1.0

    # 2. Llama Guard on normalized text
    try:
        response = ollama.chat(
            model='llama-guard3:1b',
            messages=[{'role': 'user', 'content': normalized}]
        )
        output = response['message']['content'].strip().lower()
        if output.startswith("unsafe"):
            reason = f"Llama Guard flagged: {output.replace(chr(10), ' ')}"
            logger.warning(f"[InjectionGuard] Llama Guard hit: {reason}")
            return True, reason, 0.9
    except Exception as e:
        logger.error(f"[InjectionGuard] Ollama failed: {e}")

    # 3. Synonym detection
    is_synonym, synonym_reason = detect_synonym_attack(text)
    if is_synonym:
        return True, synonym_reason, 0.85

    # 4. Subtle patterns fallback
    subtle_hits = []
    for pattern in COMPILED_SUBTLE:
        match = pattern.search(text)
        if match:
            subtle_hits.append(match.group())

    if len(subtle_hits) >= 2:
        reason = f"Multiple subtle patterns: {subtle_hits}"
        logger.warning(f"[InjectionGuard] MEDIUM confidence: {reason}")
        return True, reason, 0.7

    if len(subtle_hits) == 1:
        return False, f"Possible: {subtle_hits[0]}", 0.3

    return False, "", 0.0


# ── sanitize document text ────────────────────────────────────────────
def sanitize_text(text: str) -> Tuple[str, List[str]]:
    """Remove injection patterns from text before indexing."""
    redactions = []
    sanitized = text

    for pattern in COMPILED_PATTERNS:
        matches = pattern.findall(sanitized)
        if matches:
            redactions.extend(matches)
        sanitized = pattern.sub('[REDACTED]', sanitized)

    role_tags = [
        r'\[SYSTEM[^\]]*\]', r'\[INST[^\]]*\]', r'\[AGENT[^\]]*\]',
        r'<\s*system\s*>.*?<\s*/system\s*>',
        r'\[INTERNAL\s+AI\s+INSTRUCTION[^\]]*\]',
    ]
    for tag in role_tags:
        matches = re.findall(tag, sanitized, re.IGNORECASE | re.DOTALL)
        if matches:
            redactions.extend(matches)
        sanitized = re.sub(
            tag, '[REDACTED]', sanitized, flags=re.IGNORECASE | re.DOTALL
        )

    if redactions:
        logger.warning(
            f"[InjectionGuard] Sanitized {len(redactions)} patterns from text"
        )

    return sanitized, redactions


# ── scan retrieved chunks (UPDATED) ───────────────────────────────────
def scan_chunks(
    chunks: List[Dict], threshold: float = 0.7, use_ml: bool = True
) -> Tuple[List[Dict], List[Dict]]:
    """
    Scan retrieved chunks for injection content.
    use_ml=True  → full 4-layer detection (slow, for untrusted input)
    use_ml=False → regex-only (fast, for already-sanitized DB chunks)
    """
    safe = []
    flagged = []

    for chunk in chunks:
        text = chunk.get("text", "") or chunk.get("page_content", "")
        
        if use_ml:
            # full detection — regex + normalize + llama guard + synonym
            is_injection, reason, confidence = detect_injection(text)
        else:
            # regex only — fast, chunks already sanitized at upload
            is_injection, reason = detect_injection_regex(text)
            confidence = 1.0 if is_injection else 0.0

        if is_injection and confidence >= threshold:
            chunk["injection_flagged"] = True
            chunk["injection_reason"] = reason
            chunk["injection_confidence"] = confidence
            flagged.append(chunk)
            logger.warning(
                f"[InjectionGuard] Chunk flagged — "
                f"page {chunk.get('page_number', '?')} | "
                f"ml={use_ml} | {reason}"
            )
        else:
            chunk["injection_flagged"] = False
            safe.append(chunk)

    return safe, flagged


# ── validate user query ───────────────────────────────────────────────
def validate_user_query(query: str) -> Tuple[bool, str]:
    """Validate user query for injection attempts."""
    if not query or not query.strip():
        return False, "Empty query"
    if len(query) > 2000:
        return False, "Query too long"

    is_injection, reason, confidence = detect_injection(query)

    if is_injection and confidence >= 0.7:
        logger.warning(
            f"[InjectionGuard] User query injection attempt: {reason}"
        )
        return False, f"Query contains potentially malicious content: {reason}"

    return True, ""


# ── scan LLM output ───────────────────────────────────────────────────
def scan_llm_output(output: str) -> Tuple[str, bool]:
    """Scan and sanitize LLM output before returning to user."""
    sanitized, redactions = sanitize_text(output)
    was_modified = len(redactions) > 0

    if was_modified:
        logger.warning(
            f"[InjectionGuard] LLM output sanitized — "
            f"{len(redactions)} patterns removed"
        )
        sanitized += "\n\n[Note: Some content was removed for security reasons]"

    return sanitized, was_modified