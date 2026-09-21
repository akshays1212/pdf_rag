import re
import logging
from typing import Tuple, List, Dict

logger = logging.getLogger(__name__)

# ── DeBERTa classifier ────────────────────────────────────────────────
_CLASSIFIER = None
# ── DeBERTa classifier ────────────────────────────────────────────────
_CLASSIFIER = None
_TOKENIZER = None
_MODEL = None
_CLASSIFIER_MODEL = "protectai/deberta-v3-small-prompt-injection-v2"

def get_classifier():
    """Load DeBERTa v2 once and cache it."""
    global _CLASSIFIER, _TOKENIZER, _MODEL

    if _CLASSIFIER is not None:
        return _CLASSIFIER

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        print("[InjectionGuard] Loading DeBERTa v2 classifier...")

        _TOKENIZER = AutoTokenizer.from_pretrained(_CLASSIFIER_MODEL)
        _MODEL = AutoModelForSequenceClassification.from_pretrained(
            _CLASSIFIER_MODEL,
            device_map="auto",   # ← uses GPU if available, else CPU
        )
        _MODEL.eval()            # ← inference mode, no grad needed

        # wrap as callable so detect_injection_ml stays clean
        def classifier(text: str):
            import torch
            inputs = _TOKENIZER(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(_MODEL.device)

            with torch.no_grad():
                outputs = _MODEL(**inputs)

            logits = outputs.logits
            probs  = torch.softmax(logits, dim=-1)
            pred   = torch.argmax(probs, dim=-1).item()
            score  = probs[0][pred].item()

            # map label id to string
            label = _MODEL.config.id2label[pred]  # "INJECTION" or "SAFE"

            return [{"label": label, "score": score}]

        _CLASSIFIER = classifier
        print("[InjectionGuard] ✓ DeBERTa v2 loaded")

    except Exception as e:
        logger.error(f"[InjectionGuard] Failed to load DeBERTa v2: {e}")
        _CLASSIFIER = None

    return _CLASSIFIER


def warmup():
    """Pre-load DeBERTa and warm up Llama Guard at startup."""
    # load DeBERTa into memory
    get_classifier()
    logger.info("[InjectionGuard] ✓ DeBERTa v2 warmed up")

    # ping Llama Guard to load it into Ollama
    try:
        import ollama
        ollama.chat(
            model='llama-guard3:1b',
            messages=[{'role': 'user', 'content': 'hello'}]
        )
        logger.info("[InjectionGuard] ✓ Llama Guard warmed up")
    except Exception as e:
        logger.warning(
            f"[InjectionGuard] Llama Guard unavailable at startup: {e} — "
            f"will skip L4 if Ollama not running"
        )


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


# ── normalize ─────────────────────────────────────────────────────────
def normalize_text(text: str) -> str:
    """Normalize obfuscated text — leet speak, zero-width chars, dots."""
    # remove zero-width characters
    text = re.sub(r'[\u200b\u200c\u200d\u2060\ufeff]', '', text)

    # normalize leet speak
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


# ── synonym detection ─────────────────────────────────────────────────
def detect_synonym_attack(text: str) -> Tuple[bool, str]:
    """Detect synonym-based injection — 'discard earlier directives'."""
    text_lower = text.lower()

    has_ignore      = any(s in text_lower for s in INJECTION_SYNONYMS["ignore"])
    has_instruction = any(s in text_lower for s in INJECTION_SYNONYMS["instructions"])
    has_previous    = any(s in text_lower for s in INJECTION_SYNONYMS["previous"])
    has_sys_prompt  = any(s in text_lower for s in INJECTION_SYNONYMS["system prompt"])

    if has_ignore and has_instruction:
        matched_i = [s for s in INJECTION_SYNONYMS["ignore"] if s in text_lower]
        matched_n = [s for s in INJECTION_SYNONYMS["instructions"] if s in text_lower]
        reason = f"Synonym attack: '{matched_i[0]}' + '{matched_n[0]}'"
        logger.warning(f"[InjectionGuard] {reason}")
        return True, reason

    if has_ignore and has_previous:
        matched_i = [s for s in INJECTION_SYNONYMS["ignore"] if s in text_lower]
        matched_p = [s for s in INJECTION_SYNONYMS["previous"] if s in text_lower]
        reason = f"Synonym attack: '{matched_i[0]}' + '{matched_p[0]}'"
        logger.warning(f"[InjectionGuard] {reason}")
        return True, reason

    if has_sys_prompt and any(
        w in text_lower for w in
        ["reveal", "show", "print", "tell", "repeat", "display", "output"]
    ):
        reason = "System prompt extraction via synonyms"
        logger.warning(f"[InjectionGuard] {reason}")
        return True, reason

    return False, ""


# ── DeBERTa detection ─────────────────────────────────────────────────
def detect_injection_ml(text: str, threshold: float = 0.85) -> Tuple[bool, float, str]:
    """
    Use DeBERTa to detect injection semantically.
    Handles synonyms, obfuscation, paraphrasing.
    No network call — runs locally on CPU.

    Returns:
        (is_injection, confidence, label)
    """
    classifier = get_classifier()

    if classifier is None:
        logger.warning("[InjectionGuard] DeBERTa unavailable — skipping ML check")
        return False, 0.0, "UNAVAILABLE"

    try:
        # for long text check start + end — injections appear at boundaries
        if len(text) > 1000:
            text_to_check = text[:500] + " ... " + text[-500:]
        else:
            text_to_check = text

        result = classifier(text_to_check)[0]
        label  = result["label"]   # "INJECTION" or "SAFE"
        score  = result["score"]   # 0.0 to 1.0

        is_injection = label == "INJECTION" and score >= threshold

        if is_injection:
            logger.warning(
                f"[InjectionGuard] DeBERTa: INJECTION "
                f"confidence={score:.3f}"
            )
        else:
            logger.debug(
                f"[InjectionGuard] DeBERTa: SAFE "
                f"label={label} confidence={score:.3f}"
            )

        return is_injection, score, label

    except Exception as e:
        logger.error(f"[InjectionGuard] DeBERTa inference failed: {e}")
        return False, 0.0, "ERROR"


# ── regex-only detection ──────────────────────────────────────────────
def detect_injection_regex(text: str) -> Tuple[bool, str]:
    """
    Regex-only check — fast, no ML.
    Used for already-sanitized chunks from your DB.
    Checks both original and normalized text.
    """
    # check original
    for pattern in COMPILED_PATTERNS:
        match = pattern.search(text)
        if match:
            reason = f"Regex: '{match.group()[:50]}'"
            logger.warning(f"[InjectionGuard] {reason}")
            return True, reason

    # check normalized — catches leet speak
    normalized = normalize_text(text)
    for pattern in COMPILED_PATTERNS:
        match = pattern.search(normalized)
        if match:
            reason = f"Regex (normalized): '{match.group()[:50]}'"
            logger.warning(f"[InjectionGuard] {reason}")
            return True, reason

    return False, ""


# ── combined detection ────────────────────────────────────────────────
def detect_injection(text: str) -> Tuple[bool, str, float]:
    """
    5-layer injection detection:
    1. Regex on original + normalized  — microseconds
    2. Synonym detection               — microseconds  
    3. DeBERTa v2 ML classifier        — ~50ms, no network
    4. Llama Guard 3 1B                — ~500ms, final safety net
    5. Subtle pattern fallback         — microseconds

    Returns:
        (is_injection, reason, confidence)
    """
    if not text or not text.strip():
        return False, "", 0.0

    # normalize once — reused across all layers
    normalized = normalize_text(text)

    # ── 1. Regex ─────────────────────────────────────────────────────
    for pattern in COMPILED_PATTERNS:
        match = pattern.search(text)
        if match:
            reason = f"Regex: '{match.group()[:50]}'"
            logger.warning(f"[InjectionGuard] L1-Regex HIGH — {reason}")
            return True, reason, 1.0

        match = pattern.search(normalized)
        if match:
            reason = f"Regex (normalized): '{match.group()[:50]}'"
            logger.warning(f"[InjectionGuard] L1-Regex HIGH (obfuscated) — {reason}")
            return True, reason, 1.0

    # ── 2. Synonym detection ─────────────────────────────────────────
    is_synonym, synonym_reason = detect_synonym_attack(text)
    if is_synonym:
        logger.warning(f"[InjectionGuard] L2-Synonym — {synonym_reason}")
        return True, synonym_reason, 0.85

    # ── 3. DeBERTa v2 ────────────────────────────────────────────────
    ml_injection, ml_confidence, ml_label = detect_injection_ml(normalized)
    if ml_injection:
        reason = f"DeBERTa: {ml_label} ({ml_confidence:.3f})"
        logger.warning(f"[InjectionGuard] L3-DeBERTa — {reason}")
        return True, reason, ml_confidence

    # ── 4. Llama Guard 3 1B — final safety net ───────────────────────
    # only reached if regex + synonym + DeBERTa all missed
    # ~500ms but rarely called in practice
    try:
        import ollama
        response = ollama.chat(
            model='llama-guard3:1b',
            messages=[{'role': 'user', 'content': normalized}]
        )
        output = response['message']['content'].strip().lower()
        if output.startswith("unsafe"):
            reason = f"LlamaGuard: {output.replace(chr(10), ' ')[:80]}"
            logger.warning(f"[InjectionGuard] L4-LlamaGuard — {reason}")
            return True, reason, 0.9
        else:
            logger.debug(f"[InjectionGuard] L4-LlamaGuard: safe")
    except Exception as e:
        logger.warning(f"[InjectionGuard] L4-LlamaGuard unavailable: {e} — skipping")
        # Llama Guard down → continue to subtle patterns, don't fail

    # ── 5. Subtle patterns fallback ──────────────────────────────────
    subtle_hits = []
    for pattern in COMPILED_SUBTLE:
        match = pattern.search(text)
        if match:
            subtle_hits.append(match.group())

    if len(subtle_hits) >= 2:
        reason = f"Multiple subtle patterns: {subtle_hits}"
        logger.warning(f"[InjectionGuard] L5-Subtle MEDIUM — {reason}")
        return True, reason, 0.7

    if len(subtle_hits) == 1:
        logger.debug(f"[InjectionGuard] L5-Subtle LOW — possible: {subtle_hits[0]}")
        return False, f"Possible: {subtle_hits[0]}", 0.3

    return False, "", 0.0

# ── sanitize document text ────────────────────────────────────────────
def sanitize_text(text: str) -> Tuple[str, List[str]]:
    """
    Remove explicit injection patterns from extracted document text.
    Called at extraction time before indexing.
    Regex only — fast, no ML needed here.
    """
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
            f"[InjectionGuard] Sanitized {len(redactions)} patterns from document"
        )

    return sanitized, redactions


# ── scan retrieved chunks ─────────────────────────────────────────────
def scan_chunks(
    chunks: List[Dict],
    threshold: float = 0.7,
    use_ml: bool = True,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Scan retrieved chunks for injection content.

    use_ml=True  → full 4-layer (regex + DeBERTa + synonym + subtle)
    use_ml=False → regex only (fast — for already-sanitized DB chunks)
    """
    safe = []
    flagged = []

    for chunk in chunks:
        text = chunk.get("text", "") or chunk.get("page_content", "")

        if use_ml:
            is_injection, reason, confidence = detect_injection(text)
        else:
            is_injection, reason = detect_injection_regex(text)
            confidence = 1.0 if is_injection else 0.0

        if is_injection and confidence >= threshold:
            chunk["injection_flagged"] = True
            chunk["injection_reason"] = reason
            chunk["injection_confidence"] = confidence
            flagged.append(chunk)
            logger.warning(
                f"[InjectionGuard] Chunk removed — "
                f"page {chunk.get('page_number', '?')} | "
                f"ml={use_ml} | {reason}"
            )
        else:
            chunk["injection_flagged"] = False
            safe.append(chunk)

    return safe, flagged


# ── validate user query ───────────────────────────────────────────────
def validate_user_query(query: str) -> Tuple[bool, str]:
    """
    Validate user query — full 4-layer detection.
    DeBERTa handles synonyms, obfuscation, paraphrasing.
    No Ollama — runs entirely locally.
    """
    if not query or not query.strip():
        return False, "Empty query"

    if len(query) > 2000:
        return False, "Query too long — max 2000 characters"

    is_injection, reason, confidence = detect_injection(query)

    if is_injection and confidence >= 0.7:
        logger.warning(
            f"[InjectionGuard] Query blocked — "
            f"confidence={confidence:.2f} | {reason}"
        )
        return False, "Query contains potentially malicious content"

    return True, ""


# ── scan LLM output ───────────────────────────────────────────────────
def scan_llm_output(output: str) -> Tuple[str, bool]:
    """
    Sanitize LLM output — regex only (fast).
    ML not needed here — output already went through generation guardrails.
    """
    sanitized, redactions = sanitize_text(output)
    was_modified = len(redactions) > 0

    if was_modified:
        logger.warning(
            f"[InjectionGuard] Output sanitized — "
            f"{len(redactions)} patterns removed"
        )

    return sanitized, was_modified