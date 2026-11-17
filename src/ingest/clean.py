"""Text cleaning and untrusted-content scanning."""
import re
import unicodedata

_LIGATURES = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl"}

# Patterns that look like instructions aimed at an LLM. Chunks that match are
# FLAGGED (not deleted): papers may legitimately discuss prompt injection.
_INJECTION_PATTERNS = [
    r"ignore (all |any )?(previous|prior|above) (instructions|prompts?)",
    r"disregard (the |all )?(system|previous) (prompt|instructions)",
    r"you are now\b",
    r"(reveal|print|show) (your |the )?(system prompt|hidden instructions)",
    r"\bact as (an? )?(unrestricted|jailbroken|different)",
    r"do not (cite|mention) (any )?sources",
    r"<\s*/?\s*(system|assistant)\s*>",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)


def clean_text(text: str) -> str:
    """Normalize unicode, fix ligatures, de-hyphenate line breaks, collapse whitespace."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    for k, v in _LIGATURES.items():
        text = text.replace(k, v)
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)      # de-hyphenate
    text = re.sub(r"[ \t]*\n[ \t]*", " ", text)              # join lines
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)  # control chars
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def scan_injection(text: str) -> bool:
    """True if text contains instruction-like content aimed at an LLM."""
    return bool(_INJECTION_RE.search(text or ""))
