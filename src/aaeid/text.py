"""Text normalization and slang masking.

Normalization removes platform artifacts (URLs, @mentions, '#', digits,
repeated punctuation, emoji) that let a Twitter-trained model separate
"tweet vs. not tweet" instead of "AAE vs. not AAE".
"""
import re

# Lexical items that are common in AAE but have diffused into general
# internet English. Masking them forces the model to rely on grammar.
SLANG_WORDS = [
    "ain'?t", "aint", "finna", "ion", "lowkey", "imma", "ima", "i'?ma", "gon", "gonna",
    "tryna", "bout", "nah", "yall", "y'all", "lol", "lmao", "smh", "fr", "bruh",
]
# General internet spellings. Not AAE-specific, but LLM "SAE" translations never use them,
# so a model can separate real tweets from translations with these alone.
INTERNET_WORDS = ["u", "ur", "im", "ppl", "omg", "idk", "tbh", "rn", "wtf", "af", "ya", "yo"]
# Slurs and profanity are masked too, so the model cannot use them as dialect cues.
PROFANITY = [r"shit\w*", r"nigg\w*", r"fuck\w*", r"bitch\w*"]
SLANG_RE = re.compile(r"\b(" + "|".join(SLANG_WORDS + INTERNET_WORDS + PROFANITY) + r")\b", re.I)

_URL = re.compile(r"https?://\S+|www\.\S+")
_MENTION = re.compile(r"@\w+")
_NUM = re.compile(r"\d+(?:[.,]\d+)?")
_REPEAT_PUNCT = re.compile(r"([!?.,'\"*~<>&:;()\-])\1+")
_NON_WORD = re.compile(r"[^\w\s']")
_SPACE = re.compile(r"\s+")


def normalize(text) -> str:
    """Lowercase and strip platform artifacts. Apostrophes are kept (ain't, y'all)."""
    t = str(text).lower().replace("\u2019", "'")
    t = _URL.sub(" ", t)
    t = _MENTION.sub(" ", t)
    t = t.replace("#", " ")
    t = _NUM.sub(" 0 ", t)
    t = _REPEAT_PUNCT.sub(r"\1", t)
    t = _NON_WORD.sub(" ", t)
    return _SPACE.sub(" ", t).strip()


def mask_slang(text) -> str:
    return _SPACE.sub(" ", SLANG_RE.sub(" ", str(text))).strip()


def normalize_masked(text) -> str:
    return mask_slang(normalize(text))


# Rewrites that make formal SAE look like a casual post without adding dialect features.
_CASUAL = [
    (r"\byou're\b", "ur"), (r"\byour\b", "ur"), (r"\byou\b", "u"), (r"\bi'm\b", "im"),
    (r"\bpeople\b", "ppl"), (r"\bgoing to\b", "gonna"), (r"\bwant to\b", "wanna"),
    (r"\bgot to\b", "gotta"), (r"\bdon't\b", "dont"), (r"\bcan't\b", "cant"),
    (r"\bokay\b", "ok"), (r"\bthough\b", "tho"), (r"\bbecause\b", "cuz"), (r"\bkind of\b", "kinda"),
]
_CASUAL = [(re.compile(a), b) for a, b in _CASUAL]


def casualize(text) -> str:
    """Formal SAE -> casual SAE: lowercase, no punctuation, common internet spellings."""
    t = _NON_WORD.sub(" ", str(text).lower().replace("\u2019", "'"))
    for pat, rep in _CASUAL:
        t = pat.sub(rep, t)
    return _SPACE.sub(" ", t).strip()


def n_words(text) -> int:
    return len(str(text).split())


def truncate_words(text, k: int) -> str:
    return " ".join(str(text).split()[:k])


def match_key(text) -> str:
    """Key for duplicate detection: letters and digits only."""
    return re.sub(r"[^a-z0-9]", "", str(text).lower())
