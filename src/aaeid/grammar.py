"""Rule-based detectors for core AAE morphosyntax.

These are high-precision, low-recall patterns. They are a baseline and an
interpretable feature set, not a definition of AAE. Every rule has positive
and negative test sentences in tests/test_grammar.py; change a rule only
together with its tests.
"""
import re

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

SUBJ = r"(?:i|you|u|he|she|we|they|it|y'?all|people|dey|dem|that|this|my \w+)"
# -ing words that are not progressive verbs, to avoid "they bring", "the king".
NOT_VERB_ING = r"(?!(?:br|s|th|k|r|sw|st|w|spr|cl|fl|sl|str|d|z|p)ing\b)"
# Inside a clause: 0-3 tokens with no sentence punctuation in between.
WINDOW = r"(?:\s+[\w']+){0,3}?\s+"
# Clause start: beginning of text, after sentence punctuation, after a conjunction, or
# after a discourse marker that often opens a tweet ("lol he so funny").
# Stops object pronouns from matching ("I told you so", "give you my number").
CLAUSE = r"(?:^|[.!?,;:]\s*|\b(?:and|but|cause|cuz|cus|because|if|when|so|lol|lmao|smh|bruh|bro|man|nah|yeah|yea|ok|damn)\s+)"
NOT_DONE_OBJ = r"(?!(?:yet|it|that|with|but|right|last|first|out|about|at|what|just)\b)"

RULES = {
    # "he be working", "they always be talmbout"
    "habitual_be": rf"\b{SUBJ}\s+(?:always\s+|steady\s+|just\s+|still\s+)?be\s+\w+(?:ing|in'?)\b",
    # "don't nobody", "ain't got no", "can't find nothing"
    "neg_concord": rf"\b(?:ain'?t|aint|don'?t|dont|can'?t|cant|won'?t|wont|didn'?t|didnt|never|wasn'?t)"
                   rf"{WINDOW}(?:no|nothing|nothin|nun|nobody|none|nowhere|neither|nan)\b",
    # "ain't nobody", "can't nobody"
    "neg_inversion": r"\b(?:ain'?t|aint|can'?t|cant|don'?t|dont|won'?t|wont)\s+(?:nobody|nothing|nothin|no one|none)\b",
    # "she done left", "we done told you"
    "completive_done": rf"\b{SUBJ}\s+(?:been\s+)?done\s+{NOT_DONE_OBJ}\w+(?:ed|en)\b|\b{SUBJ}\s+done\s+(?:told|went|did|got|gave|said|ate|came|ran|saw|took|made|left|lost|bought|brought|forgot|seen|been)\b",
    # "he don't", "she don't care"
    "third_sg_dont": r"\b(?:he|she|it)\s+don'?t\b",
    # zero copula: "she gon", "they finna", "he at work", "you my dawg", "they going"
    "copula_absence": rf"{CLAUSE}(?:he|she|they|we|you|u|y'?all)\s+(?:gon|gonna|finna|bout|not|so|too|mad|at|on|here|there|my|{NOT_VERB_ING}\w{{2,}}(?:ing|in'))\b",
    # "finna", "fixin to", "bouta"
    "future_finna": r"\b(?:finna|fixin'? to|bouta|boutta)\b",
    "aint": r"\b(?:ain'?t|aint)\b",
}
RULE_NAMES = list(RULES)
_COMPILED = {k: re.compile(v, re.I) for k, v in RULES.items()}


def detect(text) -> dict:
    t = str(text).lower().replace("’", "'")
    return {k: bool(p.search(t)) for k, p in _COMPILED.items()}


def count(text) -> int:
    return sum(detect(text).values())


class GrammarFeatures(BaseEstimator, TransformerMixin):
    """Binary rule hits plus total count, usable inside a sklearn FeatureUnion."""

    def __init__(self, preprocessor=None):
        self.preprocessor = preprocessor

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for x in X:
            t = self.preprocessor(x) if self.preprocessor else x
            d = detect(t)
            rows.append([float(d[k]) for k in RULE_NAMES] + [sum(d.values()) / len(RULE_NAMES)])
        return np.asarray(rows)

    def get_feature_names_out(self, input_features=None):
        return np.array([f"rule:{k}" for k in RULE_NAMES] + ["rule:count"])
