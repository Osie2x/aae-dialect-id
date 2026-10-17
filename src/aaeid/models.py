"""Model zoo. Every model is a plain sklearn Pipeline built from a config entry."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from .grammar import GrammarFeatures
from .text import casualize, normalize, normalize_masked

PREPROCESSORS = {"raw": None, "norm": normalize, "norm_masked": normalize_masked}


def build(spec: dict, seed: int = 0) -> Pipeline:
    """spec keys: features (list of 'char', 'word', 'grammar'), preprocess, C."""
    pre = PREPROCESSORS[spec.get("preprocess", "raw")]
    parts = []
    if "char" in spec["features"]:
        parts.append(("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                                              sublinear_tf=True, preprocessor=pre, lowercase=pre is None)))
    if "word" in spec["features"]:
        parts.append(("word", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True,
                                              preprocessor=pre, lowercase=pre is None,
                                              token_pattern=r"(?u)\b[\w']+\b")))
    if "grammar" in spec["features"]:
        parts.append(("grammar", GrammarFeatures(preprocessor=pre or normalize)))
    clf = LogisticRegression(C=spec.get("C", 4.0), max_iter=5000, class_weight="balanced",
                             solver="liblinear", random_state=seed)
    return Pipeline([("features", FeatureUnion(parts)), ("clf", clf)])


def training_set(df, spec):
    """Positives: tweets natives rated natural AAE. Negatives: the LLM SAE translation of
    every training tweet. With casual_negatives, also a casual rewrite of each translation
    (lowercase, no punctuation, "u", "im", "ppl"), so informality alone stops predicting AAE.
    With hard_negatives, also tweets natives rejected and
    back-translations natives rated performed/imitated."""
    nat = df["original_naturalness"] == "natural"
    X = list(df.loc[nat, "aae_text"]) + list(df["sae_text"])
    y = [1] * int(nat.sum()) + [0] * len(df)
    if spec.get("casual_negatives"):
        X += [casualize(t) for t in df["sae_text"]]
        y += [0] * len(df)
    if spec.get("hard_negatives"):
        rej = df["original_naturalness"] == "unnatural_or_noisy"
        perf = (df["back_naturalness"] == "unnatural_or_imitated") & df["aae_back_text"].notna()
        X += list(df.loc[rej, "aae_text"]) + list(df.loc[perf, "aae_back_text"])
        y += [0] * int(rej.sum() + perf.sum())
    return X, np.asarray(y)


def top_features(model, k=25):
    names = model.named_steps["features"].get_feature_names_out()
    coef = model.named_steps["clf"].coef_[0]
    order = np.argsort(-coef)
    pos = [(str(names[i]), round(float(coef[i]), 3)) for i in order[:k]]
    neg = [(str(names[i]), round(float(coef[i]), 3)) for i in order[::-1][:k]]
    rules = [(str(n), round(float(c), 3)) for n, c in zip(names, coef) if "rule:" in str(n)]
    return {"most_aae": pos, "most_sae": neg, "grammar_rules": rules}
