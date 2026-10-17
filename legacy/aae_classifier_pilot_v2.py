import re, sys, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, make_union
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, accuracy_score
U = sys.argv[1]; rng = np.random.RandomState(0)
F = ["30576fe1-annotator_1_annotations_2026-05-19T04-05-02.317Z_6.csv",
     "ef443431-annotator_2_annotations_300_6.csv",
     "15cd2aa7-annotator_3_annotations_300_unique_7.csv"]
ann = pd.concat([pd.read_csv(f"{U}/{f}").assign(ann=i+1) for i, f in enumerate(F)], ignore_index=True)
ann = ann.dropna(subset=["aae_text", "sae_text"]).drop_duplicates("aae_text").reset_index(drop=True)
ann["nat"] = ann.original_naturalness
print("annotated tweets:", len(ann), ann.nat.value_counts().to_dict())

def model():
    return make_pipeline(make_union(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True),
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        LogisticRegression(C=4, max_iter=3000))

# grammar-only baseline: core AAE morphosyntax, no slang words
SUBJ = r"(i|you|u|he|she|we|they|it|y'?all|people|dey|that|this|my \w+)"
G = {
 "habitual_be": rf"\b{SUBJ}\s+(always\s+|steady\s+|just\s+)?be\s+\w+(in|ing|in')\b",
 "neg_concord": r"\b(ain'?t|aint|don'?t|dont|can'?t|cant|won'?t|didn'?t|never|wasn'?t)\b(\W+\w+){0,3}?\W+(no|nothing|nothin|nobody|none|nowhere|neither|nun|nan)\b",
 "neg_inversion": r"\b(ain'?t|aint|can'?t|don'?t|won'?t)\s+(nobody|nothing|nothin|no one|none)\b",
 "completive_done": rf"\b{SUBJ}\s+done\s+\w+",
 "third_sg_dont": r"\b(he|she|it)\s+don'?t\b",
 "copula_absence": rf"\b(he|she|they|we|you|u|y'?all)\s+(not|gon|finna|so|too|always|just|still|bout|my|on|at|in|real|mad|\w+in'?|\w+ing)\b",
 "future_finna": r"\b(finna|fixin to|bouta|boutta)\b",
 "ain_t": r"\b(ain'?t|aint)\b",
}
def gram(t):
    t = str(t).lower(); return sum(bool(re.search(p, t)) for p in G.values())
SLANG = r"\b(ain'?t|aint|finna|ion|lowkey|imma|ima|i'?ma|gon|gonna|tryna|bout|nah|yall|y'all|lol|lmao|smh|fr|bruh|shit|nigga\w*)\b"
mask = lambda t: re.sub(SLANG, "", str(t), flags=re.I)

# ---- Experiment 1: in-domain, 5-fold CV over items ----
nat = ann.nat == "natural"; unn = ann.nat == "unnatural_or_noisy"; some = ann.nat == "somewhat_natural"
acc, auc_cv, unn_hit, nat_hit = [], [], [], []
oof = np.full(len(ann), np.nan)
for tr, te in KFold(5, shuffle=True, random_state=0).split(ann):
    trd = ann.iloc[tr]
    X = list(trd.aae_text[trd.nat == "natural"]) + list(trd.sae_text)
    y = [1] * int((trd.nat == "natural").sum()) + [0] * len(trd)
    m = model().fit(X, y)
    ted = ann.iloc[te]; tn = ted[ted.nat == "natural"]
    Xt = list(tn.aae_text) + list(ted.sae_text); yt = [1] * len(tn) + [0] * len(ted)
    acc.append(accuracy_score(yt, m.predict(Xt)))
    oof[te] = m.predict_proba(list(ted.aae_text))[:, 1]
print("\n[1] Twitter AAE vs its SAE translation, 5-fold CV accuracy: %.3f" % np.mean(acc))

# ---- Experiment 2: can it tell native-approved AAE from tweets natives rejected? ----
lab = ann[nat | unn]; s = oof[(nat | unn).values]
print("[2] Native speakers' call on the AAE side (natural vs rejected):")
print("    classifier AUC natural-vs-rejected: %.3f  (0.5 = coin flip)" % roc_auc_score((lab.nat == "natural").astype(int), s))
print("    flagged as AAE (p>0.5): natural %.1f%%  rejected %.1f%%  somewhat %.1f%%" % (
    100 * (oof[nat.values] > .5).mean(), 100 * (oof[unn.values] > .5).mean(), 100 * (oof[some.values] > .5).mean()))
gs = ann.aae_text.map(gram)
print("    grammar-rule AUC natural-vs-rejected: %.3f" % roc_auc_score((lab.nat == "natural").astype(int), gs[(nat | unn).values]))
print("    grammar rules fire (>=1): natural %.1f%%  rejected %.1f%%" % (100 * (gs[nat] > 0).mean(), 100 * (gs[unn] > 0).mean()))
for a in (1, 2, 3):
    sub = ann[(ann.ann == a) & (nat | unn)]
    if sub.nat.nunique() == 2:
        print("    annotator %d: n=%d rejected=%d  classifier AUC %.3f" % (a, len(sub), (sub.nat != "natural").sum(),
              roc_auc_score((sub.nat == "natural").astype(int), oof[sub.index])))

# ---- Experiment 3: train on all, test human-reviewed AAE from other sources ----
X = list(ann.aae_text[nat]) + list(ann.sae_text); y = [1] * int(nat.sum()) + [0] * len(ann)
m = model().fit(X, y)
d = pd.read_csv(f"{U}/2ac736e4-AAE_Dataset_v1.0_all.csv", low_memory=False)
d["src"] = d.source_subtype.str.lower()
keys = set(ann.aae_text.str.lower().str.strip())
d = d[~d.item_text.astype(str).str.lower().str.strip().isin(keys)]
rows = []
def add(name, t):
    t = [str(x) for x in t]
    rows.append((name, len(t), m.predict(t).mean(), m.predict([mask(x) for x in t]).mean(), np.mean([gram(x) > 0 for x in t])))
pos = d[d.dialect_label == "AAE"]
for s_, g in pos.groupby("src"): add(s_, g.item_text)
acc_ = d[d.human_review_decision == "accept"]; add("ALL explicit human 'accept' rows", acc_.item_text)
add("Twitter SAE translations (should be ~0)", ann.sae_text)
print("\n[3] Trained on Twitter, tested on other sources (human-reviewed AAE only)")
print(pd.DataFrame(rows, columns=["source", "n", "classifier", "classifier_slang_removed", "grammar_rules"]).round(3).to_string(index=False))
names = np.concatenate([t.get_feature_names_out() for _, t in m[0].transformer_list]); c = m[1].coef_[0]
print("\ntop AAE cues:", [names[i] for i in np.argsort(-c)[:20]])
