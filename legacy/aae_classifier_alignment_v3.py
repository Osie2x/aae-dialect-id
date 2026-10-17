import re, sys, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, make_union
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
U = sys.argv[1]
F = ["30576fe1-annotator_1_annotations_2026-05-19T04-05-02.317Z_6.csv",
     "ef443431-annotator_2_annotations_300_6.csv",
     "15cd2aa7-annotator_3_annotations_300_unique_7.csv"]
ann = pd.concat([pd.read_csv(f"{U}/{f}") for f in F], ignore_index=True)
ann = ann.dropna(subset=["aae_text","sae_text"]).drop_duplicates("aae_text").reset_index(drop=True)
nat = (ann.original_naturalness=="natural").values; rej = (ann.original_naturalness=="unnatural_or_noisy").values
perf = (ann.back_naturalness=="unnatural_or_imitated").values; bnat = (ann.back_naturalness=="natural").values
print("tweets",len(ann),"natural",nat.sum(),"rejected",rej.sum(),"| back-translations: performed",perf.sum(),"natural",bnat.sum())

def norm(t):
    t = str(t).lower()
    t = re.sub(r"https?://\S+|@\w+|#", " ", t)       # urls, mentions, hashtag sign
    t = re.sub(r"\d+(\.\d+)?", " 0 ", t)            # numbers / metadata
    t = re.sub(r"([!?.,'\"*~<>&:;()\-])\1+", r"\1", t)  # repeated punctuation
    t = re.sub(r"[^\w\s']", " ", t)                 # drop remaining punctuation/emoji
    return re.sub(r"\s+", " ", t).strip()
def model(normed):
    pre = norm if normed else None
    return make_pipeline(make_union(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2,5), min_df=2, sublinear_tf=True, preprocessor=pre),
        TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True, preprocessor=pre)),
        LogisticRegression(C=4, max_iter=3000, class_weight="balanced"))

d = pd.read_csv(f"{U}/2ac736e4-AAE_Dataset_v1.0_all.csv", low_memory=False)
d["src"] = d.source_subtype.str.lower()
d = d[(d.dialect_label=="AAE") & (d.src!="twitteraae_blodgett")]

VARIANTS = {
 "v2 baseline (AAE vs SAE)":            dict(normed=False, rej=False, perf=False),
 "A: + text normalization":             dict(normed=True,  rej=False, perf=False),
 "B: + rejected tweets as negatives":   dict(normed=False, rej=True,  perf=False),
 "C: + performed back-transl. as neg":  dict(normed=False, rej=False, perf=True),
 "A+B+C":                               dict(normed=True,  rej=True,  perf=True),
}
def build(idx, v):
    s = ann.iloc[idx]; n = nat[idx]; X=list(s.aae_text[n]); y=[1]*n.sum()
    X += list(s.sae_text); y += [0]*len(s)
    if v["rej"]: X += list(s.aae_text[rej[idx]]); y += [0]*rej[idx].sum()
    if v["perf"]: X += list(s.aae_back_text[perf[idx]]); y += [0]*perf[idx].sum()
    return X, y
rows=[]
for name, v in VARIANTS.items():
    oof_a = np.zeros(len(ann)); oof_b = np.zeros(len(ann))
    for tr, te in KFold(5, shuffle=True, random_state=0).split(ann):
        m = model(v["normed"]).fit(*build(tr, v))
        oof_a[te] = m.predict_proba(list(ann.aae_text.iloc[te]))[:,1]
        oof_b[te] = m.predict_proba(list(ann.aae_back_text.fillna("").iloc[te]))[:,1]
    keep = nat|rej
    auc_rej = roc_auc_score(nat[keep].astype(int), oof_a[keep])
    kb = perf|bnat
    auc_perf = roc_auc_score(bnat[kb].astype(int), oof_b[kb])
    m = model(v["normed"]).fit(*build(np.arange(len(ann)), v))
    rec = {s: m.predict(list(g.item_text.astype(str))).mean() for s, g in d.groupby("src")}
    sae_fp = (oof_a*0).mean()
    rows.append(dict(variant=name, AUC_native_natural_vs_rejected=auc_rej,
        rejected_called_AAE=(oof_a[rej]>.5).mean(), natural_called_AAE=(oof_a[nat]>.5).mean(),
        AUC_natural_vs_performed_backtransl=auc_perf, **{f"recall_{k}":v_ for k,v_ in rec.items()}))
r = pd.DataFrame(rows).set_index("variant").T.round(3)
print(r.to_string())
