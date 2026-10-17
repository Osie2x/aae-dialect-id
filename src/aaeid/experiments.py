"""The four experiments. Each returns plain dicts so results serialize to JSON."""
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from . import stats
from .models import build, top_features, training_set
from .text import casualize, truncate_words


def make_folds(df, n_splits, n_repeats, seed):
    """Folds over tweets. A tweet, its SAE translation and its back-translation always
    share a fold. Stratified by annotator x naturalness so each fold has the same mix."""
    strata = df["annotator"] + "|" + df["original_naturalness"].fillna("na")
    rows = []
    for r in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        for f, (_, te) in enumerate(skf.split(df, strata)):
            rows += [{"tweet_id": df["tweet_id"].iat[i], "repeat": r, "fold": f} for i in te]
    return pd.DataFrame(rows)


def pick_threshold(neg_scores, specificity):
    """Lowest threshold whose false-positive rate on held-out negatives is at most
    1 - specificity, with "positive" meaning score >= threshold. Handles tied scores
    (the rule-count model has only a few distinct values), where a plain quantile does not."""
    s = np.sort(np.unique(neg_scores))
    for t in s:
        if np.mean(neg_scores >= t) <= 1 - specificity:
            return float(t)
    return float(np.nextafter(s[-1], np.inf))  # nothing qualifies: stricter than every negative


def cross_validate(df, spec, folds, cfg):
    """In-domain CV. Returns per-repeat metrics and repeat-0 out-of-fold scores."""
    seed = cfg["seed"]
    nat = (df["original_naturalness"] == "natural").to_numpy()
    lens = cfg["truncation_lengths"]
    per_repeat = []
    oof0 = None
    for r in sorted(folds["repeat"].unique()):
        fr = folds[folds["repeat"] == r].set_index("tweet_id").loc[df["tweet_id"], "fold"].to_numpy()
        s_aae = np.zeros(len(df)); s_sae = np.zeros(len(df)); s_cas = np.zeros(len(df)); s_back = np.full(len(df), np.nan)
        s_trunc = {k: np.zeros(len(df)) for k in lens}
        for f in np.unique(fr):
            tr, te = fr != f, fr == f
            m = build(spec, seed).fit(*training_set(df[tr], spec))
            p = lambda texts: m.predict_proba(list(texts))[:, 1]
            s_aae[te] = p(df.loc[te, "aae_text"])
            s_sae[te] = p(df.loc[te, "sae_text"])
            s_cas[te] = p(df.loc[te, "sae_text"].map(casualize))
            back = df.loc[te, "aae_back_text"]
            idx = np.where(te)[0][back.notna().to_numpy()]
            if len(idx):
                s_back[idx] = p(back.dropna())
            for k in lens:
                s_trunc[k][te] = p(df.loc[te, "aae_text"].map(lambda t: truncate_words(t, k)))
        y = np.r_[np.ones(nat.sum()), np.zeros(len(df))]
        s = np.r_[s_aae[nat], s_sae]
        per_repeat.append({
            "auc_aae_vs_sae": roc_auc_score(y, s),
            "balanced_acc_at_0.5": balanced_accuracy_score(y, s > 0.5),
            # Harder negatives: the same translations rewritten casually.
            "auc_aae_vs_casual_sae": roc_auc_score(y, np.r_[s_aae[nat], s_cas]),
        })
        if r == 0:
            oof0 = pd.DataFrame({"tweet_id": df["tweet_id"], "s_aae": s_aae, "s_sae": s_sae, "s_sae_casual": s_cas, "s_back": s_back,
                                 **{f"s_trunc{k}": s_trunc[k] for k in lens}})
    out = {k: stats.mean_sd([d[k] for d in per_repeat]) for k in per_repeat[0]}
    return out, oof0


def oof_analyses(df, oof, cfg):
    """Everything computed from out-of-fold scores (repeat 0)."""
    seed = cfg["seed"]
    on = df["original_naturalness"]
    nat, rej = (on == "natural").to_numpy(), (on == "unnatural_or_noisy").to_numpy()
    thr = pick_threshold(oof["s_sae"].to_numpy(), cfg["target_specificity"])
    res = {"threshold": thr}
    res["in_domain_recall_natural_aae"] = stats.wilson(int((oof["s_aae"][nat] >= thr).sum()), int(nat.sum()))
    res["sae_false_positive_rate"] = stats.wilson(int((oof["s_sae"] >= thr).sum()), len(df))
    res["casual_sae_false_positive_rate"] = stats.wilson(int((oof["s_sae_casual"] >= thr).sum()), len(df))

    keep = nat | rej
    res["auc_native_natural_vs_rejected"] = stats.auc_ci(nat[keep].astype(int), oof["s_aae"][keep], seed=seed)
    res["auc_native_natural_vs_rejected_by_annotator"] = {
        a: stats.auc_ci(nat[keep & (df["annotator"] == a).to_numpy()].astype(int),
                        oof["s_aae"][keep & (df["annotator"] == a).to_numpy()], seed=seed)
        for a in sorted(df["annotator"].unique())}

    bn = df["back_naturalness"]
    b_nat = ((bn == "natural") & oof["s_back"].notna()).to_numpy()
    b_perf = ((bn == "unnatural_or_imitated") & oof["s_back"].notna()).to_numpy()
    kb = b_nat | b_perf
    res["auc_real_vs_performed_backtranslation"] = stats.auc_ci(b_nat[kb].astype(int), oof["s_back"][kb], seed=seed)

    res["recall_natural_aae_truncated"] = {
        str(k): stats.wilson(int((oof[f"s_trunc{k}"][nat] >= thr).sum()), int(nat.sum()))
        for k in cfg["truncation_lengths"]}
    return res


def leave_one_annotator_out(df, spec, cfg):
    """Each annotator labeled different tweets, so this tests a shift in both data and rater."""
    out = {}
    for a in sorted(df["annotator"].unique()):
        tr, te = df[df["annotator"] != a], df[df["annotator"] == a]
        m = build(spec, cfg["seed"]).fit(*training_set(tr, spec))
        nat = te["original_naturalness"] == "natural"
        y = np.r_[np.ones(nat.sum()), np.zeros(len(te))]
        s = np.r_[m.predict_proba(list(te.loc[nat, "aae_text"]))[:, 1], m.predict_proba(list(te["sae_text"]))[:, 1]]
        out[a] = stats.auc_ci(y.astype(int), s, seed=cfg["seed"])
    return out


def cross_register(df, ext, spec, thr, cfg):
    """Train on all tweets, apply the in-domain threshold to other registers.
    Only positives exist outside Twitter, so this is recall (coverage), not accuracy."""
    m = build(spec, cfg["seed"]).fit(*training_set(df, spec))
    scores = m.predict_proba(list(ext["item_text"]))[:, 1]
    res = {}
    for (src, tier), g in ext.assign(score=scores).groupby(["source", "label_tier"]):
        res.setdefault(tier, {})[src] = {
            **stats.wilson(int((g["score"] >= thr).sum()), len(g)),
            "median_words": float(g["n_words"].median()),
        }
    return res, m, scores
