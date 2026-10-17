"""Loading and cleaning. Every exclusion is counted and reported, never silent."""
import hashlib
from pathlib import Path

import pandas as pd

from .text import match_key, n_words

# Columns that identify annotators. They are dropped on load and never written out.
ANNOTATOR_PII = [
    "aae_speaker_identity", "regional_background", "regional_background_notes",
    "written_aae_familiarity", "acknowledgement_consent", "acknowledgement_name",
]


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest(cfg) -> list:
    """File name, size and hash of every input, so a rerun can prove it used the same data."""
    d = Path(cfg["data_dir"])
    files = list(cfg["annotation_files"].values()) + [cfg["external_file"]]
    return [{"file": f, "bytes": (d / f).stat().st_size, "sha256": sha256(d / f)} for f in files]


def load_annotations(cfg):
    """One row per annotated tweet, with its LLM SAE translation and LLM back-translation.

    Returns (df, log). df columns: tweet_id, annotator, aae_text, sae_text, aae_back_text,
    original_naturalness, back_naturalness.
    """
    d = Path(cfg["data_dir"])
    frames = []
    for ann, fname in cfg["annotation_files"].items():
        x = pd.read_csv(d / fname)
        x = x.drop(columns=[c for c in ANNOTATOR_PII if c in x.columns])
        x["annotator"] = str(ann)
        x["tweet_id"] = [f"{ann}_{i}" for i in x["item_index"]]
        frames.append(x)
    df = pd.concat(frames, ignore_index=True)
    log = {"rows_loaded": len(df)}

    df = df.dropna(subset=["aae_text", "sae_text"])
    log["after_drop_missing_text"] = len(df)

    # The same tweet labeled by two annotators would leak across folds.
    df["_k"] = df["aae_text"].map(match_key)
    log["duplicate_tweets_removed"] = int(df["_k"].duplicated().sum())
    df = df.drop_duplicates("_k")

    # When the "SAE translation" is the tweet itself, the same string is both
    # a positive and a negative. Those pairs are label noise.
    same = df["_k"] == df["sae_text"].map(match_key)
    log["identical_aae_sae_pairs_removed"] = int(same.sum())
    df = df[~same]

    df = df[df["aae_text"].map(n_words) >= cfg.get("min_words", 1)]
    log["final_tweets"] = len(df)
    log["original_naturalness"] = df["original_naturalness"].value_counts().to_dict()
    log["back_naturalness"] = df["back_naturalness"].value_counts().to_dict()
    log["by_annotator"] = df.groupby("annotator")["original_naturalness"].value_counts().unstack(fill_value=0).to_dict("index")
    cols = ["tweet_id", "annotator", "aae_text", "sae_text", "aae_back_text",
            "original_naturalness", "back_naturalness"]
    return df[cols].reset_index(drop=True), log


def load_external(cfg, train_texts):
    """Non-Twitter AAE for the cross-register test.

    label_tier:
      human  = a person accepted this row (direct_human_review + accept). Primary results.
      model  = labeled AAE by the codebook-based model prediction. Secondary only:
               these labels were never checked by a person.
    """
    d = Path(cfg["data_dir"])
    x = pd.read_csv(d / cfg["external_file"], low_memory=False)
    log = {"rows_loaded": len(x)}
    x["source"] = x["source_subtype"].astype(str).str.lower()
    x = x[x["dialect_label"] == "AAE"]
    log["dialect_label_AAE"] = len(x)
    x = x[~x["source"].isin(cfg["exclude_sources"])]

    keys = set(map(match_key, train_texts))
    overlap = x["item_text"].map(match_key).isin(keys)
    log["overlap_with_training_removed"] = int(overlap.sum())
    x = x[~overlap]

    human = (x["prediction_basis"] == "direct_human_review") & (x["human_review_decision"] == "accept")
    model = x["prediction_basis"] == "codebook_based_model_prediction"
    x = x.assign(label_tier=None)
    x.loc[human, "label_tier"] = "human"
    x.loc[model, "label_tier"] = "model"
    log["human_review_uncertain_dropped"] = int((~human & ~model).sum())
    x = x[x["label_tier"].notna()]
    x["n_words"] = x["item_text"].map(n_words)
    log["by_source_and_tier"] = x.groupby(["source", "label_tier"]).size().unstack(fill_value=0).to_dict("index")
    return x[["item_id", "source", "label_tier", "item_text", "n_words"]].reset_index(drop=True), log
