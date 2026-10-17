"""Synthetic fixture data, written by hand for testing only.

These sentences are invented. They are not real tweets or transcripts and say
nothing about AAE; they only exercise the pipeline end to end on CI, where the
real data cannot be shared.
"""
import itertools

import pandas as pd
import pytest
import yaml

SUBJ = ["he", "she", "they", "we", "my brother", "my cousin"]
AAE_TPL = ["{s} be working all day lol", "{s} ain't got no time fr", "{s} finna go to the store",
           "{s} don't never listen smh", "{s} done told u already", "{s} at the crib rn"]
SAE_TPL = ["{s} usually works all day.", "{s} does not have any time.", "{s} is about to go to the store.",
           "{s} never listens.", "{s} has already told you.", "{s} is at home right now."]
EXT = [("coraal", "he be going down there every week"), ("coraal", "we ain't have nothing back then"),
       ("tv_script", "you crazy man"), ("tv_script", "she gon handle it"),
       ("rap_lyrics", "i been on my grind"), ("proverb_folklore", "what goes around comes around")]


def _annotator_rows(offset):
    rows = []
    for i, (s, j) in enumerate(itertools.product(SUBJ, range(len(AAE_TPL)))):
        nat = "unnatural_or_noisy" if (i + offset) % 7 == 0 else "natural"
        back = "unnatural_or_imitated" if (i + offset) % 5 == 0 else "natural"
        rows.append({
            "item_index": i, "aae_text": AAE_TPL[j].format(s=s) + f" #{offset}x{i}",
            "sae_text": SAE_TPL[j].format(s=s.capitalize()),
            "aae_back_text": AAE_TPL[(j + 1) % len(AAE_TPL)].format(s=s),
            "original_naturalness": nat, "back_naturalness": back,
            "acknowledgement_name": "SHOULD-BE-DROPPED", "regional_background": "SHOULD-BE-DROPPED",
        })
    # one pair where the translation equals the tweet: must be removed
    rows.append({"item_index": 999, "aae_text": "same text here", "sae_text": "Same text here.",
                 "aae_back_text": "same", "original_naturalness": "natural", "back_naturalness": "natural"})
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_cfg(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    files = {}
    for k, a in enumerate(["A1", "A2", "A3"]):
        f = f"ann_{a}.csv"
        _annotator_rows(k).to_csv(data / f, index=False)
        files[a] = f
    ext = []
    for i, (src, t) in enumerate(EXT * 3):
        ext.append({"item_id": f"x{i}", "item_text": t + ("" if i < len(EXT) else f" {i}"), "source_subtype": src,
                    "dialect_label": "AAE",
                    "prediction_basis": "direct_human_review" if i % 2 == 0 else "codebook_based_model_prediction",
                    "human_review_decision": "accept" if i % 2 == 0 else None})
    pd.DataFrame(ext).to_csv(data / "external.csv", index=False)
    cfg = {"data_dir": str(data), "out_dir": str(tmp_path / "out"), "seed": 1,
           "annotation_files": files, "external_file": "external.csv",
           "exclude_sources": ["twitteraae_blodgett"], "min_words": 1,
           "n_splits": 3, "n_repeats": 2, "target_specificity": 0.95, "truncation_lengths": [4],
           "models": {"grammar_rules": {"features": ["grammar"], "preprocess": "norm"},
                      "hybrid_casualneg": {"features": ["char", "word", "grammar"], "preprocess": "norm",
                                           "casual_negatives": True}}}
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p, cfg
