"""End-to-end run on synthetic data: outputs exist, contain no text, and are deterministic."""
import json
from pathlib import Path

import pandas as pd

from aaeid.data import load_annotations
from aaeid.run import main


def test_cleaning_drops_pii_and_identical_pairs(synthetic_cfg):
    _, cfg = synthetic_cfg
    df, log = load_annotations(cfg)
    assert "acknowledgement_name" not in df.columns and "regional_background" not in df.columns
    # "same text here" is in all three files: 2 copies go as duplicates, the last as an identical pair
    assert log["duplicate_tweets_removed"] == 2
    assert log["identical_aae_sae_pairs_removed"] == 1


def test_end_to_end_is_deterministic_and_text_free(synthetic_cfg, tmp_path):
    cfg_path, cfg = synthetic_cfg
    out1, out2 = tmp_path / "run1", tmp_path / "run2"
    main(["--config", str(cfg_path), "--out", str(out1)])
    main(["--config", str(cfg_path), "--out", str(out2)])

    for f in ["results.json", "report.md", "run_info.json", "folds.csv"]:
        assert (out1 / f).exists(), f
    # Same config + same data -> byte-identical results (run_info differs only in out_dir).
    assert (out1 / "results.json").read_text() == (out2 / "results.json").read_text()
    assert (out1 / "folds.csv").read_text() == (out2 / "folds.csv").read_text()

    # No input text may appear in any output file.
    texts = set(pd.read_csv(Path(cfg["data_dir"]) / "ann_A1.csv")["aae_text"])
    for f in out1.rglob("*"):
        if f.is_file():
            body = f.read_text()
            assert not any(t in body for t in texts), f"text leaked into {f}"
            assert "SHOULD-BE-DROPPED" not in body

    r = json.loads((out1 / "results.json").read_text())
    assert set(r["models"]) == {"grammar_rules", "hybrid_casualneg"}
    assert 0.0 <= r["models"]["hybrid_casualneg"]["cv"]["auc_aae_vs_sae"]["mean"] <= 1.0


def test_folds_keep_pairs_together(synthetic_cfg):
    from aaeid.experiments import make_folds
    _, cfg = synthetic_cfg
    df, _ = load_annotations(cfg)
    f = make_folds(df, 3, 2, 1)
    # each tweet appears exactly once per repeat
    assert (f.groupby(["repeat", "tweet_id"]).size() == 1).all()
    assert f["tweet_id"].nunique() == len(df)
