"""Entry point: python -m aaeid.run --config configs/default.yaml

Writes only IDs, scores and metrics to the output folder. No text is written,
so results can be shared even when the underlying data cannot.
"""
import argparse
import hashlib
import json
import platform
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
import yaml

from . import __version__
from .grammar import RULE_NAMES, detect
from .text import normalize
from .data import load_annotations, load_external, manifest
from .experiments import cross_register, cross_validate, leave_one_annotator_out, make_folds, oof_analyses
from .models import top_features
from .report import write_report


def _clean(o):
    """Make numpy/pandas values JSON-safe and round floats so files diff cleanly."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return None if np.isnan(o) else round(float(o), 4)
    return o


def grammar_coverage(df, ext):
    """Share of lines (%) where each grammar rule fires, per data slice."""
    slices = {
        "twitter_natural_aae": df.loc[df["original_naturalness"] == "natural", "aae_text"],
        "twitter_rejected": df.loc[df["original_naturalness"] == "unnatural_or_noisy", "aae_text"],
        "llm_sae_translation": df["sae_text"],
        "external_human_accepted": ext.loc[ext["label_tier"] == "human", "item_text"],
        "external_model_labeled": ext.loc[ext["label_tier"] == "model", "item_text"],
    }
    out = {}
    for name, texts in slices.items():
        hits = [detect(normalize(t)) for t in texts]
        out[name] = {r: 100 * float(np.mean([h[r] for h in hits])) for r in RULE_NAMES}
        out[name]["any_rule"] = 100 * float(np.mean([any(h.values()) for h in hits]))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-dir", help="override data_dir in the config")
    ap.add_argument("--out", help="override out_dir in the config")
    args = ap.parse_args(argv)

    cfg_text = Path(args.config).read_text()
    cfg = yaml.safe_load(cfg_text)
    if args.data_dir:
        cfg["data_dir"] = args.data_dir
    if args.out:
        cfg["out_dir"] = args.out
    out = Path(cfg["out_dir"])
    (out / "scores").mkdir(parents=True, exist_ok=True)

    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    df, data_log = load_annotations(cfg)
    ext, ext_log = load_external(cfg, df["aae_text"])
    folds = make_folds(df, cfg["n_splits"], cfg["n_repeats"], cfg["seed"])
    folds.to_csv(out / "folds.csv", index=False)

    results = {"data": {"annotations": data_log, "external": ext_log,
                        "grammar_coverage": grammar_coverage(df, ext)}, "models": {}}
    for name, spec in cfg["models"].items():
        print(f"[{name}] cross-validating", file=sys.stderr)
        cv, oof = cross_validate(df, spec, folds, cfg)
        ana = oof_analyses(df, oof, cfg)
        loao = leave_one_annotator_out(df, spec, cfg)
        xr, full_model, ext_scores = cross_register(df, ext, spec, ana["threshold"], cfg)
        results["models"][name] = {"spec": spec, "cv": cv, "oof": ana,
                                   "leave_one_annotator_out": loao, "cross_register": xr,
                                   "features": top_features(full_model)}
        oof.round(6).to_csv(out / "scores" / f"oof_{name}.csv", index=False)
        pd.DataFrame({"item_id": ext["item_id"], "source": ext["source"], "label_tier": ext["label_tier"],
                      "score": np.round(ext_scores, 6)}).to_csv(out / "scores" / f"external_{name}.csv", index=False)

    results = _clean(results)
    (out / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True))
    run_info = {
        "package_version": __version__,
        "config_sha256": hashlib.sha256(cfg_text.encode()).hexdigest(),
        "config": cfg,
        "data_manifest": manifest(cfg),
        "python": platform.python_version(),
        "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__, "pyyaml": yaml.__version__,
    }
    (out / "run_info.json").write_text(json.dumps(_clean(run_info), indent=2, sort_keys=True))
    write_report(results, run_info, out / "report.md")
    print(f"wrote {out}/results.json, report.md, run_info.json, folds.csv, scores/", file=sys.stderr)


if __name__ == "__main__":
    main()
