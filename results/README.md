# Saved results

The supplied experiment outputs are preserved in `default/`.

| File | Purpose |
|---|---|
| [report.md](default/report.md) | Human-readable findings and uncertainty intervals |
| [results.json](default/results.json) | Machine-readable metrics for each model |
| [folds.csv](default/folds.csv) | Saved cross-validation assignments |
| [run_info.json](default/run_info.json) | Input hashes, configuration, and dependency versions |

The separately uploaded `aae_classifier_results_report.md` was byte-identical to `default/report.md`; the report is kept in one canonical location here. Its original root-level copy remains in Git history.

These are saved artifacts from the supplied project, not a new reproduction. To reproduce them, obtain the original inputs listed in [the data guide](../docs/DATA.md), run `make run`, and compare with `make verify` from the repository root. Keep human-reviewed and model-labeled evaluation slices separate when interpreting the report.
