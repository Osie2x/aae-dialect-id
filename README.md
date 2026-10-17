# aaeid: where AAE detectors break

Baselines and an evaluation harness for **African American English (AAE) identification in text**.
They show what a Twitter-trained detector actually learns and where it stops working.

> **Scope of the claim.** These models detect *written online AAE as it appears in this Twitter sample*,
> against *LLM-written Standard American English*. They are not general AAE detectors, and nothing here
> should be used to infer anyone's race or to filter people's speech. See [MODEL_CARD.md](MODEL_CARD.md).

## Main findings (results/default/report.md)

| Finding | Evidence |
|---|---|
| The pilot classifier was mostly a **formal-vs-informal** detector | AUC 0.97 against LLM SAE drops to 0.80 when the same SAE is written casually ("u", "im", no punctuation). It then calls 47% of casual SAE "AAE". |
| Training with casual negatives fixes that in-domain | `hybrid_casualneg`: AUC 0.93 against casual SAE, 4% false positives. |
| Recall falls outside Twitter, and length is not the reason | Human-accepted lines: CORAAL 68–79%, TV 55–64%, movie 43%. Twitter tweets cut to 6 words keep 63–70% recall. |
| Native judgment is only partly tracked | Natural vs rejected tweets AUC ≈ 0.76. Real vs **performed** AAE AUC ≈ 0.60–0.64, close to chance. |
| Canonical AAE grammar is rare in tweets | Any of 8 grammar rules fires on 11% of natural AAE tweets. |
| Most "AAE" labels outside Twitter came from a model | 819 of 1,018 rows. Those rows fire grammar rules more often (46% vs 34%), so the labeler selected feature-rich lines. They are reported separately and never mixed with human labels. |

Every number has a 95% interval in the report. Several external slices are tiny (5 natural-conversation lines,
11 TV lines), so treat them as pilot evidence. [AUDIT.md](AUDIT.md) lists the 13 problems found in the
original pilot scripts and how each was fixed.

## Quick start

```bash
git clone <this repo> && cd aae-dialect-id
python -m venv .venv && source .venv/bin/activate
make install        # pinned versions from requirements.txt
make test           # 30 tests on synthetic data, no real data needed
# put the data files in data/raw/ (see DATA.md), then:
make run            # writes results/default/  (~4 min on a laptop CPU)
make verify         # reruns and checks the results are byte-identical
```

## What runs

`configs/default.yaml` defines seven models. All are logistic regression on different features:

| model | features | what it tests |
|---|---|---|
| `grammar_rules` | 8 regex grammar rules | interpretable floor |
| `tfidf_raw` | char 2–5 + word 1–2 n-grams, raw text | the original pilot |
| `tfidf_norm` | same, URLs, @, #, digits, punctuation removed | platform artifacts |
| `hybrid` | `tfidf_norm` + grammar rules | does grammar add anything |
| `hybrid_no_slang` | slang, internet spelling and profanity removed at train **and** test | reliance on words |
| `hybrid_casualneg` | + casual SAE rewrites as negatives | informality shortcut |
| `hybrid_hardneg` | + native-rejected tweets and performed back-translations as negatives | native-judgment signal |

For each model the harness reports:
1. 5×5 repeated, annotator-stratified cross-validation. A tweet and its translations always share a fold.
2. Agreement with native judgments (natural vs rejected, real vs performed).
3. Leave-one-annotator-out.
4. Cross-register recall at a threshold fixed in-domain (95% specificity on held-out SAE).
5. A length control.
6. Top features.

## Repository layout

```
configs/default.yaml   every setting that affects results
src/aaeid/             text.py, grammar.py, data.py, models.py, experiments.py, stats.py, run.py, report.py
tests/                 unit tests, grammar-rule tests, end-to-end determinism and no-text-leak test
results/default/       report.md, results.json, run_info.json (input hashes and versions), folds.csv
legacy/                the original pilot scripts, kept for comparison
AUDIT.md  DATA.md  MODEL_CARD.md
```

## Reproducibility

- **Fixed seeds and deterministic solvers.** Same config and data give byte-identical `results.json`. A test and `make verify` check this.
- **Input provenance.** `run_info.json` records SHA-256 hashes of every input file, the config hash, and Python and library versions.
- **Shareable outputs.** Outputs contain IDs, scores and metrics only, never text, so they can be shared even though the data cannot. A test enforces this.
- **Saved folds.** `folds.csv` lets another model be evaluated on exactly the same splits.

## Known limitations

- Negatives are LLM translations, not real non-AAE writing. Real non-AAE tweets (for example Blodgett's white-aligned set) and non-AAE speech (Buckeye or the Santa Barbara Corpus; CORAAL has no white speakers) are needed before claiming precision outside Twitter.
- Only 172 external lines have a human "accept".
- The annotators labeled different tweets, so annotator and data effects are confounded.
- No transformer baseline yet. Next: Spliethöver et al. (2024) DeBERTa and a fine-tuned RoBERTa on `folds.csv`.

## Citation

See [CITATION.cff](CITATION.cff). Code is MIT-licensed. The license does not cover any dataset.
