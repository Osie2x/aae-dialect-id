# Data

No data is in this repository. The files below go in `data/raw/` under these names. `run_info.json` records the
SHA-256 of each file used for the published results, so you can confirm you have the same version.

| file | contents | where it comes from | can it be shared? |
|---|---|---|---|
| `annotator_1_annotations_2026-05-19T04-05-02.317Z.csv` | 300 TwitterAAE tweets, Gemini SAE translation, back-translation, native-speaker labels | Responsible AI Lab, York University (REAL-AAE annotation round) | **No.** Lab data, and the file contains annotator personal information. Ask the lab. |
| `annotator_2_annotations_300.csv` | same, different 300 tweets | same | No |
| `annotator_3_annotations_300_unique.csv` | same, different 300 tweets | same | No |
| `AAE_Dataset_v1.0_all.csv` | 3,183 candidate lines from CORAAL, film and TV, lyrics, proverbs, podcasts, TikTok and Reddit, with provenance and review columns | AAE Challenge Benchmark working set | **No.** Most rows have `rights_status = not_established_for_redistribution`. CORAAL rows are CC BY-NC-SA 4.0. |

## Columns the code uses

**Annotator files:** `item_index`, `aae_text`, `sae_text`, `aae_back_text`, `original_naturalness`
(`natural` / `somewhat_natural` / `unnatural_or_noisy`), `back_naturalness`
(`natural` / `somewhat_natural` / `unnatural_or_imitated`).

The annotator-identity columns (`aae_speaker_identity`, `regional_background*`, `written_aae_familiarity`,
`acknowledgement_*`) are dropped on load. They never appear in outputs.

**External file:** `item_id`, `item_text`, `source_subtype`, `dialect_label`, `prediction_basis`,
`human_review_decision`. A row counts as human-labeled only if `prediction_basis == direct_human_review` and
`human_review_decision == accept`. Rows with `codebook_based_model_prediction` are reported separately as
model-labeled.

## Cleaning, with counts from the published run

- 900 tweets loaded; 0 duplicates.
- 26 removed because the "translation" is identical to the tweet. The same string can't be both a positive and a negative.
- 874 tweets remain: 720 natural, 112 rejected, 42 somewhat natural.
- External: 1,018 rows labeled AAE, 301 Twitter rows excluded, 27 human-"uncertain" rows dropped.
- 172 human-accepted rows and 518 model-labeled rows remain.

## To release a dataset later

Release tweet IDs, not text, as Twitter/X terms require. Release CORAAL line references, not transcripts. Release annotator labels only with annotator consent and without identity fields.
