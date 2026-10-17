# Model card: aaeid baselines

**Models.** Logistic regression on character and word n-grams, optionally with 8 binary grammar-rule features
(`configs/default.yaml`). Recommended default: `hybrid_casualneg`.

**Intended use.** Research only. Use it to measure how dialect detectors behave across registers, as a cheap
baseline judge in evaluation studies, and to sample candidate lines for **human** review.

**Out of scope. Do not use for:**
- inferring the race or ethnicity of any person or account;
- filtering, moderating, ranking or scoring people's speech or writing;
- deciding what "counts" as AAE in a dataset without human review. Selecting data with this model and then
  testing on that data is circular.

**Training data.** 720 TwitterAAE tweets rated natural AAE by native-speaker annotators (positives). Gemini SAE
translations of 874 tweets (negatives). Optionally casual rewrites of those translations, tweets natives
rejected, and back-translations natives rated performed. See DATA.md.

**Evaluation.** See README "Main findings" and `results/default/report.md`. In short, it works in-domain
(AUC about 0.92 against formal and casual SAE). Recall drops on spoken and scripted AAE. It barely separates
real from performed AAE (AUC about 0.6).

**Known biases and failure modes.**
- It learns platform style, profanity and internet spelling alongside dialect. Top features include "u", "im"
  and "shit" unless they are masked.
- It is trained on one platform, one period and three annotators from two regions. Southern AAE is not represented.
- Short lines get fewer features. Length was tested and does not explain the cross-register drop, but scores on
  1–3 word lines are unreliable.
- A high score means "looks like the Twitter AAE in this sample", not "authentic AAE".
