import numpy as np

from aaeid.experiments import pick_threshold
from aaeid.stats import auc_ci, gwet_ac1, wilson
from aaeid.text import casualize, mask_slang, match_key, normalize


def test_normalize_strips_platform_artifacts():
    assert normalize("@bob LOL!!! check https://t.co/x #blessed 2day") == "lol check blessed 0 day"
    assert normalize("I ain’t") == "i ain't"  # curly apostrophe kept as a real one


def test_mask_removes_slang_and_internet_spelling_only():
    assert mask_slang("u ain't finna go lol") == "go"
    assert mask_slang("the bout was good") == "the was good"  # known over-masking, documented
    assert mask_slang("he went home") == "he went home"


def test_casualize_adds_no_dialect_features():
    from aaeid.grammar import count
    s = "You're going to love it, because people like you."
    c = casualize(s)
    assert c == "ur gonna love it cuz ppl like u"
    assert count(c) == 0


def test_match_key_ignores_case_and_punctuation():
    assert match_key("Hey, U!") == match_key("hey u")


def test_threshold_handles_ties():
    neg = np.array([0.0] * 90 + [1.0] * 10)  # rule-count style scores
    t = pick_threshold(neg, 0.95)
    assert np.mean(neg >= t) <= 0.05  # a plain 95th-percentile would give 0.0 -> 100% FPR


def test_threshold_continuous():
    neg = np.linspace(0, 1, 101)
    t = pick_threshold(neg, 0.95)
    assert np.mean(neg >= t) <= 0.05 and np.mean(neg >= t) > 0.03


def test_wilson_edges():
    w = wilson(0, 5)
    assert w["p"] == 0 and w["lo"] == 0 and 0.4 < w["hi"] < 0.5
    assert wilson(0, 0)["p"] is None


def test_ac1_matches_hand_calculation():
    # A1 vs A2 "natural AAE" on 98 shared tweets: 3 both reject, 3 only A1, 17 only A2, 75 both accept
    a = np.array([0] * 3 + [0] * 3 + [1] * 17 + [1] * 75)
    b = np.array([0] * 3 + [1] * 3 + [0] * 17 + [1] * 75)
    assert abs(gwet_ac1(a, b) - 0.735) < 0.01


def test_auc_ci_brackets_point():
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(50), np.zeros(50)].astype(int)
    s = y + rng.normal(0, 1, 100)
    r = auc_ci(y, s, n_boot=200)
    assert r["lo"] <= r["auc"] <= r["hi"]
