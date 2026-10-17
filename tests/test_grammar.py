"""Each rule must fire on its positives and stay silent on its negatives.
Negatives are SAE sentences that share surface words with the pattern."""
import pytest

from aaeid.grammar import RULES, count, detect

CASES = {
    "habitual_be": (
        ["he be working late", "they always be talking", "my cousin be trippin", "she steady be lying"],
        ["he will be working late", "to be or not", "it could be nothing", "they might be coming"],
    ),
    "neg_concord": (
        ["i ain't got no money", "don't nobody care", "he can't find nothing", "we never had none"],
        ["i don't have any money", "nobody cares", "i can't. no way", "he never said that"],
    ),
    "neg_inversion": (
        ["ain't nobody coming", "can't nobody stop me", "don't nothing change"],
        ["nobody is coming", "no one can stop me", "i can't do nothing about it"],
    ),
    "completive_done": (
        ["she done left already", "we done told you", "they done finished it"],
        ["are you done yet", "i'm done with it", "she is done", "we done it before"],
    ),
    "third_sg_dont": (
        ["he don't care", "she don't know", "it don't matter"],
        ["he doesn't care", "they don't know", "i don't mind"],
    ),
    "copula_absence": (
        ["he at work", "she gon call you", "they finna leave", "you my dawg", "we going to the store",
         "lol he so funny", "and they mad"],
        ["i told you so", "give you my number", "they bring food", "he is at work",
         "we sing together", "she is going home"],
    ),
    "future_finna": (["i'm finna eat", "she bouta leave", "we fixin to go"], ["i'm about to eat", "she will leave"]),
    "aint": (["it ain't right", "that aint it"], ["it is not right", "the paint dried"]),
}


def test_every_rule_has_cases():
    assert set(CASES) == set(RULES)


@pytest.mark.parametrize("rule", list(CASES))
def test_rule_positives(rule):
    for s in CASES[rule][0]:
        assert detect(s)[rule], f"{rule} should fire on: {s!r}"


@pytest.mark.parametrize("rule", list(CASES))
def test_rule_negatives(rule):
    for s in CASES[rule][1]:
        assert not detect(s)[rule], f"{rule} should NOT fire on: {s!r}"


def test_plain_sae_fires_nothing():
    for s in ["I am going to the store later.", "She works at the hospital every day.",
              "They do not have any money right now."]:
        assert count(s) == 0, s
