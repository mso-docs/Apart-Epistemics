import unittest

from apart_epistemics.data.loaders import Item
from apart_epistemics.scoring import (best_f1, exact_match, is_abstention, normalize, parse_confidence, score_answer,
                                      string_match)


def sq(ans=True, gold=("Denver Broncos",)):
    return Item(id="squad_v2:x", dataset="squad_v2", question="q", answerable=ans, gold=list(gold) if ans else [], context="c")


def sa(ans=True, gold=("song",)):
    return Item(id="selfaware:1", dataset="selfaware", question="q", answerable=ans, gold=list(gold) if ans else [])


class ScoringTest(unittest.TestCase):
    def test_normalize_and_em(self):
        self.assertEqual(normalize("The  Denver, Broncos!"), "denver broncos")
        self.assertTrue(exact_match("the denver broncos", ["Denver Broncos"]))
        self.assertFalse(exact_match("broncos", ["Denver Broncos"]))

    def test_f1(self):
        self.assertAlmostEqual(best_f1("Denver", ["Denver Broncos"]), 2 * 1 * 0.5 / 1.5)
        self.assertEqual(best_f1("", [""]), 1.0)

    def test_string_match(self):
        self.assertTrue(string_match("It is a song.", ["song"]))
        self.assertFalse(string_match("a poem", ["song"]))

    def test_abstention_detection(self):
        for t in ("UNANSWERABLE", "I don't know.", "The passage does not mention this: not mentioned", "It cannot be answered."):
            self.assertTrue(is_abstention(t), t)
        for t in ("Denver Broncos", "", "42"):
            self.assertFalse(is_abstention(t), repr(t))

    def test_score_answer_semantics(self):
        self.assertTrue(score_answer(sq(), "Denver Broncos")["correct"])
        self.assertFalse(score_answer(sq(), "UNANSWERABLE")["correct"])                # abstaining on an answerable item is not "right"
        self.assertTrue(score_answer(sq(), "UNANSWERABLE")["abstained"])
        self.assertTrue(score_answer(sq(ans=False), "UNANSWERABLE")["correct"])         # abstaining on unanswerable is appropriate
        self.assertFalse(score_answer(sq(ans=False), "Denver Broncos")["correct"])
        self.assertTrue(score_answer(sa(), "a song")["correct"])

    def test_parse_confidence(self):
        self.assertEqual(parse_confidence("85"), 0.85)
        self.assertEqual(parse_confidence("I'd say 70%"), 0.7)
        self.assertIsNone(parse_confidence("I am very sure"))
        self.assertIsNone(parse_confidence("1000"))        # out of range = missing, not clipped
        self.assertIsNone(parse_confidence(None))


if __name__ == "__main__":
    unittest.main()
