import math
import unittest

from apart_epistemics.analyze import attach_abstain_option
from apart_epistemics.metrics import abstention_decision_metrics, pr_f1


class AbstainOptionMetricsTest(unittest.TestCase):
    def test_pr_f1(self):
        self.assertEqual(pr_f1(1, 1), 1.0)
        self.assertEqual(pr_f1(0.5, 0.5), 0.5)
        self.assertTrue(math.isnan(pr_f1(float("nan"), 0.5)))
        self.assertTrue(math.isnan(pr_f1(0.0, 0.0)))

    def test_perfect_and_never_abstain(self):
        ans = [True, True, False, False]
        perfect = abstention_decision_metrics([False, False, True, True], ans)
        self.assertEqual(perfect["abstention_precision"], 1.0)
        self.assertEqual(perfect["abstention_recall"], 1.0)
        self.assertEqual(perfect["abstention_f1"], 1.0)
        self.assertEqual(perfect["decision_accuracy"], 1.0)
        self.assertEqual(perfect["auroc_answerability"], 1.0)
        self.assertEqual(perfect["abstain_rate"], 0.5)
        self.assertEqual(perfect["abstain_rate_on_unanswerable"], 1.0)
        self.assertEqual(perfect["abstain_rate_on_answerable"], 0.0)

        never = abstention_decision_metrics([False, False, False, False], ans)
        self.assertTrue(math.isnan(never["abstention_precision"]))
        self.assertEqual(never["abstention_recall"], 0.0)
        self.assertTrue(math.isnan(never["abstention_f1"]))
        self.assertEqual(never["decision_accuracy"], 0.5)
        self.assertEqual(never["auroc_answerability"], 0.5)  # all scores tied
        self.assertEqual(never["abstain_rate"], 0.0)

    def test_partial_gate_with_correct_if_answered(self):
        # 2 answerable (first would be right), 2 unanswerable; abstain on 2nd and 3rd
        r = abstention_decision_metrics(
            [False, True, True, False],
            [True, True, False, False],
            [True, False, False, False],
        )
        self.assertEqual(r["abstention_precision"], 0.5)
        self.assertEqual(r["abstention_recall"], 0.5)
        self.assertEqual(r["abstention_f1"], 0.5)
        self.assertEqual(r["decision_accuracy"], 0.5)
        self.assertEqual(r["coverage"], 0.5)
        self.assertEqual(r["accuracy_answered"], 0.5)
        self.assertEqual(r["false_abstention_rate"], 0.0)
        self.assertAlmostEqual(r["auroc_answerability"], 0.5)


def _row(i, dataset, answerable, split, response, option, gold=None, correct=None, **kw):
    return {
        "id": f"{dataset}:{i}", "dataset": dataset, "question": "q", "answerable": answerable,
        "gold": (gold if gold is not None else (["Paris"] if answerable else [])),
        "context": "Paris is in France." if dataset == "squad_v2" else None,
        "unanswerable_type": None if answerable else "contextual", "group": f"g{i}",
        "source": "test", "split": split, "response": response,
        "abstained": False, "correct": bool(correct) if correct is not None else False,
        "abstain_option_response": option, **kw,
    }


class AttachAbstainOptionTest(unittest.TestCase):
    def test_no_option_pass_is_noop(self):
        results = {"squad_v2": {}}
        attach_abstain_option(results, [{"dataset": "squad_v2", "split": "test", "answerable": True}])
        self.assertNotIn("abstain_option", results["squad_v2"])

    def test_scores_option_responses_and_compares_on_test(self):
        # 4 test items, balanced. Option pass abstains on both unanswerable and one answerable.
        rows = [
            _row(1, "squad_v2", True, "test", "Paris", "Paris", correct=True),
            _row(2, "squad_v2", True, "test", "Paris", "UNANSWERABLE", correct=True),
            _row(3, "squad_v2", False, "test", "Paris", "UNANSWERABLE", correct=False),
            _row(4, "squad_v2", False, "test", "Paris", "Lyon", correct=False),
            # train item should appear in `all` only
            _row(5, "squad_v2", False, "train", "Paris", "UNANSWERABLE", correct=False),
        ]
        results = {"squad_v2": {"signals": {
            "probe": {"auroc_answerability": 0.9, "val_threshold": 0.1,
                      "gating_at_val_threshold": {"coverage": 0.5, "accuracy_answered": 0.8,
                                                  "abstention_precision": 0.8, "abstention_recall": 0.8,
                                                  "false_abstention_rate": 0.1, "n": 4}},
            "verbal_confidence": {"auroc_answerability": 0.6, "val_threshold": 0.95,
                                  "gating_at_val_threshold": {"coverage": 0.9, "accuracy_answered": 0.5,
                                                              "abstention_precision": 1.0, "abstention_recall": 0.2,
                                                              "false_abstention_rate": 0.0, "n": 4}},
        }}}
        attach_abstain_option(results, rows)
        te = results["squad_v2"]["abstain_option"]["test"]
        all_ = results["squad_v2"]["abstain_option"]["all"]
        # test: TP=1, FP=1, FN=1, TN=1
        self.assertEqual(te["n"], 4)
        self.assertEqual(te["n_option_responses"], 4)
        self.assertEqual(te["n_flag_mismatch"], 0)
        self.assertEqual(te["abstain_rate"], 0.5)
        self.assertEqual(te["abstention_precision"], 0.5)
        self.assertEqual(te["abstention_recall"], 0.5)
        self.assertEqual(te["abstention_f1"], 0.5)
        self.assertEqual(te["decision_accuracy"], 0.5)
        # answered items: Paris (correct) and Lyon (unanswerable, wrong) -> accuracy 0.5
        self.assertEqual(te["accuracy_answered"], 0.5)
        # appropriate behavior: correct, false, true, false -> 0.5
        self.assertEqual(te["appropriate_behavior"], 0.5)
        # plain answers were correct on both answerable items; option abstained on one of them
        self.assertEqual(te["false_abstention_rate_using_plain_answer"], 0.5)
        self.assertEqual(all_["n"], 5)
        self.assertEqual(all_["abstention_recall"], 2 / 3)
        cmp = te["comparison"]
        self.assertEqual(cmp["abstain_option"]["abstention_f1"], 0.5)
        self.assertAlmostEqual(cmp["probe"]["abstention_f1"], 0.8)
        self.assertAlmostEqual(cmp["verbal_confidence"]["abstention_f1"], 2 * 1.0 * 0.2 / 1.2)
        self.assertEqual(cmp["probe"]["auroc_answerability"], 0.9)


if __name__ == "__main__":
    unittest.main()
