"""Unit tests for analyze-time sanity checks and source-alias detection."""
import unittest

import numpy as np

from apart_epistemics.metrics import auroc
from apart_epistemics.probing.layerwise import fit_layerwise
from apart_epistemics.probing.sanity import (confound_flags, length_baselines, shuffled_label_control,
                                            source_label_crosstab, transfer_summary)


def _row(i, dataset, source, answerable, split, question="What is this?", context=None):
    return {"id": f"{dataset}:{i}", "dataset": dataset, "source": source, "answerable": answerable,
            "split": split, "question": question, "context": context}


def _splits(n, fr=(0.6, 0.2, 0.2)):
    ntr, nva = int(n * fr[0]), int(n * fr[1])
    return ["train"] * ntr + ["val"] * nva + ["test"] * (n - ntr - nva)


class CrosstabTest(unittest.TestCase):
    def test_detects_perfect_source_alias(self):
        rows = ([_row(i, "selfaware", "SelfAware", False, s) for i, s in enumerate(_splits(100))]
                + [_row(100 + i, "selfaware", "triviaqa_train", True, s) for i, s in enumerate(_splits(100))])
        rec = source_label_crosstab(rows)["selfaware"]
        self.assertTrue(rec["perfect_source_alias"])
        self.assertEqual(rec["source_label_overlap"], [])
        self.assertEqual(rec["counts"]["SelfAware"], {"answerable": 0, "unanswerable": 100})
        self.assertAlmostEqual(rec["source_feature_test_auroc"], 1.0)
        flags = confound_flags(source_label_crosstab(rows))
        self.assertEqual(flags, {"selfaware": "source_label_alias"})
        # pooled is flagged only when a confounded subset is merged with another dataset
        mixed = rows + [_row(200 + i, "squad_v2", "squad_v2-dev", i % 2 == 0, s) for i, s in enumerate(_splits(100))]
        self.assertEqual(confound_flags(source_label_crosstab(mixed))["pooled"], "includes_source_confounded_subset")

    def test_shared_source_is_not_an_alias(self):
        rows = [_row(i, "squad_v2", "squad_v2-dev", i % 2 == 0, s) for i, s in enumerate(_splits(200))]
        rec = source_label_crosstab(rows)["squad_v2"]
        self.assertFalse(rec["perfect_source_alias"])
        self.assertEqual(rec["source_label_overlap"], ["squad_v2-dev"])
        self.assertIsNone(rec["source_feature_test_auroc"])  # only one source
        self.assertEqual(confound_flags(source_label_crosstab(rows)), {})


class LengthBaselineTest(unittest.TestCase):
    def test_length_predicts_when_it_is_the_label_cue(self):
        rows = []
        splits = _splits(200)
        for i, s in enumerate(splits):
            ans = i % 2 == 0
            q = ("word " * (20 if ans else 4)).strip() + "?"
            rows.append(_row(i, "selfaware", "synthetic", ans, s, question=q))
        rec = length_baselines(rows)["selfaware"]
        self.assertGreater(rec["logistic"]["question_words"]["test_auroc"], 0.95)
        self.assertGreater(rec["univariate_test_auroc"]["question_words"]["raw"], 0.95)

    def test_length_is_weak_when_questions_are_matched(self):
        rows = [_row(i, "selfaware", "synthetic", i % 2 == 0, s, question="same length question ?")
                for i, s in enumerate(_splits(200))]
        rec = length_baselines(rows)["selfaware"]
        # all identical -> AUROC undefined (None after clean) or 0.5 if some path returns nan
        v = rec["logistic"]["question_words"]["test_auroc"]
        self.assertTrue(v is None or (isinstance(v, float) and (v != v or abs(v - 0.5) < 1e-9)), v)


class ShuffleControlTest(unittest.TestCase):
    def test_shuffled_labels_stay_near_chance_when_signal_is_real(self):
        rng = np.random.default_rng(0)
        n, L, d = 180, 4, 16
        y = np.array([i % 2 for i in range(n)])  # alternates, so hash-free splits stay mixed
        split = np.array(_splits(n))
        direction = rng.normal(size=d)
        H = rng.normal(size=(n, L, d))
        H[:, 2, :] += (2 * y[:, None] - 1) * direction
        real = fit_layerwise(H, y, split, seed=0)
        self.assertGreater(real.test_auroc[real.best_layer], 0.85)
        rec = shuffled_label_control(H, y, split, n_shuffles=8, seed=0)
        sm = rec["summary"]
        self.assertEqual(sm["n"], 8)
        self.assertLess(sm["max"], 0.85)
        self.assertLess(sm["mean"], 0.7)


class TransferSummaryTest(unittest.TestCase):
    def test_flattens_transfer_in(self):
        results = {"squad_v2": {"transfer_in": {"selfaware": {
            "test_auroc_by_layer": [0.5, 0.6, 0.7], "at_source_best_layer": 0.6}}}}
        out = transfer_summary(results)
        self.assertEqual(out["selfaware_to_squad_v2"]["max_over_layers"], 0.7)
        self.assertEqual(out["selfaware_to_squad_v2"]["argmax_layer"], 2)
        self.assertEqual(out["selfaware_to_squad_v2"]["at_source_best_layer"], 0.6)


class AurocSanity(unittest.TestCase):
    def test_perfect_ranking(self):
        self.assertEqual(auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0)


if __name__ == "__main__":
    unittest.main()
