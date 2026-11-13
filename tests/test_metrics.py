import math
import unittest

import numpy as np

from apart_epistemics.metrics import (abstention_stats, auroc, aurc, bootstrap_ci, coverage_accuracy, ece, gap_rate,
                                      gating_curve, recognition_action_gap, reliability_bins, selective_accuracy_at,
                                      youden_threshold, brier)


class MetricsTest(unittest.TestCase):
    def test_auroc_perfect_random_ties_nan(self):
        self.assertEqual(auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0)
        self.assertEqual(auroc([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9]), 0.0)
        self.assertEqual(auroc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]), 0.5)
        self.assertTrue(math.isnan(auroc([1, 1], [0.1, 0.2])))
        rng = np.random.default_rng(0)
        self.assertAlmostEqual(auroc(rng.integers(0, 2, 4000), rng.random(4000)), 0.5, delta=0.03)

    def test_auroc_matches_sklearn(self):
        from sklearn.metrics import roc_auc_score
        rng = np.random.default_rng(1)
        y = rng.integers(0, 2, 300)
        s = rng.normal(size=300) + y * 0.7
        self.assertAlmostEqual(auroc(y, s), roc_auc_score(y, s), places=10)

    def test_nan_is_dropped_not_zero(self):
        self.assertEqual(auroc([0, 0, 1, 1, 1], [0.1, 0.2, 0.8, 0.9, np.nan]), 1.0)

    def test_ece_perfect_and_overconfident(self):
        rng = np.random.default_rng(2)
        c = rng.random(20000)
        y = rng.random(20000) < c                       # calibrated by construction
        self.assertLess(ece(c, y), 0.02)
        self.assertAlmostEqual(ece(np.ones(100), np.zeros(100)), 1.0)
        self.assertAlmostEqual(ece(np.full(100, 0.9), np.r_[np.ones(50), np.zeros(50)]), 0.4)
        self.assertAlmostEqual(brier([1.0, 0.0], [1, 0]), 0.0)

    def test_reliability_bins_counts(self):
        bins = reliability_bins([0.05, 0.15, 0.95, 1.0], [0, 0, 1, 1], 10)
        self.assertEqual(sum(b["n"] for b in bins), 4)
        self.assertEqual(bins[9]["n"], 2)
        self.assertIsNone(bins[4]["accuracy"])

    def test_coverage_accuracy_and_aurc(self):
        score = [0.9, 0.8, 0.3, 0.2]
        correct = [1, 1, 0, 0]
        cov, acc = coverage_accuracy(score, correct)
        np.testing.assert_allclose(cov, [0.25, 0.5, 0.75, 1.0])
        np.testing.assert_allclose(acc, [1, 1, 2 / 3, 0.5])
        self.assertEqual(selective_accuracy_at(score, correct, 0.5), 1.0)
        self.assertLess(aurc(score, correct), aurc(score[::-1], correct))   # good ranking has lower risk

    def test_abstention_stats(self):
        # 4 items: 2 answerable (one the model would get right), 2 unanswerable
        ans = [True, True, False, False]
        corr = [True, False, False, False]
        r = abstention_stats([False, True, True, False], ans, corr)
        self.assertEqual(r["abstention_precision"], 0.5)
        self.assertEqual(r["abstention_recall"], 0.5)
        self.assertEqual(r["false_abstention_rate"], 0.0)
        self.assertEqual(r["coverage"], 0.5)
        self.assertEqual(r["accuracy_answered"], 0.5)
        r2 = abstention_stats([True, False, False, False], ans, corr)
        self.assertEqual(r2["false_abstention_rate"], 1.0)

    def test_gating_curve_monotone_coverage(self):
        rng = np.random.default_rng(3)
        ans = rng.random(500) < 0.5
        score = rng.normal(size=500) + 2 * ans
        corr = ans & (rng.random(500) < 0.8)
        rows = gating_curve(score, ans, corr)
        cov = [r["coverage"] for r in rows]
        self.assertEqual(cov, sorted(cov, reverse=True))
        self.assertEqual(rows[-1]["coverage"], 0.0)

    def test_youden(self):
        t = youden_threshold([0, 0, 0, 1, 1, 1], [0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        self.assertAlmostEqual(t, 0.7)

    def test_gap_rate(self):
        r = gap_rate([True, True, False, True], [0.9, 0.5, 0.95, 0.85], [True, True, True, False], [False, False, True, False])
        self.assertEqual(r["n_gap"], 1)
        self.assertEqual(r["gap_rate"], 0.25)
        self.assertEqual(r["error_rate_in_gap"], 1.0)

    def test_recognition_action_gap(self):
        ans = np.array([1, 1, 0, 0], bool)
        probe = np.array([2, 1, -1, -2.0])
        abst = np.array([0, 0, 0, 0], bool)             # model never abstains
        r = recognition_action_gap(probe, ans, abst, thr=0.0)
        self.assertEqual(r["recognition"], 1.0)
        self.assertEqual(r["action"], 0.5)
        self.assertEqual(r["gap"], 0.5)

    def test_bootstrap_ci_contains_estimate(self):
        rng = np.random.default_rng(4)
        y = rng.integers(0, 2, 400)
        s = rng.normal(size=400) + y
        lo, hi = bootstrap_ci(auroc, [y, s], n_boot=200)
        self.assertLess(lo, auroc(y, s))
        self.assertGreater(hi, auroc(y, s))


if __name__ == "__main__":
    unittest.main()
