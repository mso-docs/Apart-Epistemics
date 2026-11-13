import json
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from apart_epistemics import guard
from apart_epistemics.analyze import run_analyze
from apart_epistemics.config import ROOT, load_config
from apart_epistemics.data.loaders import (Item, assign_splits, balanced_subsample, load_selfaware, load_squad_v2)
from apart_epistemics.generate import run_generate
from apart_epistemics.model import GenOut
from apart_epistemics.probing import fit_layerwise, project_scores
from apart_epistemics.runio import read_jsonl, load_hidden
from apart_epistemics.synthetic import make_synthetic_run

ET = ZoneInfo("America/New_York")


def tmp_cfg(tmp, name="synthetic"):
    cfg = load_config(ROOT / "configs" / f"{name}.yaml")
    cfg["outputs"]["dir"] = str(tmp)
    return cfg


class ProbeTest(unittest.TestCase):
    def test_layerwise_probe_finds_planted_signal(self):
        rng = np.random.default_rng(0)
        n, L, d = 400, 6, 20
        y = rng.integers(0, 2, n)
        direction = rng.normal(size=d)
        H = rng.normal(size=(n, L, d))
        H[:, 4, :] += (2 * y[:, None] - 1) * direction * 1.2        # planted at layer 4 only
        split = np.array(["train"] * 240 + ["val"] * 80 + ["test"] * 80)
        res = fit_layerwise(H, y, split, seed=0)
        self.assertEqual(res.best_layer, 4)
        self.assertGreater(res.test_auroc[4], 0.9)
        self.assertLess(max(res.test_auroc[i] for i in (0, 1, 2)), 0.7)
        s = project_scores(res, H)
        self.assertEqual(s.shape, (n,))
        mm = fit_layerwise(H, y, split, kind="massmean")
        self.assertEqual(mm.best_layer, 4)

    def test_probe_requires_both_classes(self):
        with self.assertRaises(ValueError):
            fit_layerwise(np.zeros((10, 2, 3)), np.ones(10), np.array(["train"] * 10))


class DataTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_squad_loader(self):
        data = {"data": [{"title": "A", "paragraphs": [{"context": "ctx", "qas": [
            {"id": "1", "question": "q1", "is_impossible": False, "answers": [{"text": "x"}, {"text": "x"}]},
            {"id": "2", "question": "q2", "is_impossible": True, "answers": [], "plausible_answers": [{"text": "y"}]}]}]}]}
        f = self.tmp / "dev-v2.0.json"
        f.write_text(json.dumps(data))
        items = load_squad_v2(f)
        self.assertEqual([i.answerable for i in items], [True, False])
        self.assertEqual(items[0].gold, ["x"])
        self.assertEqual(items[1].gold, [])                      # plausible answers must not leak as gold
        self.assertEqual(items[1].unanswerable_type, "contextual")

    def test_selfaware_loader(self):
        f = self.tmp / "sa.json"
        f.write_text(json.dumps({"example": [
            {"question_id": 1, "question": "a?", "answer": ["song"], "answerable": True, "source": "hotpot_train"},
            {"question_id": 2, "question": "b?", "answer": None, "answerable": False, "source": "SelfAware"}]}))
        items = load_selfaware(f)
        self.assertEqual(items[1].unanswerable_type, "inherent")
        self.assertEqual(items[0].gold, ["song"])

    def test_split_is_deterministic_grouped_and_balanced_subsample(self):
        items = [Item(id=f"s:{i}", dataset="squad_v2", question="q", answerable=i % 2 == 0, group=f"art{i % 10}") for i in range(2000)]
        fr = {"train": 0.6, "val": 0.2, "test": 0.2}
        assign_splits(items, 1, fr)
        by_group = {}
        for it in items:
            by_group.setdefault(it.group, set()).add(it.split)
        self.assertTrue(all(len(v) == 1 for v in by_group.values()))      # no passage shared across splits
        again = [Item(id=i.id, dataset=i.dataset, question="q", answerable=i.answerable, group=i.group) for i in items]
        assign_splits(again, 1, fr)
        self.assertEqual([i.split for i in items], [i.split for i in again])
        sub = balanced_subsample(items, 100, 1)
        self.assertEqual(sum(i.answerable for i in sub), 100)
        self.assertEqual(sum(not i.answerable for i in sub), 100)

    def test_splits_are_stratified_by_class_for_singleton_groups(self):
        items = [Item(id=f"sa:{i}", dataset="selfaware", question="q", answerable=i < 100,
                      group=f"selfaware:{i}") for i in range(200)]
        fr = {"train": 0.6, "val": 0.2, "test": 0.2}
        assign_splits(items, 13, fr)
        for flag in (True, False):
            sub = [it for it in items if it.answerable is flag]
            counts = {s: sum(it.split == s for it in sub) for s in ("train", "val", "test")}
            self.assertEqual(counts, {"train": 60, "val": 20, "test": 20}, counts)
        # leftover counts still sum and stay deterministic
        again = [Item(id=i.id, dataset=i.dataset, question="q", answerable=i.answerable, group=i.group) for i in items]
        assign_splits(again, 13, fr)
        self.assertEqual([i.split for i in items], [i.split for i in again])

    def test_real_data_if_downloaded(self):
        p = ROOT / "data" / "raw" / "selfaware" / "SelfAware.json"
        if not p.exists():
            self.skipTest("run scripts/download_data.py first")
        items = load_selfaware(p)
        self.assertEqual(len(items), 3369)
        self.assertEqual(sum(not i.answerable for i in items), 1031)     # note: plan text says 1,032 / 2,337


class FakeModel:
    """Stands in for Phi: deterministic, no torch, no GPU."""
    def generate(self, messages, want_hidden=True, do_sample=False, temperature=0.7, max_new=None):
        user = messages[-1]["content"]
        if "confident" in user:
            return GenOut("90", None, -0.1, -0.1, -0.1, 0.1, 0.1, 1)
        h = np.random.default_rng(abs(hash(user)) % 1000).normal(size=(5, 8)).astype(np.float32) if want_hidden else None
        return GenOut("some answer", h, -0.5, -0.1, -1.0, 0.7, 0.9, 3)


class PipelineTest(unittest.TestCase):
    def test_generate_refuses_before_5pm_weekday(self):
        tmp = Path(tempfile.mkdtemp())
        cfg = tmp_cfg(tmp)
        with self.assertRaises(guard.OutsideRunWindow):
            run_generate(cfg, "T-1", model=FakeModel(), now=datetime(2026, 11, 13, 10, 0, tzinfo=ET))
        self.assertFalse((tmp / "T-1" / "items.jsonl").exists())             # nothing ran

    @unittest.skipUnless((ROOT / "data" / "raw" / "squad_v2" / "dev-v2.0.json").exists(), "download data first")
    def test_generate_resume_with_fake_model(self):
        tmp = Path(tempfile.mkdtemp())
        cfg = tmp_cfg(tmp, "dev")
        for d in cfg["datasets"].values():
            d["n_per_class"] = 3
        sat = datetime(2026, 11, 14, 10, 0, tzinfo=ET)
        run_generate(cfg, "T-2", limit=8, model=FakeModel(), shard_size=4, now=sat)
        rows = read_jsonl(tmp / "T-2" / "items.jsonl")
        self.assertEqual(len(rows), 8)
        ids, H = load_hidden(tmp / "T-2")
        self.assertEqual(sorted(ids), sorted(r["id"] for r in rows))
        run_generate(cfg, "T-2", limit=8, model=FakeModel(), shard_size=4, now=sat)       # resume: nothing duplicated
        self.assertEqual(len(read_jsonl(tmp / "T-2" / "items.jsonl")), 8)

    def test_analyze_synthetic_end_to_end(self):
        tmp = Path(tempfile.mkdtemp())
        cfg = tmp_cfg(tmp)
        cfg["metrics"]["bootstrap"] = 30
        cfg["metrics"]["n_shuffles"] = 5
        make_synthetic_run(cfg, "SYNTH-t--00000000", n_per_class=80)
        out = run_analyze(cfg, "SYNTH-t--00000000")
        self.assertEqual(out["meta"]["primary_probe_dataset"], "squad_v2")
        self.assertIn("sanity", out)
        self.assertIn("source_label_crosstab", out["sanity"])
        self.assertIn("shuffled_label", out["sanity"])
        self.assertIn("length_baseline", out["sanity"])
        self.assertIn("transfer", out["sanity"])
        # synthetic source is shared across classes: not a source alias
        self.assertFalse(out["sanity"]["source_label_crosstab"]["selfaware"]["perfect_source_alias"])
        self.assertNotIn("selfaware", out["meta"]["confounded_datasets"])
        self.assertTrue((tmp / "SYNTH-t--00000000" / "analysis" / "sanity.json").exists())
        sq = out["datasets"]["squad_v2"]
        self.assertGreater(sq["probe_logistic"]["test_auroc"][sq["probe_logistic"]["best_layer"]], 0.85)
        self.assertGreaterEqual(sq["probe_logistic"]["best_layer"], 3)          # planted signal grows with depth
        self.assertLess(sq["probe_logistic"]["test_auroc"][0], 0.75)
        # synthetic model is overconfident by construction: gap and ECE must show it
        self.assertGreater(sq["calibration"]["verbal_confidence"]["ece"], 0.3)
        self.assertGreater(sq["gap"]["recognition_action"]["gap"], 0.0)
        self.assertIn("selfaware", sq["transfer_in"])
        self.assertTrue((tmp / "SYNTH-t--00000000" / "analysis" / "metrics.json").exists())
        json.loads((tmp / "SYNTH-t--00000000" / "analysis" / "metrics.json").read_text())   # strict JSON (no NaN)


if __name__ == "__main__":
    unittest.main()
