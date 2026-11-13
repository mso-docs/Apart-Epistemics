import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

from apart_epistemics.analyze import run_analyze
from apart_epistemics.config import ROOT, load_config
from apart_epistemics.evalboard_export import attach_interp_refs, build_artifacts, build_traces
from apart_epistemics.runio import read_jsonl
from apart_epistemics.synthetic import make_synthetic_run

TOOLKIT = Path.home() / "Documents" / "Code" / "modular_dashboard"
RUN = "SYNTH-t--00000000"


def built():
    tmp = Path(tempfile.mkdtemp())
    cfg = load_config(ROOT / "configs" / "synthetic.yaml")
    cfg["outputs"]["dir"] = str(tmp)
    cfg["metrics"]["bootstrap"] = 20
    make_synthetic_run(cfg, RUN, n_per_class=60)
    run_analyze(cfg, RUN)
    rdir = tmp / RUN
    metrics = json.loads((rdir / "analysis/metrics.json").read_text())
    curves = json.loads((rdir / "analysis/curves.json").read_text())
    ps = {r["id"]: r["probe_score"] for r in read_jsonl(rdir / "analysis/probe_scores.jsonl")}
    rows = read_jsonl(rdir / "items.jsonl")
    traces = build_traces(rdir, RUN, "m", metrics, ps)
    arts = build_artifacts(RUN, "m", metrics, curves, rows, ps, synthetic=True)
    attach_interp_refs(traces, arts, RUN)
    return cfg, tmp, traces, arts


class ExportTest(unittest.TestCase):
    def test_shapes_and_groups(self):
        _, _, traces, arts = built()
        self.assertEqual(len(traces), 240)
        runs = {t["run_id"] for t in traces}
        self.assertEqual(len(runs), 4)                                   # dataset x answerability
        self.assertTrue(all(r.startswith(RUN + ":") for r in runs))
        t = traces[0]
        self.assertIn("appropriate_behavior", {e["metric"] for e in t["evaluations"]})
        self.assertTrue(all(0.0 <= e["value"] <= 1.0 for tr in traces for e in tr["evaluations"]))
        kinds = {a["id"].split(":", 1)[1].rsplit("-", 1)[0] for a in arts}
        for k in ("probe-auroc-by-layer", "probe-accuracy-by-layer", "reliability", "coverage-accuracy", "abstention-pr", "false-abstention"):
            self.assertTrue(any(i.startswith(k) for i in kinds) or any(a["id"].split(":", 1)[1].startswith(k) for a in arts), k)
        self.assertTrue(all(a.get("synthetic") for a in arts))           # synthetic runs are always labeled
        self.assertTrue(all(t.get("synthetic") for t in traces))

    def test_missing_stays_missing(self):
        _, _, traces, _ = built()
        # probe_correct exists only on test-split items; other items must not carry a fake 0
        by = [e["metric"] for t in traces if t["metadata"]["split"] != "test" for e in t["evaluations"]]
        self.assertNotIn("probe_correct", by)
        self.assertTrue(all(("answer_correct" in {e["metric"] for e in t["evaluations"]}) == t["metadata"]["answerable"] for t in traces))

    @unittest.skipUnless((TOOLKIT / "evalboard" / "core.py").exists(), "Evalboard toolkit not on this machine")
    def test_validates_and_ingests_into_evalboard(self):
        sys.path.insert(0, str(TOOLKIT))
        from evalboard.core import Store
        _, tmp, traces, arts = built()
        store = Store(tmp / "ws.sqlite3")
        self.assertEqual(store.ingest(traces), len(traces))
        self.assertEqual(store.ingest_interp(arts), len(arts))
        self.assertEqual(store.ingest(traces), len(traces))              # idempotent retry
        layout = json.loads((ROOT / "configs" / "evalboard_layout.json").read_text())
        layout = {k: v for k, v in layout.items() if not k.startswith("_")}
        saved = store.save_layout(layout)
        self.assertEqual([m["key"] for m in saved["metrics"] if m.get("primary")], ["appropriate_behavior"])

    def test_ingest_script_refuses_synthetic_into_real_project(self):
        spec = importlib.util.spec_from_file_location("ingest_evalboard", ROOT / "scripts" / "ingest_evalboard.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cfg, tmp, _, _ = built()
        import os
        os.environ.pop("APART_EVALBOARD_URL", None)
        cfgfile = tmp / "cfg.yaml"
        import yaml
        c = {k: v for k, v in cfg.items() if not k.startswith("_")}
        c["outputs"] = {"dir": str(tmp)}
        cfgfile.write_text(yaml.safe_dump(c))
        with self.assertRaises(SystemExit) as cm:
            mod.main(["--run", RUN, "--config", str(cfgfile), "--dry-run"])
        self.assertIn("synthetic", str(cm.exception).lower())
        with self.assertRaises(SystemExit):
            mod.main(["--run", RUN, "--config", str(cfgfile), "--db", str(tmp / "real.sqlite3")])
        self.assertEqual(mod.main(["--run", RUN, "--config", str(cfgfile), "--db", str(tmp / "x-DEMO.sqlite3"), "--dry-run"]), 0)


if __name__ == "__main__":
    unittest.main()
