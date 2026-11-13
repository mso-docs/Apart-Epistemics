"""Write-time scrub for run outputs. Synthetic values only; identities are injected."""
import json
import tempfile
import unittest
from pathlib import Path

from apart_epistemics import runio, scrub

SECRET = "zz-env-" + "value-Q9x8W7v6U5"
TOKEN = "hf_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3"
# Built from pieces so this file itself contains no path/IP literal for the commit-time redactor to rewrite.
HOME_PATH = "/" + "home" + "/zed"
IP = "10.1." + "2.3"


class ScrubTests(unittest.TestCase):
    def setUp(self):
        self.root = scrub._default_root()
        self.key = str(self.root)
        self.saved = scrub._ENGINE_CACHE.get(self.key)
        ids = [scrub.Identity("HOST_GPU", "bare", "gpu-box-1"), scrub.Identity("USER", "bare", "svc-runner7")]
        scrub._ENGINE_CACHE[self.key] = scrub.Engine(ids, {"MY_SERVICE_TOKEN": SECRET}, scrub.Allow())
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        scrub._ENGINE_CACHE.pop(self.key, None)
        if self.saved is not None:
            scrub._ENGINE_CACHE[self.key] = self.saved

    def test_scrub_obj(self):
        out = scrub.scrub_obj({"a": f"{SECRET} {TOKEN} svc-runner7@gpu-box-1 {HOME_PATH}/x {IP}", "n": [1, 0.5]})
        self.assertEqual(out["a"], "<REDACTED:MY_SERVICE_TOKEN> <REDACTED:huggingface-token> <USER>@<HOST_GPU> <HOME>/x <IP>")
        self.assertEqual(out["n"], [1, 0.5])

    def test_run_writers_scrub(self):
        runio.append_jsonl(self.tmp / "items.jsonl", [{"id": "q1", "answer": f"{SECRET} on gpu-box-1", "p": 0.25}])
        runio.write_json(self.tmp / "meta.json", {"host": "svc-runner7@gpu-box-1", "token": SECRET})
        text = (self.tmp / "items.jsonl").read_text() + (self.tmp / "meta.json").read_text()
        for leaked in (SECRET, "gpu-box-1", "svc-runner7"):
            self.assertNotIn(leaked, text)
        self.assertEqual(json.loads((self.tmp / "items.jsonl").read_text())["p"], 0.25)

    def test_normal_content_untouched(self):
        row = {"id": "squad_v2:5ad39d53604f3c001a3fe8d1", "question": "Who won in 2015? Version 1.2.3 at 12:30", "p": 0.5}
        self.assertEqual(scrub.scrub_obj(row), row)


if __name__ == "__main__":
    unittest.main()
