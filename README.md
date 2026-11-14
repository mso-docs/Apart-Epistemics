# When does a model know it should abstain?

Apart Research **AI x Epistemics sprint** (Fri Nov 13 – Sun Nov 15, 2026; submission cutoff Sun 11:59 PM AoE = Mon Nov 16, 6:59 AM ET), Track 1: model epistemics & decision-making evals.

All code, experiments and writing in this repository were produced during the sprint (Nov 13–15, 2026); there is no pre-sprint code.

Question: does a small open-weights model (Phi-3.5-mini-instruct) internally encode that a question is **unanswerable** even when it answers confidently? We fit layer-wise linear probes on last-prompt-token hidden states and compare them with verbalized confidence, token log-probability / entropy, and sampling consistency, on SQuAD 2.0 (no evidence in the passage) and SelfAware (inherently unanswerable). Full plan, related work, risks and schedule: **[docs/PROPOSAL.md](docs/PROPOSAL.md)**.

> **Status:** first 400-item dev run is in (`20261113T221506Z-dev--apart`, Phi-3.5-mini 8-bit). **The primary probe result is SQuAD 2.0** (contextual unanswerability). SelfAware and pooled probe AUROCs are reported with a source/label confound warning (see Data notes) and are not headline numbers.
> **Version control:** weekday commits and pushes are blocked 8 AM–6 PM ET by a global guard. Do not bypass it.

## Layout

```
src/apart_epistemics/   config, guard, budget, data, prompts, scoring, model, generate, probing, metrics, judge, analyze, evalboard_export, cli
configs/                default.yaml (full), dev.yaml (400-item smoke slice), full.yaml (adds sampling consistency), synthetic.yaml (tests/demo only), evalboard_layout.json
scripts/                download_data.py, run_pipeline.py (time-guarded), run_remote.sh, sync_*.sh, ingest_evalboard.py, setup_evalboard.py,
                        evalboard_service.sh, make_demo_project.py, build_release.sh, prepublish_scan.sh, pii_scan.py
tests/                  unit tests on synthetic data (python -m unittest)
docs/PROPOSAL.md       the sprint plan (Markdown copy of the planning page)
data/raw/               SQuAD 2.0 + SelfAware (git-ignored; re-create with scripts/download_data.py)
models/ outputs/ logs/  weights, run outputs, logs (git-ignored)
```

## Pipeline

1. **generate** (needs the GPU host): for each item, a greedy short answer with token log-probs and entropy; the residual-stream vector at the **last prompt token for every layer** (embeddings + 32 layers = 33 × 3072, stored fp16); a second turn eliciting **verbalized confidence** (0–100); optional "UNANSWERABLE allowed" behavior pass; optional sampling consistency on a subset. Resumable (`items.jsonl` + `hidden/*.npz`).
2. **analyze** (CPU): per-dataset, pooled and cross-dataset-transfer probes (logistic regression, plus mass-mean as a robustness check); best layer chosen on **val only**; test-split metrics: probe/baseline AUROC for answerability and correctness, **ECE** and Brier of verbalized confidence and a token-probability proxy, coverage–accuracy and **selective accuracy**, abstention precision / recall / **false-abstention rate**, **gap rate** (probe says "insufficient evidence" but the model commits with confidence ≥ 0.8), and the **recognition-vs-action gap** (balanced accuracy of the probe at its val threshold minus balanced accuracy of the model's own abstention behavior), with bootstrap intervals. Every analysis also writes a **sanity** block (source-by-label crosstab, shuffled-label control, prompt/question-length baseline, and cross-dataset transfer AUROC) to `analysis/metrics.json`, `analysis/sanity.json`, and the Evalboard ingest. **Report SQuAD 2.0 as the primary probe result**; mark SelfAware and pooled as confounded when source aliases the label.
3. **export / ingest**: traces and `evalboard.interp/v1` artifacts for the Evalboard project (below).
4. **judge hook** (`judge.backend`): default `none` (rule-based scoring only). `openrouter` is available for a sampled second opinion; it only runs with `APART_ALLOW_PAID_CALLS=1` and always goes through the standalone $30 spend guard (`openrouter_budget.py`, path via `APART_BUDGET_MODULE_DIR`). A call skipped for budget reasons is missing, never zero.

Scoring: SQuAD answers use exact match or token-F1 ≥ 0.5; SelfAware answers use normalized string containment (documented limitation: will miss valid paraphrases). An unanswerable item counts as handled correctly only if the model abstains. Abstention is detected by pattern (`UNANSWERABLE`, "I don't know", "cannot be answered", …).

### Data notes
- SelfAware's data file has **2,338 answerable / 1,031 unanswerable** questions (the plan text says 1,032 / 2,337); we use the file as downloaded and `data/raw/SOURCES.json` records URL, size and sha256 for each file.
- SQuAD 2.0 uses the official **dev** set (11,873 questions, 5,945 unanswerable), balanced-subsampled per config, split by **passage** so no passage appears in two splits. The train file is downloaded but not used by default.
- Splits are deterministic and **stratified by answerability class**. SQuAD still splits by passage (a passage is never shared across splits); SelfAware splits by question id, so class quotas are exact. Recorded runs keep the splits stored in `items.jsonl`.

### SelfAware source/label confound
In the official SelfAware dataset, unanswerable questions are almost all author-written items with source `SelfAware`, while answerable questions are harvested from SQuAD, TriviaQA, and HotpotQA. Source therefore aliases the label: a trivial "is the source `SelfAware`?" feature gets AUROC 1.0, and a hidden-state probe can look perfect by detecting question style rather than answerability. **Treat SQuAD 2.0 as the primary probe result.** Do not read a high SelfAware or pooled probe AUROC as evidence that hidden states encode answerability. The analyze stage flags a perfect source/label alias in the sanity crosstab; transfer of a SelfAware-trained probe onto SQuAD is the better check that the direction is general.
- TriviaQA is optional and not downloaded (license caveat unresolved; see the proposal).
- Licenses: SQuAD 2.0 and SelfAware data are CC BY-SA 4.0: attribute them in the report.

## Setup

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e . --no-deps
.venv/bin/python scripts/download_data.py          # SQuAD 2.0 + SelfAware -> data/raw/
.venv/bin/python -m pytest
.venv/bin/python -m apart_epistemics check --config configs/dev.yaml      # validate config, data, budget wiring, run window
.venv/bin/python -m apart_epistemics generate --config configs/dev.yaml --dry-run   # loads no model
cp .env.example .env                                # then edit (git-ignored)
```

### Where the model runs
Phi-3.5-mini's bf16 weights are ~7.6 GB, which does not leave room on an **8 GB GPU**, so the default config loads it **8-bit** (batch size 1, last-token hooks only; the plan's own memory fallback). The 8 GB host has a venv (`torch` cu128, `transformers`, `bitsandbytes`) and the weights under `~/apart-epistemics/` (see `scripts/setup_remote.sh`). The Mac only does CPU work: tests, analysis, ingest. **A separate local GPU server is not used for Apart** (it runs another project's queue).

### Run window
Weekday runs may only **start at or after 5 PM ET**; weekends are unrestricted. This is enforced in `apart generate`/`apart run` themselves (so no script can bypass it), in `scripts/run_pipeline.py`, and in `scripts/run_remote.sh`. There is deliberately no override flag. `--dry-run` and `check` are always allowed.

## Running

```sh
scripts/run_remote.sh configs/dev.yaml            # smoke slice (400 items) on the 8 GB host; prints the run id
scripts/sync_from_remote.sh <run_id>              # pull outputs/<run_id> and its log
.venv/bin/python scripts/ingest_evalboard.py --run <run_id> --dry-run   # validate the Evalboard payload
.venv/bin/python scripts/ingest_evalboard.py --run <run_id>             # load it into the Apart project
scripts/run_remote.sh configs/full.yaml           # the full run (adds sampling consistency + abstain-option pass)
```

The run id is `<UTC timestamp>-<config>--apart`; the remote log is `logs/<run id>.log` and is read by the run monitor.

## Dashboards (local only; never public)
- **Evalboard** project `apart-epistemics` (separate database from every other project): loopback `127.0.0.1:8767`, tailnet-only via `tailscale serve` on :8444. Pages: Overview · Answerable vs unanswerable (each *dataset × answerability* is a "run", so the Scoreboard compares them directly) · Per-question traces · Probes/calibration/abstention (layer AUROC and accuracy curves, reliability diagrams, coverage–accuracy, abstention precision/recall, false-abstention, probe-score and verbal-confidence distributions, with gap cases linked to their traces). Headline metric: *appropriate behavior* (right answer on answerable, abstention on unanswerable).
  `scripts/evalboard_service.sh install|tailnet|status`. Real data only: `scripts/ingest_evalboard.py` refuses synthetic runs; `scripts/make_demo_project.py` builds a separate, clearly labeled **SYNTHETIC** demo project.
- **Run monitor** (global `build-run-monitor` toolkit) on the Mac, reading the model host over SSH, read-only: loopback `127.0.0.1:8795`, tailnet-only on :8445. Its config lives under `~/.run-monitor/` (not in this repo: it names hosts and paths).

## Reproducibility
- Seeds, splits and subsampling are deterministic (SHA-256 of seed + id); splits are stratified by class as of this revision. The run id embeds a hash of the full config; `meta.json` stores the config snapshot and library versions.
- Greedy decoding for answers; confidence is elicited once per item with a fixed prompt (`src/apart_epistemics/prompts.py`). Quantization (8-bit) slightly changes hidden states relative to bf16: report it.
- Probes are standardized logistic regression (`C=1`), trained on train, layer chosen on val, reported on test; thresholds for gating are chosen on val, never test.
- Everything the report cites should come from `outputs/<run_id>/analysis/metrics.json` of a recorded run. Cite the SQuAD 2.0 probe as the primary number; cite SelfAware and pooled only with the source-confound caveat in that file's `meta` and `sanity` blocks.

## Release (later; nothing is published automatically)
`scripts/build_release.sh` stages a clean copy in `release/` and runs `scripts/prepublish_scan.sh` (gitleaks with `.gitleaks.toml` + a PII / private-link scan). Public release only after both are clean and you confirm.

## Limitations and ethics
See the appendix section in [docs/PROPOSAL.md](docs/PROPOSAL.md). In short: one small model, two English public benchmarks, a last-token linear probe that correlates with answerability but does not show the model "knows" anything; automatic scoring misses paraphrases; abstention metrics depend on the operating point. SelfAware probe AUROC is confounded by dataset source (see above) and is not a primary result. Only public datasets are used; no personal data.
