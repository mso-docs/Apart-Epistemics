# Apart sprint plan: When does a model know it should abstain?

> One-page project plan for the Apart Research AI x Epistemics sprint (Fri Nov 13–Sun Nov 15, 2026; submit by Mon Nov 16, 6:59 AM ET). Track 1: Model Epistemics & Decision-Making Evals. Standalone Apart project — not connected to other research workstreams.

- **Track 1**: Model epistemics & decision-making evals (Same 1–5 rubric across tracks)
- **Mon 6:59 AM ET**: Submission cutoff (Sun Nov 15, 11:59 PM AoE)
- **Phi-3.5-mini**: Primary model (verified) (3.8B, MIT, Hugging Face)
- **≤8 pages**: PDF on official template (150–250 word abstract + required appendix)

> **Recommended title:** The Internal–Expressed Calibration Gap: When Does a Model Know It Should Abstain?

## Title options

1. The Internal–Expressed Calibration Gap: When Does a Model Know It Should Abstain? (recommended)
2. When Does a Model Know It Should Abstain? Internal Evidence vs. Verbalized Confidence
3. Probe-Gated Abstention as an Epistemic Test: Layer-wise Signals, Expressed Confidence, and False Abstention
4. Does the Model Internally Know It Doesn’t Know?

## One-sentence thesis

A small open-weights model may linearly encode, in its hidden states, whether it has sufficient evidence to answer, even when its verbalized confidence and its actual answers are overconfident; measuring that internal-versus-expressed gap, and testing whether a lightweight probe can gate abstention, is the epistemic result — not merely training another probe.

## Research questions

1. RQ1 — Layer-wise detectability.
   - Do linear probes on residual-stream hidden states distinguish answerable or known questions from unanswerable or unknown ones better than verbalized confidence, token log-probability or entropy, and sampling consistency? At which layers is the probe signal strongest?
2. RQ2 — Internal-versus-expressed calibration gap.
   - How often does the probe indicate insufficient evidence while the model still answers confidently — and often incorrectly? Does that gap differ for contextual unanswerability (no evidence in a passage) versus inherent unanswerability (the question itself should not be answered)?
3. RQ3 — Probe-gated abstention.
   - If we threshold the best probe to decide when the model should abstain, do abstention precision and recall, false-abstention rate, and coverage-accuracy improve relative to the same three baselines? Does a probe trained on one unanswerability type transfer to the other?

## Novelty versus prior work

> **Do not claim to invent probing for truth or unanswerability:** Linear probes for truthfulness and unanswerability already exist, and 2026 papers already show a representation-versus-behavior gap on abstention. This weekend is a tight epistemics measurement and a clean intervention test, not a new method family. If results mainly reproduce prior numbers, say so in the report.

Closest verified work, and how this plan stays distinct:

- The Commit-Abstain Circuit (Nguyen et al., 2026, arXiv:2609.32964) finds a sparse commit-abstain circuit and trains a policy on those activations. We will not try to rediscover or ablate that circuit in three days. We use it as motivation for a representation-versus-behavior gap, then measure the gap with standard epistemics metrics (ECE, coverage-accuracy, false abstention) and a layer-wise residual-stream probe.
- Two Axes of LLM Abstention (Wagner, 2026, arXiv:2607.08456) already shows that answerability and answer-correctness are different axes, that hidden-state readouts beat output-confidence for answerability on SelfAware, and that probe-gated prompting can help. Their models include Qwen2.5-3B but not Phi-3.5-mini. Our distinct cut is: a full layer sweep; verbalized (prompted) confidence with ECE; sampling-consistency as a baseline; probe-gated abstention with coverage-accuracy and false-abstention; and a second unanswerability source, SQuAD 2.0 (no evidence in context), versus SelfAware (inherently unanswerable).
- Detecting (Un)answerability in LLMs with Linear Directions (Lavi, Milo, and Geva, EACL 2026) learns a causal unanswerability direction in extractive QA and tests transfer to SelfAware. We use a simpler logistic or mass-mean probe, emphasize calibration and gating tradeoffs rather than causal direction finding, and report the expressed-confidence gap explicitly.
- SAPLMA (Azaria and Mitchell, Findings of EMNLP 2023) and The Geometry of Truth (Marks and Tegmark, COLM 2024), plus Contrast-Consistent Search (Burns, Ye, Klein, and Steinhardt, arXiv:2212.03827), show that hidden states encode truth or falsehood of statements. Those papers are about whether a statement is true, not about when the model should refuse to answer.

Three verified safety or steering papers are adjacent but not central, because they study refusal or safety subspaces rather than epistemic abstention: From Detection to Refusal (EMNLP 2026 Findings), Safety Subspaces are Not Linearly Distinct (ICLR 2026), and What Drives Representation Steering? (EMNLP 2026). Cite them only if needed to contrast linear interventions; they are not the core related work.

## Model (verified on Hugging Face)

- **Primary:** microsoft/Phi-3.5-mini-instruct — 3.8B instruction-tuned model, MIT license, model card at https://huggingface.co/microsoft/Phi-3.5-mini-instruct. Small enough for Colab or a single 16 GB GPU in bf16 or 8-bit. Not in the Two Axes model list, so a positive result is not a direct copy of those numbers. The model card did not describe a access gate; if a gate appears, switch to the fallback.
- **Optional comparison:** Qwen/Qwen2.5-3B-Instruct — 3.09B, 36 layers, verified at https://huggingface.co/Qwen/Qwen2.5-3B-Instruct. Licensed under the Qwen Research License (non-commercial research and evaluation only). Useful only if Friday setup is done early and we want a qualitative check against Two Axes, which already evaluated this model.
- **Fallback if memory is tight:** Keep Phi-3.5-mini and load it in 8-bit, with batch size 1 and hooks only on the last prompt token. Do not add an unverified second architecture unless the primary model cannot be loaded.

## Datasets (verified)

| Dataset | Role | License | Access |
|---|---|---|---|
| SQuAD 2.0 (Rajpurkar et al., 2018) | Contextual unanswerability: question plus passage, some questions have no answer in the passage. Use a balanced subsample of the official splits; evaluate on the validation set. | CC BY-SA 4.0 | https://huggingface.co/datasets/rajpurkar/squad_v2 |
| SelfAware (Yin et al., Findings of ACL 2023) | Inherent unanswerability: 1,032 unanswerable and 2,337 answerable questions. Stratified train/test split. Unanswerables include lack of consensus, subjectivity, and similar categories. | Data: CC BY-SA 4.0 (repo code is Apache 2.0) | https://github.com/yinzhangyue/SelfAware (data/SelfAware.json) |
| TriviaQA (optional only) | Closed-book knowns if more factual answerable items are needed. Not required for the minimum viable study. | The original repo states Apache 2.0 for code and data; Hugging Face lists the license as unknown because the University of Washington does not own copyright in the underlying questions and documents. Treat this caveat as unresolved. | https://nlp.cs.washington.edu/triviaqa/ and https://huggingface.co/datasets/mandarjoshi/trivia_qa |

These two primary sets test different notions of “should abstain”: missing evidence in a provided context versus a question that should not be answered even without a passage. That contrast is part of the claim, not just extra data. Do not scrape additional unlabeled web text this weekend. Attribute CC BY-SA datasets in the report.

## Method

1. **Collect hidden states and behavior.** For each item, run the chat template, record the residual-stream vector at the last prompt token for every layer, then greedily generate an answer. Score correctness with SQuAD exact match or token overlap, and SelfAware answer-string match. Separately elicit verbalized confidence (for example a 0–100 or 1–5 rating). Record mean and max answer-token log-probability and predictive entropy.
2. **Train layer-wise probes.** On the training split, fit a simple linear probe (logistic regression; optional mass-mean probe as a robustness check) to predict answerable versus unanswerable. No heavy classifiers. Report AUROC on a held-out split for each layer, and pick the best layer before looking at gating results on the test split.
3. **Measure the gap.** Find items where the probe says insufficient evidence but verbalized confidence is high and the model commits to an answer. Report how often those answers are wrong. Compare the rate on SQuAD 2.0 versus SelfAware.
4. **Probe-gated abstention.** Sweep a threshold on the probe score. When the score says abstain, replace the answer with an abstention. Compare the resulting coverage-accuracy curve and false-abstention rate with thresholds on verbal confidence, log-probability or entropy, and (on a subset) sampling consistency.
5. **Transfer check, if time.** Train on one unanswerability type and evaluate on the other. A failure to transfer is still a useful bound. Circuit-level analysis is explicitly out of scope unless everything else is done.

## Metrics and baselines

| Quantity | What it answers |
|---|---|
| Probe AUROC (per layer) | How well hidden states separate answerable or known items from unanswerable or unknown ones. |
| ECE of verbalized confidence | Whether stated confidence is calibrated against actual correctness, including confident-but-wrong answers. |
| Abstention precision / recall | When we gate, how often abstentions are justified, and how many cases that should have been abstained are caught. |
| False-abstention rate | How often the gate refuses a question the model would have answered correctly. |
| Coverage-accuracy curves | Accuracy among answered items as a function of the fraction answered; the main policy tradeoff plot. |
| Gap rate | Share of items with high expressed confidence and a committed answer, but a probe score on the unanswerable or unknown side. |

Baselines, all thresholded in the same way as the probe: (1) verbalized confidence; (2) token log-probability and/or entropy of the generated answer; (3) sampling consistency (agreement across 3–5 samples at nonzero temperature, on a subset if compute is tight). The probe has to beat these, or we report that it does not.

## Weekend schedule (America/New_York)

| When (ET) | Focus | Done when |
|---|---|---|
| Fri Nov 13, 9:15–10:00 AM | Logistics talk (Kamil Alaa). RSVP on Luma for the Zoom link: https://luma.com/LogisticsTalk. Then join Discord (https://discord.gg/XswWBvugYs) if not already in it. | Signup confirmed; template downloaded; environment decided (Colab or local GPU). |
| Fri late morning through evening | Load Phi-3.5-mini, hook last-token hidden states, download SQuAD 2.0 and SelfAware, generate answers plus verbal confidence and log-probs on a development slice (a few hundred items). Start layer-wise probes if the pipeline is stable. | A script that, for one item, dumps hidden states, an answer, and a confidence score. One preliminary AUROC, even on a small split. |
| Sat Nov 14 | Full train/test runs on both datasets, layer sweep, ECE, gap cases, and the first probe-versus-baseline gating curves. Optional office hours or a talk (Jonathan Stray 9:15–10:00 AM; Joel Suro 1:15–2:00 PM). Sampling consistency on a subset in the afternoon. | Main tables and the coverage-accuracy plot exist, including at least verbal confidence and log-prob baselines. |
| Sun Nov 15, morning | Transfer check if Friday/Saturday worked; otherwise freeze results and start writing. No new datasets after noon. | Numbers are locked enough to write against. |
| Sun afternoon and evening | Write the PDF on the official template: question, method, results with variance or intervals where possible, limitations. Required appendix: Limitations and Dual-Use / Ethical Considerations. Optional GitHub README that marks all code as written during the sprint. Submit on the sprint page; do not email the PDF. Aim to submit Sunday evening, not at the Monday-morning cutoff. | Confirmation email received with the correct PDF attached. Hard cutoff: Mon Nov 16, 6:59 AM ET. |

Talks are optional. The only hard deadline is the Sunday/Monday submission cutoff. Judging uses three 1–5 scores: Impact Potential & Innovation, Execution Quality, and Presentation & Clarity. Clear method, honest limitations, and readable plots matter as much as a positive result.

## Compute estimate

- Hardware: one Colab T4 or L4 (16 GB), or any single local GPU in that range. Phi-3.5-mini in bf16 is roughly 8 GB of weights; 8-bit leaves more room for hooks.
- Forward passes: on the order of a few thousand items, each with a hidden-state dump, one greedy answer, and one confidence prompt. Expect about 1–3 hours if the pipeline is batched cleanly; budget Friday evening plus Saturday morning.
- Sampling consistency (3–5 samples) is the expensive baseline. Run it on 300–500 held-out items, not the full set, unless there is time left.
- Probe fitting is minutes on CPU. The bottleneck is generation and activation logging, not sklearn.
- No fine-tuning and no multi-model sweep unless the primary run finishes early. Adaption credits from the sprint, if claimed, are optional and not required for this plan.

## Risks and fallbacks

| Risk | Fallback |
|---|---|
| GPU out of memory, or hidden-state hooks blow up memory. | 8-bit load, batch size 1, last-token only, fewer layers logged if needed (even layers, then interpolate the story carefully). |
| The probe learns shallow cues (question form, passage overlap) rather than “knowing that it doesn’t know. | Held-out sources, the SQuAD-versus-SelfAware transfer test, and a short note on residual lexical artifacts. If transfer fails, that is a result, not a reason to hide the number. |
| Verbal confidence is badly elicited (all 100s, or refusal to give a number). | Try one alternative prompt on the dev split; if it stays broken, say so and lean on log-prob and consistency as the expressed-side baselines. |
| SelfAware is small; SQuAD 2.0 is large but stylized. | Fixed, pre-declared splits; subsample SQuAD in a balanced way; do not tune thresholds on the test split. |
| Sampling consistency does not finish. | Drop it to a subset, or drop it entirely and say the comparison is incomplete. |
| Results look like a straight reproduction of Two Axes or Lavi et al. | Lead with what is the same, then with the layer sweep, ECE, coverage-accuracy, and the two unanswerability types. Honest incremental work still scores on execution and clarity. |

## What a null result means

A null result is still a valid Track 1 paper if the tests are clean. If probes do not beat verbal confidence, log-probability, or entropy, that is evidence that this model does not linearly encode a usable “I should not answer” signal beyond what it already expresses. If probes detect unanswerability but gating raises false abstentions enough to erase the coverage-accuracy gain, the signal is diagnostic rather than a drop-in policy. If the gap appears on only one unanswerability type (for example SQuAD 2.0 but not SelfAware), that bounds the claim instead of killing it. Write the null clearly; do not stretch a weak AUROC into a mechanism story.

## Limitations and dual-use appendix sketch

Draft this Sunday, then place it in the required appendix (the page limit does not count this appendix or the references). Suggested points, to be rewritten against the actual results:

- What the result does and does not establish: a linear probe can correlate with answerability or knownness; it does not prove that the model “knows” anything, and this study is not a causal circuit analysis.
- Scope: one small instruction-tuned model, two public English benchmarks, a last-token residual probe, and short answers. Findings may not transfer to larger models, multilingual questions, long-form answers, or tool-using agents.
- Evaluation limits: automatic correctness scoring will miss some valid paraphrases; verbal confidence depends on the elicitation prompt; thresholded abstention metrics depend on the chosen operating point.
- Dual-use: a probe that detects “should abstain” could be used to suppress questions in a chosen domain without disclosing the filter, or ignored in order to present unsupported answers as confident. We will not release a tuned production gate; we will report metrics, prompts, and limitations. Do not build a tool whose main use is hiding uncertainty.
- Data and privacy: only public datasets (SQuAD 2.0 and SelfAware, both CC BY-SA 4.0). No personal or private data is collected.
- If any figure or number is incomplete or failed, say so here rather than in a footnote only.

## Citations: verified and not verified

All of the following were checked against a publisher or arXiv page this turn. Use these, not remembered citations.

| Paper | Status | Link |
|---|---|---|
| Nguyen et al., The Commit-Abstain Circuit: Why Language Models Hallucinate Instead of Abstaining (2026) | Verified (arXiv preprint, 26 Sep 2026) | https://arxiv.org/abs/2609.32964 |
| Wagner, Two Axes of LLM Abstention: Answer Correctness and Question Answerability (2026) | Verified (arXiv) | https://arxiv.org/abs/2607.08456 |
| Lavi, Milo, and Geva, Detecting (Un)answerability in Large Language Models with Linear Directions, EACL 2026 | Verified | https://aclanthology.org/2026.eacl-long.29/ |
| Azaria and Mitchell, The Internal State of an LLM Knows When It’s Lying, Findings of EMNLP 2023 (SAPLMA) | Verified | https://aclanthology.org/2023.findings-emnlp.68/ and https://arxiv.org/abs/2304.13734 |
| Marks and Tegmark, The Geometry of Truth, COLM 2024 | Verified (arXiv and COLM 2024 accepted-papers list) | https://arxiv.org/abs/2310.06824 |
| Burns, Ye, Klein, and Steinhardt, Discovering Latent Knowledge in Language Models Without Supervision (CCS) | Verified on arXiv; venue beyond arXiv:2212.03827 was not independently confirmed here | https://arxiv.org/abs/2212.03827 |
| Yin et al., Do Large Language Models Know What They Don’t Know?, Findings of ACL 2023 (SelfAware) | Verified | https://aclanthology.org/2023.findings-acl.551/ |
| Chu, Sun, and Weng, From Detection to Refusal: Safer LLMs via Circuit-Guided Weight Scaling, Findings of EMNLP 2026 | Verified (authors confirmed from arXiv:2609.00051) | https://aclanthology.org/2026.findings-emnlp.1629/ and https://arxiv.org/abs/2609.00051 |
| Ponkshe, Shah, Singhal, and Vepakomma, Safety Subspaces are Not Linearly Distinct, ICLR 2026 | Verified | https://proceedings.iclr.cc/paper_files/paper/2026/hash/9a9f4e15ad0d680429a3e0570a96f763-Abstract-Conference.html |
| Cheng, Wiegreffe, and Manocha, What Drives Representation Steering? A Mechanistic Case Study on Steering Refusal, EMNLP 2026 | Verified | https://aclanthology.org/2026.emnlp-main.79/ |

> **Citation caveats:** No candidate paper from this planning pass was left unverified and then cited as real. TriviaQA’s license remains partly unresolved, as noted above. For CCS, the arXiv preprint is verified; a later conference venue was not independently confirmed here.

## Sprint deliverable checklist

- [ ] Sign up on the Apart sprint page and join Discord, if not already done (Status as of Thu Nov 12, 2026)
- [ ] Copy the official report template (use the Guidelines-tab link, not an older email copy) (https://docs.google.com/document/d/17b3JWki-ZkaqdmqGI5d0gof-HwlGuWU9Mc1_klNZXKs/copy?usp=sharing)
- [ ] RSVP to Friday Logistics, 9:15 AM ET (https://luma.com/LogisticsTalk)
- [ ] PDF ≤8 pages excluding references and appendix; 150–250 word abstract; authors and affiliations
- [ ] Required Limitations and Dual-Use / Ethical Considerations appendix
- [ ] Optional public GitHub repo with a reproducibility README
- [ ] Submit on the sprint page and confirm the email contains the correct PDF (Before Mon Nov 16, 6:59 AM ET)

---

- [AI x Epistemics Research Sprint](https://apartresearch.com/sprints/ai-x-epistemics-research-sprint-2026-11-13-to-2026-11-15) (apartresearch.com)
- [Official report template](https://docs.google.com/document/d/17b3JWki-ZkaqdmqGI5d0gof-HwlGuWU9Mc1_klNZXKs/copy?usp=sharing) (Google Docs)
- [Phi-3.5-mini-instruct](https://huggingface.co/microsoft/Phi-3.5-mini-instruct) (Hugging Face)
- [SQuAD 2.0](https://huggingface.co/datasets/rajpurkar/squad_v2) (Hugging Face)
- [SelfAware dataset and paper repo](https://github.com/yinzhangyue/SelfAware) (GitHub)

## Sources

- <https://apartresearch.com/sprints/ai-x-epistemics-research-sprint-2026-11-13-to-2026-11-15>
- <https://docs.google.com/document/d/17b3JWki-ZkaqdmqGI5d0gof-HwlGuWU9Mc1_klNZXKs/copy?usp=sharing>
- <https://huggingface.co/microsoft/Phi-3.5-mini-instruct>
- <https://huggingface.co/datasets/rajpurkar/squad_v2>
- <https://github.com/yinzhangyue/SelfAware>
- <https://arxiv.org/abs/2609.32964>
- <https://arxiv.org/abs/2607.08456>
- <https://aclanthology.org/2026.eacl-long.29/>
