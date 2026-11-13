"""Model host code. torch/transformers are imported lazily so everything else runs (and is tested) without a GPU.

Phi-3.5-mini-instruct: ~3.8B params, 32 layers, d=3072. bf16 weights alone are ~7.6 GB, which does not leave room on an
8 GB card, so the default config loads 8-bit (plan fallback: batch size 1, hooks on the last prompt token only)."""
from __future__ import annotations

import math
from dataclasses import dataclass

from .config import resolve_path


@dataclass
class GenOut:
    text: str
    hidden_last: "object"          # np.ndarray (n_layers+1, d) float32 at the last prompt token (embeddings + each layer)
    mean_logprob: float | None
    max_logprob: float | None
    min_logprob: float | None
    mean_entropy: float | None
    first_token_entropy: float | None
    n_tokens: int


class Phi:
    def __init__(self, cfg: dict):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        m = cfg["model"]
        path = resolve_path(m["path"]) if m.get("path") and resolve_path(m["path"]).exists() else m["name"]
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(str(path))
        kw = {"dtype": torch.bfloat16}      # transformers>=4.56 / 5.x name (torch_dtype is deprecated)
        q = m.get("quantization", "none")
        if q in ("8bit", "4bit"):
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = (BitsAndBytesConfig(load_in_8bit=True) if q == "8bit" else
                                         BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                                            bnb_4bit_quant_type="nf4"))
            kw["device_map"] = {"": 0}
        else:
            kw["device_map"] = m.get("device", "cuda")
        self.model = AutoModelForCausalLM.from_pretrained(str(path), **kw).eval()
        self.cfg = m
        self.max_new = int(m.get("max_new_tokens", 32))
        self.max_prompt = int(m.get("max_prompt_tokens", 1024))

    def _encode(self, messages):
        enc = self.tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        ids = enc["input_ids"] if hasattr(enc, "keys") else enc
        return ids[:, -self.max_prompt:].to(self.model.device)

    def generate(self, messages, want_hidden: bool = True, do_sample: bool = False, temperature: float = 0.7,
                 max_new: int | None = None) -> GenOut:
        torch = self.torch
        ids = self._encode(messages)
        kw = dict(max_new_tokens=max_new or self.max_new, do_sample=do_sample, return_dict_in_generate=True,
                  output_scores=True, output_hidden_states=want_hidden, pad_token_id=self.tok.eos_token_id)
        if do_sample:
            kw["temperature"] = temperature
        with torch.no_grad():
            out = self.model.generate(ids, **kw)
        gen = out.sequences[0, ids.shape[1]:]
        text = self.tok.decode(gen, skip_special_tokens=True).strip()
        lps, ents = [], []
        for step, tok_id in zip(out.scores, gen):
            logp = torch.log_softmax(step[0].float(), -1)
            lps.append(float(logp[tok_id]))
            ents.append(float(-(logp.exp() * logp).sum()))
        hid = None
        if want_hidden:
            first = out.hidden_states[0]                       # tuple over layers of (1, prompt_len, d)
            hid = torch.stack([h[0, -1, :] for h in first]).float().cpu().numpy()
        return GenOut(text, hid, (sum(lps) / len(lps)) if lps else None, max(lps) if lps else None,
                      min(lps) if lps else None, (sum(ents) / len(ents)) if ents else None,
                      ents[0] if ents else None, len(lps))
