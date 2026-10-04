#!/usr/bin/env python3
"""Convert a HuggingFace Llama / SmolLM checkpoint to Inferno INF1 (float32).

Supports LlamaForCausalLM-style weights (including tied embeddings).

    python3 scripts/hf_to_inf1.py HuggingFaceTB/SmolLM2-135M-Instruct -o models/smollm2-135m.inf1
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM


def write_inf1(path: Path, cfg: dict, weights: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(b"INF1")
        f.write(struct.pack("<7i", cfg["vocab"], cfg["hidden"], cfg["n_layers"], cfg["n_heads"],
                            cfg["n_kv_heads"], cfg["head_dim"], cfg["intermediate"]))
        f.write(struct.pack("<ff", cfg["rms_eps"], cfg["rope_theta"]))
        f.write(struct.pack("<i", cfg["eos_id"]))
        n = weights.numel()
        f.write(struct.pack("<q", n))
        f.write(weights.numpy().tobytes())


def pack_llama(model) -> tuple[dict, torch.Tensor]:
    c = model.config
    hidden = int(c.hidden_size)
    n_heads = int(c.num_attention_heads)
    n_kv = int(getattr(c, "num_key_value_heads", n_heads))
    head_dim = hidden // n_heads
    if hasattr(c, "head_dim") and c.head_dim:
        head_dim = int(c.head_dim)
    intermediate = int(c.intermediate_size)
    n_layers = int(c.num_hidden_layers)
    vocab = int(c.vocab_size)
    rms_eps = float(getattr(c, "rms_norm_eps", 1e-5))
    rope_theta = float(getattr(c, "rope_theta", 10000.0))
    eos_id = int(getattr(c, "eos_token_id", -1) or -1)

    sd = model.state_dict()

    def t(name: str) -> torch.Tensor:
        if name not in sd:
            raise KeyError(f"missing weight {name}")
        return sd[name].detach().float().contiguous().reshape(-1)

    chunks: list[torch.Tensor] = []
    # Inferno layout: row-major W for y = x @ W.T, so store [out, in] flattened.
    emb = sd["model.embed_tokens.weight"].detach().float().contiguous()
    chunks.append(emb.reshape(-1))

    for layer in range(n_layers):
        p = f"model.layers.{layer}"
        chunks.append(t(f"{p}.input_layernorm.weight"))
        # HF stores [out, in]; Inferno linear is the same layout.
        chunks.append(t(f"{p}.self_attn.q_proj.weight"))
        chunks.append(t(f"{p}.self_attn.k_proj.weight"))
        chunks.append(t(f"{p}.self_attn.v_proj.weight"))
        chunks.append(t(f"{p}.self_attn.o_proj.weight"))
        chunks.append(t(f"{p}.post_attention_layernorm.weight"))
        chunks.append(t(f"{p}.mlp.gate_proj.weight"))
        chunks.append(t(f"{p}.mlp.up_proj.weight"))
        chunks.append(t(f"{p}.mlp.down_proj.weight"))

    chunks.append(t("model.norm.weight"))
    if "lm_head.weight" in sd:
        chunks.append(t("lm_head.weight"))
    else:
        # Tied embeddings.
        chunks.append(emb.reshape(-1))

    weights = torch.cat(chunks)
    cfg = {
        "vocab": vocab,
        "hidden": hidden,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "n_kv_heads": n_kv,
        "head_dim": head_dim,
        "intermediate": intermediate,
        "rms_eps": rms_eps,
        "rope_theta": rope_theta,
        "eos_id": eos_id,
    }
    return cfg, weights


def expected_count(cfg: dict) -> int:
    v, h, L = cfg["vocab"], cfg["hidden"], cfg["n_layers"]
    q = cfg["n_heads"] * cfg["head_dim"]
    kv = cfg["n_kv_heads"] * cfg["head_dim"]
    i = cfg["intermediate"]
    n = v * h
    for _ in range(L):
        n += h  # rms1
        n += q * h + kv * h + kv * h + h * q
        n += h  # rms2
        n += i * h + i * h + h * i
    n += h  # final rms
    n += v * h  # lm_head
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="HF hub id or local directory")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--dtype-load", default="float16", choices=["float16", "bfloat16", "float32"])
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype_load]
    print(f"loading {args.model} …")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.eval()
    cfg, weights = pack_llama(model)
    exp = expected_count(cfg)
    if weights.numel() != exp:
        raise SystemExit(f"weight count {weights.numel()} != expected {exp} for {cfg}")
    out = Path(args.out)
    write_inf1(out, cfg, weights)
    meta = out.with_suffix(out.suffix + ".json")
    meta.write_text(json.dumps({**cfg, "source": args.model, "bytes": out.stat().st_size}, indent=2) + "\n")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB) cfg={cfg}")
    print(f"meta {meta}")


if __name__ == "__main__":
    main()
