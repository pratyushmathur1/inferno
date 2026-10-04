#!/usr/bin/env python3
"""Compare Inferno against PyTorch (and optionally vLLM) on one INF1 checkpoint.

Workload matches ``inferno bench`` / ``inferno compare-workload``:
  8 sequences, 12 prompt tokens, 12 new tokens, greedy.

    python3 scripts/compare_bench.py model.bin --inferno ./build-gpu/inferno

Requires PyTorch with CUDA. vLLM is optional: if importable, the script exports
the INF1 weights as a tiny HuggingFace Llama directory and times vLLM on it.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def load_inferno(path: str):
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"INF1":
            raise SystemExit("not an Inferno model")
        vocab, hidden, layers, heads, kv_heads, head_dim, intermediate = struct.unpack(
            "<7i", f.read(28)
        )
        rms_eps, rope_theta = struct.unpack("<2f", f.read(8))
        eos = struct.unpack("<i", f.read(4))[0]
        (n,) = struct.unpack("<q", f.read(8))
        import array

        raw = array.array("f")
        raw.fromfile(f, n)
    return {
        "vocab": vocab,
        "hidden": hidden,
        "layers": layers,
        "heads": heads,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "intermediate": intermediate,
        "eps": rms_eps,
        "theta": rope_theta,
        "eos": eos,
        "weights": raw,
    }


def split_weights(blob):
    import torch

    weights = torch.tensor(blob["weights"], dtype=torch.float32)
    cursor = 0

    def take(n):
        nonlocal cursor
        sl = weights[cursor : cursor + n]
        cursor += n
        return sl

    H = blob["hidden"]
    emb = take(blob["vocab"] * H).view(blob["vocab"], H)
    layers = []
    for _ in range(blob["layers"]):
        layers.append(
            {
                "rms1": take(H),
                "wq": take(blob["heads"] * blob["head_dim"] * H).view(
                    blob["heads"] * blob["head_dim"], H
                ),
                "wk": take(blob["kv_heads"] * blob["head_dim"] * H).view(
                    blob["kv_heads"] * blob["head_dim"], H
                ),
                "wv": take(blob["kv_heads"] * blob["head_dim"] * H).view(
                    blob["kv_heads"] * blob["head_dim"], H
                ),
                "wo": take(H * blob["heads"] * blob["head_dim"]).view(
                    H, blob["heads"] * blob["head_dim"]
                ),
                "rms2": take(H),
                "wgate": take(blob["intermediate"] * H).view(blob["intermediate"], H),
                "wup": take(blob["intermediate"] * H).view(blob["intermediate"], H),
                "wdown": take(H * blob["intermediate"]).view(H, blob["intermediate"]),
            }
        )
    final = take(H)
    lm = take(blob["vocab"] * H).view(blob["vocab"], H)
    if cursor != weights.numel():
        raise SystemExit("weight cursor did not consume the file")
    return emb, layers, final, lm


def make_prompts(n_seq: int, prompt_len: int, vocab: int):
    prompts = []
    for i in range(n_seq):
        prompts.append([(i * 3 + j) % vocab for j in range(prompt_len)])
    return prompts


class InfernoPt:
    """PyTorch reference of the Inferno Llama block, with a simple KV cache."""

    def __init__(self, blob, device):
        import torch
        import torch.nn.functional as F

        self.torch = torch
        self.F = F
        self.device = device
        self.cfg = blob
        emb, layers, final, lm = split_weights(blob)
        self.emb = emb.to(device)
        self.layers = [
            {k: v.to(device) for k, v in layer.items()} for layer in layers
        ]
        self.final = final.to(device)
        self.lm = lm.to(device)
        self.d = blob["head_dim"]
        self.half = self.d // 2
        self.group = blob["heads"] // blob["kv_heads"]
        self.scale = self.d ** -0.5
        inv = blob["theta"] ** (
            -torch.arange(self.half, dtype=torch.float32, device=device) / (self.d / 2)
        )
        self.inv_freq = inv

    def rms(self, u, w):
        var = u.pow(2).mean(dim=-1, keepdim=True)
        return u * self.torch.rsqrt(var + self.cfg["eps"]) * w

    def rope(self, t, n_heads, positions):
        # rotate-half, matching Inferno
        z = t.view(-1, n_heads, self.d)
        angles = self.torch.outer(positions.float(), self.inv_freq)
        cos = self.torch.cos(angles)[:, None, :]
        sin = self.torch.sin(angles)[:, None, :]
        x0, x1 = z[..., : self.half], z[..., self.half :]
        out0 = x0 * cos - x1 * sin
        out1 = x0 * sin + x1 * cos
        return self.torch.cat([out0, out1], dim=-1)

    def forward_tokens(self, token_ids, start_pos=0, cache=None):
        """token_ids: 1D LongTensor. Returns last-position logits [vocab]."""
        torch = self.torch
        T = token_ids.numel()
        positions = torch.arange(start_pos, start_pos + T, device=self.device)
        h = self.emb[token_ids]
        if cache is None:
            cache = [None] * self.cfg["layers"]
        for li, lw in enumerate(self.layers):
            res = h
            n = self.rms(h, lw["rms1"])
            q = self.rope(n @ lw["wq"].T, self.cfg["heads"], positions)
            k = self.rope(n @ lw["wk"].T, self.cfg["kv_heads"], positions)
            v = (n @ lw["wv"].T).view(-1, self.cfg["kv_heads"], self.d)
            q = q.view(-1, self.cfg["heads"], self.d)
            k = k.view(-1, self.cfg["kv_heads"], self.d)
            if cache[li] is None:
                cache[li] = (k, v)
            else:
                cache[li] = (
                    torch.cat([cache[li][0], k], dim=0),
                    torch.cat([cache[li][1], v], dim=0),
                )
            kk, vv = cache[li]
            kk = kk.repeat_interleave(self.group, dim=1)
            vv = vv.repeat_interleave(self.group, dim=1)
            # q: [Tq, H, D], kk: [Tk, H, D]
            scores = torch.einsum("qhd,khd->hqk", q, kk) * self.scale
            if start_pos == 0 and T == kk.size(0):
                causal = torch.triu(
                    torch.ones(T, T, dtype=torch.bool, device=self.device), diagonal=1
                )
                scores = scores.masked_fill(causal, float("-inf"))
            else:
                # decode: each new query sees all keys
                pass
            attn = torch.softmax(scores, dim=-1)
            mixed = torch.einsum("hqk,khd->qhd", attn, vv).reshape(T, -1)
            h = res + mixed @ lw["wo"].T
            res = h
            n = self.rms(h, lw["rms2"])
            h = res + (
                self.F.silu(n @ lw["wgate"].T) * (n @ lw["wup"].T)
            ) @ lw["wdown"].T
        logits = self.rms(h, self.final)[-1:] @ self.lm.T
        return logits.squeeze(0), cache

    def generate(self, prompt, max_new: int):
        torch = self.torch
        ids = torch.tensor(prompt, dtype=torch.long, device=self.device)
        logits, cache = self.forward_tokens(ids, start_pos=0, cache=None)
        out = list(prompt)
        for step in range(max_new):
            tok = int(torch.argmax(logits).item())
            out.append(tok)
            if step + 1 == max_new:
                break
            nxt = torch.tensor([tok], dtype=torch.long, device=self.device)
            logits, cache = self.forward_tokens(nxt, start_pos=len(out) - 1, cache=cache)
        return out


def sync(device):
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize()


def time_pytorch(blob, prompts, max_new, device, warmup=1):
    model = InfernoPt(blob, device)
    for _ in range(warmup):
        model.generate(prompts[0], max_new)
    sync(device)
    t0 = time.perf_counter()
    outs = [model.generate(p, max_new) for p in prompts]
    sync(device)
    ms = (time.perf_counter() - t0) * 1000.0
    gen = sum(len(o) - len(p) for o, p in zip(outs, prompts))
    return outs, ms, gen


def run_inferno(bin_path, model_path, use_cuda=True):
    cmd = [
        bin_path,
        "compare-workload",
        "-m",
        model_path,
        "--seqs",
        "8",
        "--prompt",
        "12",
        "--max-new",
        "12",
    ]
    if use_cuda:
        cmd.append("--cuda")
    else:
        cmd.append("--cpu")
    out = subprocess.check_output(cmd, text=True)
    # machine line: COMPARE json={...}
    for line in out.splitlines():
        if line.startswith("COMPARE "):
            return json.loads(line[len("COMPARE ") :])
    raise SystemExit("inferno compare-workload missing COMPARE line:\n" + out)


def export_hf(blob, out_dir: Path):
    """Write a tiny LlamaForCausalLM directory vLLM / transformers can load."""
    import torch

    out_dir.mkdir(parents=True, exist_ok=True)
    emb, layers, final, lm = split_weights(blob)
    state = {
        "model.embed_tokens.weight": emb,
        "model.norm.weight": final,
        "lm_head.weight": lm,
    }
    for i, lw in enumerate(layers):
        p = f"model.layers.{i}."
        state[p + "input_layernorm.weight"] = lw["rms1"]
        state[p + "self_attn.q_proj.weight"] = lw["wq"]
        state[p + "self_attn.k_proj.weight"] = lw["wk"]
        state[p + "self_attn.v_proj.weight"] = lw["wv"]
        state[p + "self_attn.o_proj.weight"] = lw["wo"]
        state[p + "post_attention_layernorm.weight"] = lw["rms2"]
        state[p + "mlp.gate_proj.weight"] = lw["wgate"]
        state[p + "mlp.up_proj.weight"] = lw["wup"]
        state[p + "mlp.down_proj.weight"] = lw["wdown"]
    torch.save(state, out_dir / "pytorch_model.bin")
    config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": blob["hidden"],
        "intermediate_size": blob["intermediate"],
        "num_hidden_layers": blob["layers"],
        "num_attention_heads": blob["heads"],
        "num_key_value_heads": blob["kv_heads"],
        "head_dim": blob["head_dim"],
        "vocab_size": blob["vocab"],
        "rms_norm_eps": blob["eps"],
        "rope_theta": blob["theta"],
        "rope_scaling": None,
        "max_position_embeddings": 2048,
        "tie_word_embeddings": False,
        "hidden_act": "silu",
        "torch_dtype": "float32",
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    # Tiny whitespace tokenizer so vLLM/transformers can load the directory.
    # Encode = ord(c) % vocab for printable chars; decode is inverse for ids < 128.
    vocab = blob["vocab"]
    tok_json = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {
                "id": i,
                "content": f"<t{i}>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
            for i in range(vocab)
        ],
        "normalizer": None,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": None,
        "decoder": None,
        "model": {
            "type": "WordLevel",
            "unk_token": "<t0>",
            "vocab": {f"<t{i}>": i for i in range(vocab)},
        },
    }
    (out_dir / "tokenizer.json").write_text(json.dumps(tok_json))
    (out_dir / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "model_max_length": 2048,
                "unk_token": "<t0>",
                "pad_token": "<t0>",
                "bos_token": "<t0>",
                "eos_token": "<t0>",
            }
        )
    )
    (out_dir / "special_tokens_map.json").write_text(
        json.dumps(
            {
                "unk_token": "<t0>",
                "pad_token": "<t0>",
                "bos_token": "<t0>",
                "eos_token": "<t0>",
            }
        )
    )
    return out_dir


def time_vllm(hf_dir: Path, prompts, max_new):
    try:
        from vllm import LLM, SamplingParams
    except Exception as ex:
        return None, f"vllm unavailable: {ex}"

    try:
        # vLLM wants text prompts; feed space-joined token ids as a stand-in and
        # use prompt_token_ids when the installed API supports it.
        llm = LLM(
            model=str(hf_dir),
            dtype="float32",
            gpu_memory_utilization=0.3,
            max_model_len=256,
            enforce_eager=True,
            trust_remote_code=True,
            disable_log_stats=True,
            skip_tokenizer_init=True,
        )
        params = SamplingParams(temperature=0.0, max_tokens=max_new, top_p=1.0)
        t0 = time.perf_counter()
        outs = llm.generate(
            prompt_token_ids=prompts,
            sampling_params=params,
            use_tqdm=False,
        )
        ms = (time.perf_counter() - t0) * 1000.0
        tokens = []
        for o, p in zip(outs, prompts):
            tokens.append(list(p) + list(o.outputs[0].token_ids))
        gen = sum(len(o.outputs[0].token_ids) for o in outs)
        return {"ms": ms, "gen": gen, "tokens": tokens}, None
    except Exception as ex:
        return None, f"vllm generate failed: {ex}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--inferno", default="./build-gpu/inferno")
    parser.add_argument("--seqs", type=int, default=8)
    parser.add_argument("--prompt", type=int, default=12)
    parser.add_argument("--max-new", type=int, default=12)
    parser.add_argument("--skip-vllm", action="store_true")
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        print("PyTorch is required for this comparison.")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blob = load_inferno(args.model)
    prompts = make_prompts(args.seqs, args.prompt, blob["vocab"])

    print(f"device {device}")
    print(
        f"model layers={blob['layers']} hidden={blob['hidden']} "
        f"heads={blob['heads']} kv={blob['kv_heads']} vocab={blob['vocab']}"
    )
    print(f"workload {args.seqs} seqs x ({args.prompt} prompt + {args.max_new} new)")

    # --- Inferno ---
    if not Path(args.inferno).exists():
        print(f"missing inferno binary: {args.inferno}")
        return 1
    inf = run_inferno(args.inferno, args.model, use_cuda=(device.type == "cuda"))
    print(
        f"inferno  {inf['ms']:.3f} ms  {inf['gen']} tok  "
        f"{inf['tok_s']:.1f} tok/s  device={inf.get('device')}  "
        f"graph_replays={inf.get('graph_replays', 0)}"
    )

    # --- PyTorch ---
    pt_outs, pt_ms, pt_gen = time_pytorch(blob, prompts, args.max_new, device)
    pt_tok_s = 1000.0 * pt_gen / pt_ms if pt_ms > 0 else 0.0
    print(f"pytorch  {pt_ms:.3f} ms  {pt_gen} tok  {pt_tok_s:.1f} tok/s  (serial KV decode)")

    # Correctness: first sequence must match Inferno's first sequence tokens.
    inf_tokens = inf["sequences"][0]["tokens"]
    if pt_outs[0] != inf_tokens:
        print("MISMATCH first sequence")
        print("  inferno", inf_tokens)
        print("  pytorch", pt_outs[0])
        return 2
    print("correctness  pytorch matches inferno on sequence 0")

    # --- vLLM (optional) ---
    vllm_line = None
    if not args.skip_vllm:
        with tempfile.TemporaryDirectory(prefix="inferno-hf-") as tmp:
            hf_dir = export_hf(blob, Path(tmp) / "hf")
            vres, verr = time_vllm(hf_dir, prompts, args.max_new)
            if verr:
                print(f"vllm     skipped ({verr})")
            else:
                v_tok_s = 1000.0 * vres["gen"] / vres["ms"] if vres["ms"] > 0 else 0.0
                print(
                    f"vllm     {vres['ms']:.3f} ms  {vres['gen']} tok  {v_tok_s:.1f} tok/s"
                )
                vllm_line = v_tok_s
                if vres["tokens"][0] != inf_tokens:
                    print("note: vllm token stream differs (tokenizer / sampler path)")
                    print("  inferno", inf_tokens)
                    print("  vllm   ", vres["tokens"][0])

    # Summary table
    print("\nsummary")
    print(f"  Inferno continuous batch : {inf['tok_s']:.1f} tok/s")
    print(f"  PyTorch serial KV decode : {pt_tok_s:.1f} tok/s")
    if pt_tok_s:
        print(f"  speedup Inferno/PyTorch  : {inf['tok_s'] / pt_tok_s:.2f}x")
    if vllm_line is not None:
        print(f"  vLLM continuous batch    : {vllm_line:.1f} tok/s")
        if vllm_line > 0:
            print(f"  speedup Inferno/vLLM     : {inf['tok_s'] / vllm_line:.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
