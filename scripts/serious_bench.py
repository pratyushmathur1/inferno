#!/usr/bin/env python3
"""Serious Inferno vs PyTorch (and optional vLLM) bench: TTFT, ITL, throughput, memory.

Uses `inferno bench-latency` so the INF1 weights stay resident across the timed window.

    python3 scripts/serious_bench.py models/smollm2-135m.inf1 \
        --hf HuggingFaceTB/SmolLM2-135M-Instruct \
        --inferno ./build-gpu/inferno \
        --prompt "Explain continuous batching in two sentences." --chat \
        --max-new 32 --out results/serious_bench.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def gpu_mem_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def reset_mem() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def bench_inferno(bin_path: str, model: str, ids: list[int], max_new: int, fuse: bool) -> dict:
    token_str = ",".join(map(str, ids))
    cmd = [bin_path, "bench-latency", "-m", model, "--tokens", token_str, "--max-new", str(max_new), "--cuda"]
    if not fuse:
        cmd.append("--no-fuse")
    out = subprocess.check_output(cmd, text=True)
    line = [ln for ln in out.splitlines() if ln.startswith("LATENCY ")][0]
    data = json.loads(line[len("LATENCY ") :])
    data["engine"] = "inferno"
    return data


def bench_pytorch(hf: str, ids: list[int], max_new: int) -> dict:
    reset_mem()
    model = AutoModelForCausalLM.from_pretrained(hf, torch_dtype=torch.float16).cuda().eval()
    input_ids = torch.tensor([ids], device="cuda")
    with torch.no_grad():
        model.generate(input_ids, max_new_tokens=max_new, do_sample=False)
    torch.cuda.synchronize()
    with torch.no_grad():
        t0 = time.perf_counter()
        out1 = model.generate(input_ids, max_new_tokens=1, do_sample=False)
        torch.cuda.synchronize()
        ttft = (time.perf_counter() - t0) * 1000.0
    with torch.no_grad():
        t0 = time.perf_counter()
        out = model.generate(input_ids, max_new_tokens=max_new, do_sample=False)
        torch.cuda.synchronize()
        wall = (time.perf_counter() - t0) * 1000.0
    new_n = int(out.shape[1] - input_ids.shape[1])
    itl = (wall - ttft) / max(new_n - 1, 1) if new_n > 1 else 0.0
    mem = gpu_mem_mb()
    del model
    return {
        "engine": "pytorch",
        "wall_ms": wall,
        "ttft_ms": ttft,
        "itl_ms": itl,
        "tok_s": (1000.0 * new_n / wall) if wall > 0 else 0.0,
        "new_tokens": new_n,
        "gpu_mem_mb": mem,
    }


def bench_vllm(hf: str, ids: list[int], max_new: int) -> dict:
    try:
        from vllm import LLM, SamplingParams
    except Exception as e:
        return {"engine": "vllm", "error": str(e)}
    reset_mem()
    llm = LLM(model=hf, dtype="float16", gpu_memory_utilization=0.5, max_model_len=2048)
    params = SamplingParams(temperature=0.0, max_tokens=max_new)
    params1 = SamplingParams(temperature=0.0, max_tokens=1)
    prompt_token_ids = [ids]
    llm.generate(prompt_token_ids=prompt_token_ids, sampling_params=params)
    t0 = time.perf_counter()
    outs = llm.generate(prompt_token_ids=prompt_token_ids, sampling_params=params)
    wall = (time.perf_counter() - t0) * 1000.0
    new_n = len(outs[0].outputs[0].token_ids)
    llm.generate(prompt_token_ids=prompt_token_ids, sampling_params=params1)
    t1 = time.perf_counter()
    llm.generate(prompt_token_ids=prompt_token_ids, sampling_params=params1)
    ttft = (time.perf_counter() - t1) * 1000.0
    itl = (wall - ttft) / max(new_n - 1, 1) if new_n > 1 else 0.0
    return {
        "engine": "vllm",
        "wall_ms": wall,
        "ttft_ms": ttft,
        "itl_ms": itl,
        "tok_s": (1000.0 * new_n / wall) if wall > 0 else 0.0,
        "new_tokens": new_n,
        "gpu_mem_mb": gpu_mem_mb(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inf1")
    ap.add_argument("--hf", required=True)
    ap.add_argument("--inferno", default="./build-gpu/inferno")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--skip-vllm", action="store_true")
    ap.add_argument("--out", default="results/serious_bench.json")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.hf)
    text = args.prompt
    if args.chat and hasattr(tok, "apply_chat_template"):
        text = tok.apply_chat_template(
            [{"role": "user", "content": args.prompt}], tokenize=False, add_generation_prompt=True
        )
    ids = tok.encode(text, add_special_tokens=True)

    rows = [
        bench_inferno(args.inferno, args.inf1, ids, args.max_new, fuse=True),
        bench_inferno(args.inferno, args.inf1, ids, args.max_new, fuse=False),
        bench_pytorch(args.hf, ids, args.max_new),
    ]
    if not args.skip_vllm:
        rows.append(bench_vllm(args.hf, ids, args.max_new))

    out = {
        "prompt": args.prompt,
        "prompt_tokens": len(ids),
        "max_new": args.max_new,
        "hf": args.hf,
        "inf1": args.inf1,
        "results": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
