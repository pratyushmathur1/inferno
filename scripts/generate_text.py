#!/usr/bin/env python3
"""Text demo: tokenize with HF, run Inferno generate, detokenize.

    python3 scripts/generate_text.py \
        -m models/smollm2-135m.inf1 \
        --hf HuggingFaceTB/SmolLM2-135M-Instruct \
        --prompt "Explain continuous batching in one sentence." \
        --inferno ./build-gpu/inferno --max-new 64
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from transformers import AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True, help="INF1 path")
    ap.add_argument("--hf", required=True, help="HF id/dir for tokenizer")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--inferno", default="./build-gpu/inferno")
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--naive-gemm", action="store_true")
    ap.add_argument("--no-graphs", action="store_true")
    ap.add_argument("--chat", action="store_true", help="wrap prompt in chat template if available")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.hf, use_fast=True)
    text = args.prompt
    if args.chat and hasattr(tok, "apply_chat_template"):
        text = tok.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    ids = tok.encode(text, add_special_tokens=True)
    token_str = ",".join(str(i) for i in ids)
    cmd = [
        args.inferno, "generate", "-m", args.model,
        "--tokens", token_str, "--max-new", str(args.max_new), "--temp", str(args.temp),
    ]
    if args.cpu:
        cmd.append("--cpu")
    else:
        cmd.append("--cuda")
    if args.naive_gemm:
        cmd.append("--naive-gemm")
    if args.no_graphs:
        cmd.append("--no-graphs")

    print(f"prompt_tokens={len(ids)} running: {' '.join(cmd[:6])} …", file=sys.stderr)
    out = subprocess.check_output(cmd, text=True).strip()
    all_ids = [int(x) for x in out.split() if x]
    new_ids = all_ids[len(ids):]
    print(tok.decode(new_ids, skip_special_tokens=True))
    print(f"\n[ids prompt={len(ids)} new={len(new_ids)} total={len(all_ids)}]", file=sys.stderr)


if __name__ == "__main__":
    main()
