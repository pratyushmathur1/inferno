#!/usr/bin/env python3
"""Convert a trained draft (optional) and bench Inferno speculative decoding.

    python3 scripts/bench_speculate.py \
        --target models/smollm2-135m.inf1 \
        --draft-hf runs/draft-smollm2-l6 \
        --draft-out models/draft-smollm2-l6.inf1 \
        --hf HuggingFaceTB/SmolLM2-135M-Instruct \
        --inferno ./build-gpu/inferno \
        --prompts-file scripts/speculate_prompts.txt \
        --out results/speculate_trained_draft.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from transformers import AutoTokenizer

SPEC_RE = re.compile(r"SPECULATE\s*(\{.*?\})\s*\n", re.S)


DEFAULT_PROMPTS = [
    "Explain continuous batching in two sentences.",
    "What is a paged KV cache?",
    "Summarize speculative decoding briefly.",
]


def ensure_inf1(draft_hf: Path | None, draft_out: Path, existing: Path | None) -> Path:
    if existing is not None:
        if not existing.exists():
            raise SystemExit(f"draft INF1 not found: {existing}")
        return existing
    if draft_hf is None:
        raise SystemExit("provide --draft or --draft-hf")
    if not draft_hf.exists():
        raise SystemExit(f"draft HF dir not found: {draft_hf}")
    cmd = [sys.executable, str(Path(__file__).resolve().parent / "hf_to_inf1.py"), str(draft_hf), "-o", str(draft_out)]
    print("convert:", " ".join(cmd))
    subprocess.check_call(cmd)
    return draft_out


def load_prompts(path: Path | None) -> list[str]:
    if path is None:
        return list(DEFAULT_PROMPTS)
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    if not lines:
        raise SystemExit(f"no prompts in {path}")
    return lines


def run_speculate(inferno: Path, target: Path, draft: Path, token_ids: list[int], max_new: int, gamma: int) -> dict:
    token_str = ",".join(str(t) for t in token_ids)
    cmd = [
        str(inferno),
        "speculate",
        "-m",
        str(target),
        "--draft",
        str(draft),
        "--tokens",
        token_str,
        "--max-new",
        str(max_new),
        "--gamma",
        str(gamma),
        "--cuda",
        "--no-graphs",
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    text = proc.stdout
    m = SPEC_RE.search(text)
    if not m:
        raise RuntimeError("speculate output missing SPECULATE JSON block:\n" + text[-1000:])
    stats = json.loads(m.group(1))
    stats["raw_tail"] = text.strip().splitlines()[-1]
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", type=Path, required=True, help="target INF1")
    ap.add_argument("--draft", type=Path, default=None, help="existing draft INF1")
    ap.add_argument("--draft-hf", type=Path, default=None, help="HF checkpoint to convert")
    ap.add_argument("--draft-out", type=Path, default=Path("models/draft.inf1"))
    ap.add_argument("--hf", default="HuggingFaceTB/SmolLM2-135M-Instruct", help="tokenizer id")
    ap.add_argument("--inferno", type=Path, default=Path("./build-gpu/inferno"))
    ap.add_argument("--prompts-file", type=Path, default=None)
    ap.add_argument("--chat", action="store_true", help="apply chat template")
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--gamma", type=int, default=4)
    ap.add_argument("--out", type=Path, default=Path("results/speculate_trained_draft.json"))
    ap.add_argument("--baseline-truncated", type=Path, default=None, help="optional shallow INF1 for comparison")
    args = ap.parse_args()

    if not args.inferno.exists():
        raise SystemExit(f"inferno binary missing: {args.inferno}")
    if not args.target.exists():
        raise SystemExit(f"target INF1 missing: {args.target}")

    draft = ensure_inf1(args.draft_hf, args.draft_out, args.draft)
    tok = AutoTokenizer.from_pretrained(args.hf, use_fast=True)
    prompts = load_prompts(args.prompts_file)

    rows = []
    for prompt in prompts:
        text = prompt
        if args.chat and hasattr(tok, "apply_chat_template"):
            text = tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        ids = tok.encode(text, add_special_tokens=False)
        stats = run_speculate(args.inferno, args.target, draft, ids, args.max_new, args.gamma)
        row = {"prompt": prompt, "n_prompt_tokens": len(ids), **{k: v for k, v in stats.items() if k != "raw_tail"}}
        rows.append(row)
        print(
            f"accept={stats['accept_pct']:.1f}%  speedup={stats['speedup']:.2f}x  "
            f"spec_ms={stats['spec_ms']:.1f}  target_only_ms={stats['target_only_ms']:.1f}  | {prompt[:60]}"
        )

    summary = {
        "draft": str(draft),
        "target": str(args.target),
        "gamma": args.gamma,
        "max_new": args.max_new,
        "n": len(rows),
        "mean_accept_pct": sum(r["accept_pct"] for r in rows) / len(rows),
        "mean_speedup": sum(r["speedup"] for r in rows) / len(rows),
        "mean_spec_ms": sum(r["spec_ms"] for r in rows) / len(rows),
        "mean_target_only_ms": sum(r["target_only_ms"] for r in rows) / len(rows),
        "rows": rows,
    }

    if args.baseline_truncated is not None:
        base_rows = []
        for prompt in prompts:
            text = prompt
            if args.chat and hasattr(tok, "apply_chat_template"):
                text = tok.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            ids = tok.encode(text, add_special_tokens=False)
            stats = run_speculate(args.inferno, args.target, args.baseline_truncated, ids, args.max_new, args.gamma)
            base_rows.append(stats)
        summary["baseline_truncated"] = str(args.baseline_truncated)
        summary["baseline_mean_accept_pct"] = sum(r["accept_pct"] for r in base_rows) / len(base_rows)
        summary["baseline_mean_speedup"] = sum(r["speedup"] for r in base_rows) / len(base_rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {args.out}")
    print(
        f"MEAN accept={summary['mean_accept_pct']:.1f}%  speedup={summary['mean_speedup']:.2f}x"
    )


if __name__ == "__main__":
    main()
