#!/usr/bin/env python3
"""GPU experiment sweep for Inferno. Writes JSON + a markdown table.

Intended to run on a node that already has the CUDA binary and the compare venv.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results"
INFERNO = Path(os.environ.get("INFERNO_BIN", ROOT / "build-gpu" / "inferno"))
PY = Path(os.environ.get("INFERNO_PY", "/anvil/scratch/x-pmathur1/inferno-cmp-env311/bin/python"))
SCRATCH = Path(os.environ.get("SCRATCH", f"/anvil/scratch/{os.environ.get('USER', 'user')}"))


def run(cmd, timeout=600):
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def compare_workload(model, seqs, prompt, max_new, cuda=True, graphs=True):
    cmd = [
        str(INFERNO),
        "compare-workload",
        "-m",
        str(model),
        "--seqs",
        str(seqs),
        "--prompt",
        str(prompt),
        "--max-new",
        str(max_new),
        "--cuda" if cuda else "--cpu",
    ]
    if cuda and not graphs:
        cmd.append("--no-graphs")
    code, out, err = run(cmd)
    if code != 0:
        raise RuntimeError(f"inferno failed ({code}): {err or out}")
    for line in out.splitlines():
        if line.startswith("COMPARE "):
            return json.loads(line[len("COMPARE ") :])
    raise RuntimeError("no COMPARE line:\n" + out)


def init_model(path, layers, hidden, head_dim, vocab=128, seed=1):
    cmd = [
        str(INFERNO),
        "init-model",
        "-o",
        str(path),
        "--layers",
        str(layers),
        "--hidden",
        str(hidden),
        "--head-dim",
        str(head_dim),
        "--vocab",
        str(vocab),
        "--seed",
        str(seed),
    ]
    code, out, err = run(cmd)
    if code != 0:
        raise RuntimeError(err or out)
    return out.strip()


def pytorch_vllm(model, seqs=8, prompt=12, max_new=12):
    cmd = [
        str(PY),
        str(ROOT / "scripts" / "compare_bench.py"),
        str(model),
        "--inferno",
        str(INFERNO),
        "--seqs",
        str(seqs),
        "--prompt",
        str(prompt),
        "--max-new",
        str(max_new),
    ]
    code, out, err = run(cmd, timeout=900)
    return {
        "ok": code == 0,
        "stdout": out,
        "stderr": err[-2000:] if err else "",
        "code": code,
    }


def parse_compare_stdout(text: str):
    got = {}
    for line in text.splitlines():
        if line.startswith("inferno "):
            # inferno  18.512 ms  96 tok  5185.8 tok/s  ...
            parts = line.split()
            got["inferno_ms"] = float(parts[1])
            got["inferno_tok_s"] = float(parts[5])
        elif line.startswith("pytorch "):
            parts = line.split()
            got["pytorch_ms"] = float(parts[1])
            got["pytorch_tok_s"] = float(parts[5])
        elif line.startswith("vllm ") and "skipped" not in line:
            parts = line.split()
            got["vllm_ms"] = float(parts[1])
            got["vllm_tok_s"] = float(parts[5])
        elif "pytorch matches inferno" in line:
            got["correctness"] = True
        elif line.startswith("MISMATCH"):
            got["correctness"] = False
    return got


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    host = subprocess.check_output(["hostname"], text=True).strip()
    gpu = "unknown"
    try:
        gpu = subprocess.check_output(
            ["nvidia-smi", "-L"], text=True
        ).strip().splitlines()[0]
    except Exception:
        pass

    report = {
        "stamp": stamp,
        "host": host,
        "gpu": gpu,
        "inferno_bin": str(INFERNO),
        "experiments": [],
    }

    models = {
        "tiny_h16": SCRATCH / "inferno-exp-tiny.bin",
        "mid_h64": SCRATCH / "inferno-exp-mid.bin",
        "large_h64": SCRATCH / "inferno-exp-large.bin",
    }
    print("=== init models ===", flush=True)
    print(init_model(models["tiny_h16"], layers=2, hidden=64, head_dim=16), flush=True)
    print(init_model(models["mid_h64"], layers=2, hidden=256, head_dim=64), flush=True)
    print(init_model(models["large_h64"], layers=4, hidden=512, head_dim=64), flush=True)

    # 1) Batch-size scaling on mid model
    print("=== batch scaling ===", flush=True)
    for seqs in [1, 2, 4, 8, 16, 32]:
        row = compare_workload(models["mid_h64"], seqs, 12, 12, cuda=True, graphs=True)
        exp = {
            "name": f"scale_seqs_{seqs}",
            "model": "mid_h64",
            "seqs": seqs,
            "prompt": 12,
            "max_new": 12,
            "cuda": True,
            "graphs": True,
            **{k: row[k] for k in ("ms", "gen", "tok_s", "device", "graph_replays")},
        }
        report["experiments"].append(exp)
        print(json.dumps(exp), flush=True)

    # 2) CUDA graphs on/off
    print("=== graphs ablation ===", flush=True)
    for graphs in [True, False]:
        row = compare_workload(models["mid_h64"], 8, 12, 12, cuda=True, graphs=graphs)
        exp = {
            "name": f"graphs_{'on' if graphs else 'off'}",
            "model": "mid_h64",
            "seqs": 8,
            "prompt": 12,
            "max_new": 12,
            "cuda": True,
            "graphs": graphs,
            **{k: row[k] for k in ("ms", "gen", "tok_s", "device", "graph_replays")},
        }
        report["experiments"].append(exp)
        print(json.dumps(exp), flush=True)

    # 3) Sequence length sweep
    print("=== length sweep ===", flush=True)
    for prompt, max_new in [(8, 8), (12, 12), (32, 32), (64, 64)]:
        row = compare_workload(models["mid_h64"], 8, prompt, max_new, cuda=True, graphs=True)
        exp = {
            "name": f"len_p{prompt}_n{max_new}",
            "model": "mid_h64",
            "seqs": 8,
            "prompt": prompt,
            "max_new": max_new,
            "cuda": True,
            "graphs": True,
            **{k: row[k] for k in ("ms", "gen", "tok_s", "device", "graph_replays")},
        }
        report["experiments"].append(exp)
        print(json.dumps(exp), flush=True)

    # 4) Model size sweep
    print("=== model size ===", flush=True)
    for key in ["tiny_h16", "mid_h64", "large_h64"]:
        row = compare_workload(models[key], 8, 12, 12, cuda=True, graphs=True)
        exp = {
            "name": f"model_{key}",
            "model": key,
            "seqs": 8,
            "prompt": 12,
            "max_new": 12,
            "cuda": True,
            "graphs": True,
            **{k: row[k] for k in ("ms", "gen", "tok_s", "device", "graph_replays")},
        }
        report["experiments"].append(exp)
        print(json.dumps(exp), flush=True)

    # 5) Cross-engine on mid (vLLM needs head_dim 64)
    print("=== cross engine mid_h64 ===", flush=True)
    cross = pytorch_vllm(models["mid_h64"], seqs=8, prompt=12, max_new=12)
    parsed = parse_compare_stdout(cross["stdout"])
    exp = {
        "name": "cross_engine_mid_h64",
        "model": "mid_h64",
        "seqs": 8,
        "prompt": 12,
        "max_new": 12,
        "ok": cross["ok"],
        **parsed,
        "raw_tail": cross["stdout"][-1500:],
    }
    report["experiments"].append(exp)
    print(json.dumps({k: exp[k] for k in exp if k != "raw_tail"}), flush=True)

    # 6) Cross-engine on large
    print("=== cross engine large_h64 ===", flush=True)
    cross = pytorch_vllm(models["large_h64"], seqs=8, prompt=12, max_new=12)
    parsed = parse_compare_stdout(cross["stdout"])
    exp = {
        "name": "cross_engine_large_h64",
        "model": "large_h64",
        "seqs": 8,
        "prompt": 12,
        "max_new": 12,
        "ok": cross["ok"],
        **parsed,
        "raw_tail": cross["stdout"][-1500:],
    }
    report["experiments"].append(exp)
    print(json.dumps({k: exp[k] for k in exp if k != "raw_tail"}), flush=True)

    json_path = OUT_DIR / f"gpu_sweep_{stamp}.json"
    latest = OUT_DIR / "gpu_sweep_latest.json"
    json_path.write_text(json.dumps(report, indent=2))
    latest.write_text(json.dumps(report, indent=2))
    print(f"wrote {json_path}", flush=True)

    # Markdown summary
    md = []
    md.append(f"# Inferno GPU experiments ({stamp})\n")
    md.append(f"- Host: `{host}`")
    md.append(f"- GPU: `{gpu}`")
    md.append(f"- Binary: `{INFERNO}`\n")
    md.append("## Batch scaling (mid_h64, 12+12)\n")
    md.append("| seqs | ms | tok/s | graph_replays |")
    md.append("|---:|---:|---:|---:|")
    for e in report["experiments"]:
        if e["name"].startswith("scale_seqs_"):
            md.append(
                f"| {e['seqs']} | {e['ms']:.2f} | {e['tok_s']:.1f} | {e['graph_replays']} |"
            )
    md.append("\n## CUDA graphs (mid_h64, 8×12+12)\n")
    md.append("| graphs | ms | tok/s | replays |")
    md.append("|---|---:|---:|---:|")
    for e in report["experiments"]:
        if e["name"].startswith("graphs_"):
            md.append(
                f"| {e['graphs']} | {e['ms']:.2f} | {e['tok_s']:.1f} | {e['graph_replays']} |"
            )
    md.append("\n## Sequence length (mid_h64, 8 seqs)\n")
    md.append("| prompt | new | ms | tok/s |")
    md.append("|---:|---:|---:|---:|")
    for e in report["experiments"]:
        if e["name"].startswith("len_"):
            md.append(
                f"| {e['prompt']} | {e['max_new']} | {e['ms']:.2f} | {e['tok_s']:.1f} |"
            )
    md.append("\n## Model size (8×12+12)\n")
    md.append("| model | ms | tok/s |")
    md.append("|---|---:|---:|")
    for e in report["experiments"]:
        if e["name"].startswith("model_"):
            md.append(f"| {e['model']} | {e['ms']:.2f} | {e['tok_s']:.1f} |")
    md.append("\n## Cross-engine\n")
    md.append("| model | Inferno tok/s | PyTorch tok/s | vLLM tok/s | match |")
    md.append("|---|---:|---:|---:|---|")
    for e in report["experiments"]:
        if e["name"].startswith("cross_engine_"):
            md.append(
                f"| {e['model']} | {e.get('inferno_tok_s', float('nan')):.1f} | "
                f"{e.get('pytorch_tok_s', float('nan')):.1f} | "
                f"{e.get('vllm_tok_s', float('nan')):.1f} | {e.get('correctness')} |"
            )
    md.append(
        "\nNotes: PyTorch baseline is serial per-sequence KV decode. "
        "vLLM needs `head_dim` in {64,80,96,112,128,256}. "
        "These are random tiny models in float32, not a production LLM bake-off.\n"
    )
    md_path = ROOT / "RESULTS.md"
    md_path.write_text("\n".join(md) + "\n")
    print(f"wrote {md_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
