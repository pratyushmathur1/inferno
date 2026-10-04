#!/usr/bin/env python3
"""Build a shallower draft INF1 from a target by keeping the first N layers.

Embeddings + lm_head are copied so vocab matches. Useful for speculative decoding demos.

    python3 scripts/make_draft.py models/smollm2-135m.inf1 -o models/smollm2-draft.inf1 --layers 4
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path


def read_inf1(path: Path):
    with path.open("rb") as f:
        magic = f.read(4)
        if magic != b"INF1":
            raise SystemExit("not INF1")
        vocab, hidden, n_layers, n_heads, n_kv, head_dim, intermediate = struct.unpack("<7i", f.read(28))
        rms_eps, rope_theta = struct.unpack("<ff", f.read(8))
        eos_id = struct.unpack("<i", f.read(4))[0]
        n = struct.unpack("<q", f.read(8))[0]
        raw = f.read(n * 4)
    cfg = dict(vocab=vocab, hidden=hidden, n_layers=n_layers, n_heads=n_heads, n_kv_heads=n_kv,
               head_dim=head_dim, intermediate=intermediate, rms_eps=rms_eps, rope_theta=rope_theta,
               eos_id=eos_id)
    return cfg, memoryview(raw).cast("f")


def layer_stride(cfg) -> int:
    h, q = cfg["hidden"], cfg["n_heads"] * cfg["head_dim"]
    kv, i = cfg["n_kv_heads"] * cfg["head_dim"], cfg["intermediate"]
    return h + q * h + kv * h + kv * h + h * q + h + i * h + i * h + h * i


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("target")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--layers", type=int, required=True)
    args = ap.parse_args()

    cfg, w = read_inf1(Path(args.target))
    if args.layers < 1 or args.layers > cfg["n_layers"]:
        raise SystemExit("bad --layers")
    h, v = cfg["hidden"], cfg["vocab"]
    emb_n = v * h
    ls = layer_stride(cfg)
    # emb + first N layers + final_rms + lm_head from target (final_rms at end of full stack)
    final_rms_off = emb_n + cfg["n_layers"] * ls
    lm_off = final_rms_off + h

    out_cfg = dict(cfg)
    out_cfg["n_layers"] = args.layers
    parts = []
    parts.append(w[0:emb_n])
    parts.append(w[emb_n:emb_n + args.layers * ls])
    parts.append(w[final_rms_off:final_rms_off + h])
    parts.append(w[lm_off:lm_off + v * h])
    flat = bytearray()
    for p in parts:
        flat.extend(memoryview(p).cast("B"))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        f.write(b"INF1")
        f.write(struct.pack("<7i", out_cfg["vocab"], out_cfg["hidden"], out_cfg["n_layers"],
                            out_cfg["n_heads"], out_cfg["n_kv_heads"], out_cfg["head_dim"],
                            out_cfg["intermediate"]))
        f.write(struct.pack("<ff", out_cfg["rms_eps"], out_cfg["rope_theta"]))
        f.write(struct.pack("<i", out_cfg["eos_id"]))
        n = len(flat) // 4
        f.write(struct.pack("<q", n))
        f.write(flat)
    meta = out.with_suffix(out.suffix + ".json")
    meta.write_text(json.dumps({**out_cfg, "draft_of": args.target, "bytes": out.stat().st_size}, indent=2) + "\n")
    print(f"wrote draft {out} layers={args.layers} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
