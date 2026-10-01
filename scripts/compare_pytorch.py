"""Optional PyTorch cross-check for an Inferno ``INF1`` checkpoint.

Runs one Llama-style prefill in PyTorch (RMSNorm, rotate-half RoPE, grouped-query
attention, SwiGLU) and prints the last-token greedy id. Inferno's own test suite
is the source of truth for the serving stack; use this on a machine with a recent
PyTorch when you want a second implementation of the math.

    python3 scripts/compare_pytorch.py model.bin --tokens 1,2,3,4
"""

import argparse
import struct
import sys


def load_inferno(path):
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"INF1":
            raise SystemExit("not an Inferno model")
        vocab, hidden, layers, heads, kv_heads, head_dim, intermediate = struct.unpack("<7i", f.read(28))
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--tokens", default="1,2,3,4")
    args = parser.parse_args()
    try:
        import torch
    except ImportError:
        print("PyTorch is not installed. Inferno's C++ tests do not need it.")
        print("On a GPU node: pip install torch  then re-run this script, and compare")
        print("throughput with `inferno bench` against `vllm bench serve`.")
        return 0

    blob = load_inferno(args.model)
    weights = torch.tensor(blob["weights"], dtype=torch.float32)
    cfg = blob
    cursor = 0

    def take(n):
        nonlocal cursor
        sl = weights[cursor : cursor + n]
        cursor += n
        return sl

    H = cfg["hidden"]
    emb = take(cfg["vocab"] * H).view(cfg["vocab"], H)
    layer_w = []
    for _ in range(cfg["layers"]):
        layer_w.append(
            {
                "rms1": take(H),
                "wq": take(cfg["heads"] * cfg["head_dim"] * H).view(cfg["heads"] * cfg["head_dim"], H),
                "wk": take(cfg["kv_heads"] * cfg["head_dim"] * H).view(cfg["kv_heads"] * cfg["head_dim"], H),
                "wv": take(cfg["kv_heads"] * cfg["head_dim"] * H).view(cfg["kv_heads"] * cfg["head_dim"], H),
                "wo": take(H * cfg["heads"] * cfg["head_dim"]).view(H, cfg["heads"] * cfg["head_dim"]),
                "rms2": take(H),
                "wgate": take(cfg["intermediate"] * H).view(cfg["intermediate"], H),
                "wup": take(cfg["intermediate"] * H).view(cfg["intermediate"], H),
                "wdown": take(H * cfg["intermediate"]).view(H, cfg["intermediate"]),
            }
        )
    final = take(H)
    lm = take(cfg["vocab"] * H).view(cfg["vocab"], H)
    if cursor != weights.numel():
        raise SystemExit("weight cursor did not consume the file")

    ids = torch.tensor([int(x) for x in args.tokens.split(",") if x != ""], dtype=torch.long)
    x = emb[ids]
    d = cfg["head_dim"]
    half = d // 2
    inv_freq = cfg["theta"] ** (-torch.arange(half, dtype=torch.float32) / (d / 2))
    pos = torch.arange(ids.numel(), dtype=torch.float32)
    angles = torch.outer(pos, inv_freq)
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    def rope(t, n_heads):
        # t: [T, heads, d] rotate-half, matching kernels::apply_rope
        z = t.view(-1, n_heads, d)
        x0, x1 = z[..., :half], z[..., half:]
        out0 = x0 * cos[:, None, :] - x1 * sin[:, None, :]
        out1 = x0 * sin[:, None, :] + x1 * cos[:, None, :]
        return torch.cat([out0, out1], dim=-1).reshape(z.shape[0], n_heads * d)

    def rms(u, w):
        var = u.pow(2).mean(dim=-1, keepdim=True)
        return u * torch.rsqrt(var + cfg["eps"]) * w

    h = x
    group = cfg["heads"] // cfg["kv_heads"]
    scale = d ** -0.5
    for lw in layer_w:
        res = h
        n = rms(h, lw["rms1"])
        q = n @ lw["wq"].T
        k = n @ lw["wk"].T
        v = n @ lw["wv"].T
        q = rope(q, cfg["heads"]).view(-1, cfg["heads"], d)
        k = rope(k, cfg["kv_heads"]).view(-1, cfg["kv_heads"], d)
        v = v.view(-1, cfg["kv_heads"], d)
        k = k.repeat_interleave(group, dim=1)
        v = v.repeat_interleave(group, dim=1)
        scores = torch.einsum("thd,shd->hts", q, k) * scale
        causal = torch.triu(torch.ones(ids.numel(), ids.numel(), dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        mixed = torch.einsum("hts,shd->thd", attn, v).reshape(ids.numel(), -1)
        h = res + mixed @ lw["wo"].T
        res = h
        n = rms(h, lw["rms2"])
        h = res + (torch.nn.functional.silu(n @ lw["wgate"].T) * (n @ lw["wup"].T)) @ lw["wdown"].T
    logits = rms(h, final)[-1:] @ lm.T
    print(int(torch.argmax(logits, dim=-1)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
