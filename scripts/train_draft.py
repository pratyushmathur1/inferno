#!/usr/bin/env python3
"""Train a shallow Llama-arch speculative draft model matched to a teacher.

The student keeps the teacher's vocab / width / tokenizer and uses fewer layers.
Weights are initialized from the teacher's embeddings, first N layers, and lm_head,
then distilled with next-token CE + KL on teacher logits.

    # quick CPU smoke (tiny steps)
    python3 scripts/train_draft.py --steps 20 --device cpu --out runs/draft-smoke

    # A100 training
    python3 scripts/train_draft.py \
        --teacher HuggingFaceTB/SmolLM2-135M-Instruct \
        --layers 6 --steps 2000 --batch 8 --lr 2e-4 \
        --out runs/draft-smollm2-l6

Export the HF checkpoint with scripts/hf_to_inf1.py, then:
    ./build-gpu/inferno speculate -m target.inf1 --draft draft.inf1 ...
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_student_from_teacher(teacher: AutoModelForCausalLM, n_layers: int) -> AutoModelForCausalLM:
    tc = teacher.config
    if n_layers < 1 or n_layers > int(tc.num_hidden_layers):
        raise SystemExit(f"--layers must be in [1, {tc.num_hidden_layers}]")

    # Same arch family as the teacher (Llama / SmolLM), fewer layers.
    cfg = copy.deepcopy(tc)
    cfg.num_hidden_layers = n_layers
    student = type(teacher)(cfg)
    tsd = teacher.state_dict()
    ssd = student.state_dict()

    def copy_(dst: str, src: str) -> None:
        if dst not in ssd or src not in tsd:
            raise KeyError(f"cannot copy {src} -> {dst}")
        if ssd[dst].shape != tsd[src].shape:
            raise ValueError(f"shape mismatch {src} {tuple(tsd[src].shape)} vs {dst} {tuple(ssd[dst].shape)}")
        ssd[dst].copy_(tsd[src])

    copy_("model.embed_tokens.weight", "model.embed_tokens.weight")
    copy_("model.norm.weight", "model.norm.weight")
    if "lm_head.weight" in ssd and "lm_head.weight" in tsd:
        copy_("lm_head.weight", "lm_head.weight")

    for i in range(n_layers):
        prefix = f"model.layers.{i}."
        for suffix in (
            "input_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "post_attention_layernorm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ):
            copy_(prefix + suffix, prefix + suffix)

    student.load_state_dict(ssd)
    return student


class TokenPackDataset(Dataset):
    """Packed token blocks for causal LM / distillation."""

    def __init__(self, tokens: list[int], seq_len: int):
        if seq_len < 2:
            raise ValueError("seq_len must be >= 2")
        usable = (len(tokens) // seq_len) * seq_len
        if usable < seq_len:
            raise SystemExit("not enough tokens for one training block; provide more data")
        self.data = torch.tensor(tokens[:usable], dtype=torch.long).view(-1, seq_len)

    def __len__(self) -> int:
        return self.data.size(0)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


def load_tokens_wikitext(tokenizer, max_tokens: int) -> list[int]:
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit("pip install datasets  (needed for --data wikitext)") from e

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [t for t in ds["text"] if t and t.strip()]
    random.shuffle(texts)
    ids: list[int] = []
    for t in texts:
        ids.extend(tokenizer.encode(t, add_special_tokens=False))
        ids.append(tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0)
        if len(ids) >= max_tokens:
            break
    return ids[:max_tokens]


def load_tokens_jsonl(path: Path, tokenizer, max_tokens: int, text_key: str) -> list[int]:
    ids: list[int] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if text_key not in obj:
                raise SystemExit(f"jsonl missing key {text_key!r}: {path}")
            ids.extend(tokenizer.encode(str(obj[text_key]), add_special_tokens=False))
            ids.append(tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0)
            if len(ids) >= max_tokens:
                break
    return ids[:max_tokens]


DEFAULT_PROMPTS = [
    "Explain continuous batching in two sentences.",
    "What is a paged KV cache and why does it help LLM serving?",
    "Write a short checklist for profiling a CUDA kernel.",
    "Summarize speculative decoding for a systems engineer.",
    "Give three reasons greedy decoding is useful for correctness tests.",
    "Describe prefix caching for shared system prompts.",
    "Compare TTFT and inter-token latency in one paragraph.",
    "What does preemption mean in a continuous batching scheduler?",
]


def load_tokens_teacher_rollout(
    teacher,
    tokenizer,
    device: torch.device,
    max_tokens: int,
    gen_len: int,
    prompts: list[str],
) -> list[int]:
    teacher.eval()
    ids: list[int] = []
    i = 0
    while len(ids) < max_tokens:
        prompt = prompts[i % len(prompts)]
        i += 1
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = teacher.generate(
                **enc,
                max_new_tokens=gen_len,
                do_sample=True,
                temperature=0.8,
                top_k=50,
                pad_token_id=tokenizer.eos_token_id,
            )
        ids.extend(out[0].tolist())
        if tokenizer.eos_token_id is not None:
            ids.append(int(tokenizer.eos_token_id))
    return ids[:max_tokens]


def distill_step(
    student,
    teacher,
    batch: torch.Tensor,
    temperature: float,
    alpha_kl: float,
) -> tuple[torch.Tensor, dict]:
    # batch: [B, T]
    inputs = batch[:, :-1]
    labels = batch[:, 1:]
    with torch.no_grad():
        t_logits = teacher(inputs).logits
    s_logits = student(inputs).logits

    ce = F.cross_entropy(
        s_logits.reshape(-1, s_logits.size(-1)),
        labels.reshape(-1),
    )
    t = max(temperature, 1e-5)
    s_log_p = F.log_softmax(s_logits / t, dim=-1)
    t_p = F.softmax(t_logits / t, dim=-1)
    kl = F.kl_div(s_log_p, t_p, reduction="batchmean") * (t * t)
    loss = (1.0 - alpha_kl) * ce + alpha_kl * kl
    return loss, {"ce": float(ce.detach()), "kl": float(kl.detach()), "loss": float(loss.detach())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    ap.add_argument("--layers", type=int, default=6, help="student depth (initialized from teacher)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data", choices=["wikitext", "jsonl", "teacher-rollout"], default="wikitext")
    ap.add_argument("--jsonl", type=Path, default=None, help="required when --data jsonl")
    ap.add_argument("--text-key", default="text")
    ap.add_argument("--max-tokens", type=int, default=200_000)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--temperature", type=float, default=2.0, help="distillation temperature")
    ap.add_argument("--alpha-kl", type=float, default=0.7, help="weight on KL vs CE")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--rollout-gen", type=int, default=64)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    if device.type == "cpu":
        dtype = torch.float32

    print(f"loading teacher {args.teacher} on {device} ({dtype})")
    tokenizer = AutoTokenizer.from_pretrained(args.teacher, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    teacher = AutoModelForCausalLM.from_pretrained(args.teacher, torch_dtype=dtype)
    teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    print(f"building student layers={args.layers} (init from teacher)")
    student = build_student_from_teacher(teacher, args.layers)
    student.to(device=device, dtype=dtype)
    student.train()

    if args.data == "wikitext":
        tokens = load_tokens_wikitext(tokenizer, args.max_tokens)
    elif args.data == "jsonl":
        if args.jsonl is None:
            raise SystemExit("--jsonl path required for --data jsonl")
        tokens = load_tokens_jsonl(args.jsonl, tokenizer, args.max_tokens, args.text_key)
    else:
        tokens = load_tokens_teacher_rollout(
            teacher, tokenizer, device, args.max_tokens, args.rollout_gen, DEFAULT_PROMPTS
        )
    print(f"packed tokens={len(tokens)} seq_len={args.seq_len}")

    ds = TokenPackDataset(tokens, args.seq_len)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=True)
    if len(loader) == 0:
        raise SystemExit("dataloader empty; lower --batch or gather more tokens")

    opt = torch.optim.AdamW((p for p in student.parameters() if p.requires_grad), lr=args.lr, weight_decay=0.01)

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return args.lr * (step + 1) / max(1, args.warmup)
        progress = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    args.out.mkdir(parents=True, exist_ok=True)
    meta = {
        "teacher": args.teacher,
        "layers": args.layers,
        "data": args.data,
        "max_tokens": args.max_tokens,
        "seq_len": args.seq_len,
        "batch": args.batch,
        "steps": args.steps,
        "lr": args.lr,
        "temperature": args.temperature,
        "alpha_kl": args.alpha_kl,
        "seed": args.seed,
    }
    (args.out / "train_config.json").write_text(json.dumps(meta, indent=2) + "\n")

    data_iter = iter(loader)
    t0 = time.time()
    running = {"loss": 0.0, "ce": 0.0, "kl": 0.0, "n": 0}

    for step in range(args.steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        batch = batch.to(device)

        for g in opt.param_groups:
            g["lr"] = lr_at(step)

        # teacher stays eval/frozen; student may be lower precision
        if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
            with torch.autocast(device_type="cuda", dtype=dtype):
                loss, parts = distill_step(student, teacher, batch, args.temperature, args.alpha_kl)
        else:
            loss, parts = distill_step(student, teacher, batch, args.temperature, args.alpha_kl)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()

        running["loss"] += parts["loss"]
        running["ce"] += parts["ce"]
        running["kl"] += parts["kl"]
        running["n"] += 1

        if (step + 1) % args.log_every == 0 or step == 0:
            n = max(1, running["n"])
            print(
                f"step {step + 1}/{args.steps}  loss={running['loss']/n:.4f}  "
                f"ce={running['ce']/n:.4f}  kl={running['kl']/n:.4f}  "
                f"lr={lr_at(step):.2e}  tok/s~{args.batch * (args.seq_len - 1) * n / max(1e-6, time.time() - t0):.0f}"
            )
            running = {"loss": 0.0, "ce": 0.0, "kl": 0.0, "n": 0}
            t0 = time.time()

        if (step + 1) % args.save_every == 0:
            ckpt = args.out / f"checkpoint-{step + 1}"
            student.save_pretrained(ckpt)
            tokenizer.save_pretrained(ckpt)
            print(f"saved {ckpt}")

    student.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    # keep a small readme for the run dir
    (args.out / "README.md").write_text(
        "# Speculative draft checkpoint\n\n"
        f"Teacher: `{args.teacher}`\n\n"
        f"Student layers: `{args.layers}`\n\n"
        "Convert for Inferno:\n\n"
        f"```bash\npython3 scripts/hf_to_inf1.py {args.out} -o models/draft.inf1\n```\n"
    )
    print(f"done. HF model + tokenizer written to {args.out}")
    print(f"next: python3 scripts/hf_to_inf1.py {args.out} -o models/draft.inf1")


if __name__ == "__main__":
    main()
