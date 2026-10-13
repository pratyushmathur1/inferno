# Speculative draft training

Inferno's `speculate` path is correctness-complete, but a **truncated teacher**
(first N layers, no training) does not beat target-only latency. This pipeline
trains a real shallow drafter, exports INF1, and measures accept% / speedup.

Persona / product UI is intentionally out of scope here.

## Pipeline

```
teacher (SmolLM2-135M)
        |
        |  init first N layers + embed + lm_head
        v
student (N layers, same width/vocab)
        |
        |  distill: CE + KL on teacher logits
        v
HF checkpoint  --hf_to_inf1-->  draft.inf1
        |
        v
inferno speculate -m target.inf1 --draft draft.inf1
```

## Local / interactive GPU

```bash
pip install -r requirements.txt

# train (A100-class GPU recommended)
python3 scripts/train_draft.py \
  --teacher HuggingFaceTB/SmolLM2-135M-Instruct \
  --layers 6 --steps 2000 --batch 8 \
  --data wikitext \
  --out runs/draft-smollm2-l6

# ensure target INF1 exists
python3 scripts/hf_to_inf1.py HuggingFaceTB/SmolLM2-135M-Instruct \
  -o models/smollm2-135m.inf1

# convert draft + bench against Inferno
python3 scripts/bench_speculate.py \
  --target models/smollm2-135m.inf1 \
  --draft-hf runs/draft-smollm2-l6 \
  --draft-out models/draft-smollm2-l6.inf1 \
  --hf HuggingFaceTB/SmolLM2-135M-Instruct \
  --inferno ./build-gpu/inferno \
  --prompts-file scripts/speculate_prompts.txt \
  --chat \
  --out results/speculate_trained_draft.json
```

## Anvil Slurm

```bash
mkdir -p logs
sbatch scripts/draft_train_job.sh
# after train finishes:
DRAFT_HF=runs/draft-smollm2-l6 sbatch scripts/draft_eval_job.sh
```

Env knobs: `LAYERS`, `STEPS`, `BATCH`, `DATA`, `OUT`, `TEACHER`, `GAMMA`.

## Data modes

| `--data` | Use when |
|---|---|
| `wikitext` | default general distillation corpus |
| `teacher-rollout` | no `datasets` download; teacher samples its own text |
| `jsonl` | later persona / domain text (`{"text": "..."}` lines) |

## Success criteria

Compare against the truncated baseline (`make_draft.py --layers 15`):

- higher **accept_pct**
- **speedup > 1.0** vs target-only on the prompt set in `scripts/speculate_prompts.txt`

Raw numbers land in `results/speculate_trained_draft.json` for `RESULTS.md`.

## Smoke test (CPU)

```bash
python3 scripts/train_draft.py \
  --steps 5 --batch 1 --seq-len 64 --max-tokens 2048 \
  --data teacher-rollout --device cpu --dtype float32 \
  --out runs/draft-smoke
```
