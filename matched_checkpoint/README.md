# Matched checkpoints

Estimates Safety Cost without refusal-direction ablation. Qwen3-8B-Base is fine-tuned twice on
mixtures that differ only in a 20K slice:

| Checkpoint | Shared 60K | 20K slice |
|---|---|---|
| `c0` (unaligned) | multilingual Aya instructions | non-safety Aya instructions |
| `c1` (aligned) | same | English safety data (WildJailbreak + CoCoNot) |

## Data

```bash
python matched_checkpoint/build_instruction_base.py
python matched_checkpoint/build_english_safety.py
python matched_checkpoint/build_c0.py
python matched_checkpoint/build_c1.py
python scripts/00_translate_benchmarks.py --benchmarks falsereject   # evaluation prompts
```

## Training (verl)

```bash
bash matched_checkpoint/train_sft.sh c0
bash matched_checkpoint/train_sft.sh c1
bash matched_checkpoint/merge_checkpoint.sh c0
bash matched_checkpoint/merge_checkpoint.sh c1
```

Full-parameter SFT, 1 epoch, lr 2e-5 (cosine), bf16, effective batch size 512, max length 2,048.

## Evaluation

```bash
python matched_checkpoint/infer.py --model outputs/matched_checkpoint/models/bsft-qwen3-8b-c0 --model_tag qwen3-8b-c0
python matched_checkpoint/infer.py --model outputs/matched_checkpoint/models/bsft-qwen3-8b-c1 --model_tag qwen3-8b-c1
python matched_checkpoint/judge_cost.py   # Safety Cost of c1 vs. c0
python matched_checkpoint/judge_gain.py   # refusal rates on MultiJail
```
