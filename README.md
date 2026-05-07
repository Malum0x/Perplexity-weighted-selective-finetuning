# Perplexity weighted selective finetuning

Dataset optimization via perplexity filtering.

## Overview

To improve fine-tuning efficiency, I implemented a script that scores
dataset samples based on their cross-entropy loss using a pretrained
model (Qwen2.5-3B-Instruct). By selecting only the top 30% hardest
samples, the goal was to reduce training noise and focus the model's
compute on non-trivial data.

While similar methods exist, this implementation was a custom
experiment to observe firsthand how model behavior and training
dynamics shift when exposed to a specifically curated subset.

All experimental results, including loss curves and performance
metrics, were tracked through W&B.

The filtered dataset (top 30% highest perplexity samples from
OpenHermes 2.5) is publicly available on Hugging Face:
https://huggingface.co/datasets/Malum0x/openhermes2.5-Perplexity_filtered_top30

## Results — original (with chat template)

| Model               | ARC-Challenge | GSM8K |
|---------------------|---------------|-------|
| Base Model          | 82.0%         | 68.0% |
| Baseline Fine-tuned | 80.5%         | 60.0% |
| Filtered Fine-tuned | 82.5%         | 61.0% |

These numbers were produced with chat template applied at generation
time. The eval script in this repo (`eval_all.py`) does not apply a
chat template, so it does not reproduce these numbers as written.

## Results — re-run with consistent eval (2026-05-07)

Same lm-eval settings the follow-up project (mlp-surgery) commits to:
GSM8K flexible-extract 5-shot, ARC Challenge acc_norm 0-shot,
no chat template, no dtype override, batch_size 8. Single seed.

| Model               | GSM8K  | ARC Challenge |
|---------------------|--------|---------------|
| Base model          | 63.15% | 48.12%        |
| Baseline fine-tuned | 65.88% | 45.48%        |
| Filtered fine-tuned | 60.73% | 44.80%        |

Vs base:

| Model               | GSM8K  | ARC    |
|---------------------|--------|--------|
| Baseline fine-tuned | +2.73  | -2.64  |
| Filtered fine-tuned | -2.42  | -3.32  |

## Findings (corrected)

The original narrative — that perplexity filtering partially recovers
ARC — was an artifact of using chat-template at generation time in
the original eval. With the consistent no-chat-template eval that
mlp-surgery uses:

- Standard SFT on full OpenHermes shifts performance unevenly:
  GSM8K goes up, ARC goes down.
- Perplexity-filtered SFT degrades both metrics relative to base.

So in this setup the dataset filtering did not help. The experiment
did establish that fine-tuning on conversational data damages parts
of the base model's capability — and that observation is what set up
the follow-up project, [mlp-surgery](https://github.com/Malum0x/mlp-surgery),
which diagnoses which layers get damaged and shows that simple weight
restoration recovers them (top-30 restoration crosses base on GSM8K).

Keeping the negative result in the open. The chat-template numbers
are not deleted because they were honest at the time; they are
contextualized so future readers understand what they show and what
they don't.

## Caveats

- Both result tables are single-seed.
- The chat-template / no-chat-template difference matters a lot
  for absolute numbers on Qwen-Instruct. Relative deltas within each
  setup are still informative; numbers across setups are not directly
  comparable.

## Related

- Follow-up project: [mlp-surgery](https://github.com/Malum0x/mlp-surgery) — gradient-norm-based MLP layer restoration. Shows top-30 restoration recovers GSM8K above base (+1.14 in re-run, +1.44 original) without retraining.
- Filtered dataset: https://huggingface.co/datasets/Malum0x/openhermes2.5-Perplexity_filtered_top30
