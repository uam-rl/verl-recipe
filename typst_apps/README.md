# Typst APPS GRPO

CPU-side glue for running VERL GRPO on APPS-style stdin problems answered in
Typst.

## Inputs

- Harness repo: `/home/user/typst_harness` on the host, expected at
  `/workspace/typst_harness` in the VERL container.
- Dataset: `codeparrot/apps`.
- Example solution files: `sol_apps_*.typ` in the harness repo. Their numeric
  problem IDs are excluded from training by default.

## Data Prep

Run inside the VERL container after mounting the harness:

```bash
python3 /workspace/verl/recipe/typst_apps/prepare_apps_data.py \
  --local_save_dir /workspace/typst_apps_data \
  --harness_dir /workspace/typst_harness \
  --max_train_samples 0 \
  --max_val_samples 256
```

Use `--max_train_samples 0` for the full filtered train set. For smoke tests,
set it to a small positive value.

## Reward

Use this custom reward function in the GRPO command:

```bash
reward.custom_reward_function.path=/workspace/verl/recipe/typst_apps/typst_apps_reward.py \
reward.custom_reward_function.name=compute_score \
+reward.custom_reward_function.reward_kwargs.harness_dir=/workspace/typst_harness \
+reward.custom_reward_function.reward_kwargs.timeout=10
```

The score is the pass ratio: `passed_tests / total_tests`. Harness or compile
errors return `0.0` with diagnostic fields in the reward dict.

## RL Defaults

The GRPO launch script is configured for DrGRPO-style training:

- base model: merged SFT checkpoint at
  `/workspace/typst_universe_scrape/outputs/qwen35-9b-merged`
- fresh RL LoRA: rank 64, alpha 128
- optimizer: official `torch.optim.Muon` for 2D parameters, with AdamW fallback
  for any non-2D trainables
- schedule: warmup-stable-decay, with fixed `WARMUP_STEPS=10` and linear decay
  over the final `WSD_DECAY_RATIO=0.1` of post-warmup steps
- DrGRPO knobs: `loss_agg_mode=seq-mean-token-sum-norm`,
  `loss_scale_factor=max_response_length`, no KL loss, and no GRPO std
  normalization

Set `LORA_ADAPTER_PATH` only if intentionally continuing an adapter with a
matching rank. The default starts from the warm merged SFT model and trains a new
rank-64 RL adapter.

## Rollout Report

Render incremental rollout JSONL files to a PDF for inspection with:

```bash
python3 /workspace/verl/recipe/typst_apps/render_rollouts_report.py \
  /workspace/eval_results/typst_grpo_real/incremental_rollouts \
  --step 1 \
  --output /workspace/eval_results/typst_grpo_real/reports/step_1.pdf
```

Use `--all-steps` to render every step, `--include-prompt` to add prompt tails,
and `--keep-typst` if you want the intermediate `.typ` source for debugging.
Each sample shows the score, stop reason, token count, cap hits, reward metadata,
and the raw response body in a preformatted Typst block.
