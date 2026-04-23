# Typst APPS VERL Run Configs

This file records the two VERL configurations that matter for the next training attempt and for the adapter pushed to Hugging Face.

## Clean One-Step Checkpoint

- Host checkpoint: `/home/user/eval_results/typst_grpo_real_bf16_sp/checkpoints/global_step_1`
- Container checkpoint: `/workspace/eval_results/typst_grpo_real_bf16_sp/checkpoints/global_step_1`
- Log: `/home/user/eval_results/typst_grpo_real_bf16_sp/logs/grpo_real_20260423_071430_fixmem_canary.log`
- W&B run: `https://wandb.ai/chalk22-none/typst_apps_grpo/runs/9tqayhh6`
- Outcome: completed one rollout-update-checkpoint step and exited cleanly.
- Dataset: 1461 train rows, 275 validation rows.
- Total steps: 1 canary step.
- Batch: `data.train_batch_size=16`, rollout `n=5`, effective generated responses per step 80.
- Model start: `/workspace/typst_universe_scrape/outputs/qwen35-9b-merged`.
- Tokenizer: `/workspace/hf_cache/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
- Reasoning: `+data.apply_chat_template_kwargs.enable_thinking=True`.
- LoRA: rank 64, alpha 128.
- Optimizer: `MuonWithAdamW`, LR `3e-6`, weight decay `0.1`.
- Scheduler: WSD requested warmup 5 and decay ratio 0.1, but degenerated for the one-step canary: warmup 0, stable 0, decay 1, final logged LR 0.
- PPO: mini batch 8, micro batch per GPU 1 at the time of this run, epochs 4.
- Sequence parallelism: `actor/ref.ulysses_sequence_parallel_size=2`.
- Actor/ref dtype: `bf16`.
- Gradient checkpointing: disabled.
- Activation offload: disabled.
- vLLM: TP 2, `gpu_memory_utilization=0.5`, `max_model_len=32768`, `max_num_seqs=32`, `max_num_batched_tokens=32768`, CUDA graphs enabled (`enforce_eager=False`).
- Checkpointing: `save_freq=1`; this run saved full model shards because adapter-only checkpoint save was patched after this run.
- Step 1 metrics: score mean 0.025, response length mean 15827.7, max 32225, aborted ratio 0, step time 6435.7 s, generation 920.8 s, actor update 5239.3 s, checkpoint 28.7 s.

## Crashed Resume Attempt

- Log: `/home/user/eval_results/typst_grpo_real_bf16_sp/logs/grpo_real_20260423_092950_mb2.log`
- W&B run: `https://wandb.ai/chalk22-none/typst_apps_grpo/runs/yhhcef8z`
- Outcome: resumed from global step 1, then died before completing step 2.
- Failure mode: Ray killed an `AgentLoopWorker` because host RAM crossed the Ray memory threshold, not a CUDA OOM in backward.
- Ray memory at kill: 75.84 GB / 78.54 GB, threshold 0.95.
- Largest resident processes at kill: vLLM TP workers about 13.5 GB each; actor workers about 2.6 GB and 2.2 GB; each agent loop worker about 0.94-0.99 GB.
- Total steps configured: 91.
- Batch: `data.train_batch_size=16`, rollout `n=5`, effective generated responses per step 80.
- Model start: `/workspace/typst_universe_scrape/outputs/qwen35-9b-merged`.
- Reasoning: `+data.apply_chat_template_kwargs.enable_thinking=True`.
- LoRA: rank 64, alpha 128.
- Optimizer: `MuonWithAdamW`, LR `3e-6`, weight decay `0.1`.
- Scheduler: WSD with warmup 5, stable 76, final decay 10 steps.
- PPO: mini batch 8, micro batch per GPU 2, epochs 4.
- Sequence parallelism: `actor/ref.ulysses_sequence_parallel_size=2`.
- Actor/ref dtype: `bf16`.
- Gradient checkpointing: disabled.
- Activation offload: disabled.
- vLLM: TP 2, `gpu_memory_utilization=0.5`, `max_model_len=32768`, `max_num_seqs=32`, `max_num_batched_tokens=32768`, CUDA graphs enabled (`enforce_eager=False`).
- Checkpointing: adapter-oriented contents requested with `actor_rollout_ref.actor.checkpoint.save_contents=[lora_model,optimizer,extra]` and load contents matching.

## Next Run Implications

- The clean checkpoint is usable and has been exported as an adapter-only HF artifact.
- The immediate bottleneck for the crashed resume is host RAM pressure during rollout worker fanout; lowering `AGENT_NUM_WORKERS` and `REWARD_NUM_WORKERS` is the first lever before changing sequence length.
- The long run should use `AGENT_NUM_WORKERS=2`, `REWARD_NUM_WORKERS=2`, `TEST_FREQ=0`, and `LOG_VAL_GENERATIONS=0` unless there is enough host RAM headroom to turn validation generations back on.
- Keep vLLM VRAM capped at 0.5 unless GPU memory telemetry proves there is headroom.
- Keep checkpointing every step, but use adapter-only save contents to avoid writing full 9B shards.
