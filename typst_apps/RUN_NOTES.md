# Typst APPS GRPO Run Notes

- The main memory culprit in the failed runs was vLLM being too greedy with VRAM, not the reward parser.
- Reducing `actor_rollout_ref.rollout.gpu_memory_utilization` was the key lever.
- The run should stay at a sane RL batch size and use smaller vLLM headroom before shrinking training batch sizes.
- The 32k run still OOMed in actor `loss.backward()` with `ppo_micro_batch_size_per_gpu=1`. The log showed actor FSDP defaulting to `model_dtype=fp32` and Flash Attention warning about fp32 model dtype, so the next run should force actor/ref `fsdp_config.model_dtype=bf16`.
- For 32k reasoning rollouts, use actor/ref Ulysses sequence parallelism (`ulysses_sequence_parallel_size=2`) and activation offload before lowering the response cap. This targets per-sequence backward memory, which is the current failure mode.
- Full runs should use a fixed WSD warmup such as 5 steps, then stable LR, then the final 10% linear decay to zero. One-step canaries will show a degenerate WSD schedule by construction.
- The log and incremental rollout JSON files are written under `/workspace/eval_results/typst_grpo_real/` in the container and `/home/user/eval_results/typst_grpo_real/` on the host.
- The adapter-only checkpoint path is now enabled in the launch script: save/load contents default to `["lora_model", "optimizer", "extra"]`, with `VERL_FSDP_CKPT_OFFLOAD_TO_CPU=1` by default so checkpointing can offload shards to CPU when needed.
- The next run defaults to `enable_gradient_checkpointing=False`, `enable_activation_offload=False`, and `ppo_micro_batch_size_per_gpu=2` / `log_prob_micro_batch_size_per_gpu=2`. That microbatch bump is the risky-but-still-plausible setting the user approved.
- Detailed configs for the clean one-step checkpoint and the crashed resume attempt are recorded in `recipe/typst_apps/RUN_CONFIGS.md`.
- For the actual long run after the Ray host-RAM crash, default rollout/reward worker fanout is reduced to 2/2 and validation generations are disabled (`TEST_FREQ=0`, `LOG_VAL_GENERATIONS=0`) to preserve host RAM.
