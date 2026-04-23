# Typst APPS GRPO Run Notes

- The main memory culprit in the failed runs was vLLM being too greedy with VRAM, not the reward parser.
- Reducing `actor_rollout_ref.rollout.gpu_memory_utilization` was the key lever.
- The run should stay at a sane RL batch size and use smaller vLLM headroom before shrinking training batch sizes.
- The 32k run still OOMed in actor `loss.backward()` with `ppo_micro_batch_size_per_gpu=1`. The log showed actor FSDP defaulting to `model_dtype=fp32` and Flash Attention warning about fp32 model dtype, so the next run should force actor/ref `fsdp_config.model_dtype=bf16`.
- For 32k reasoning rollouts, use actor/ref Ulysses sequence parallelism (`ulysses_sequence_parallel_size=2`) and activation offload before lowering the response cap. This targets per-sequence backward memory, which is the current failure mode.
- Full runs should use a fixed WSD warmup such as 5 steps, then stable LR, then the final 10% linear decay to zero. One-step canaries will show a degenerate WSD schedule by construction.
- The log and incremental rollout JSON files are written under `/workspace/eval_results/typst_grpo_real/` in the container and `/home/user/eval_results/typst_grpo_real/` on the host.
- The adapter-only checkpoint path is now enabled in the launch script: save/load contents default to `["lora_model", "optimizer", "extra"]`, with `VERL_FSDP_CKPT_OFFLOAD_TO_CPU=1` by default so checkpointing can offload shards to CPU when needed.
- The risky no-gradient-checkpointing experiment is no longer viable at 32k: both microbatch 2 and microbatch 1 OOM during actor update/backward. Long-run defaults now use `enable_gradient_checkpointing=True`, `enable_activation_offload=False`, and update/logprob microbatch 1.
- Detailed configs for the clean one-step checkpoint and the crashed resume attempt are recorded in `recipe/typst_apps/RUN_CONFIGS.md`.
- For the actual long run after the Ray host-RAM crash, default rollout/reward worker fanout is reduced to 2/2 and validation generations are disabled (`TEST_FREQ=0`, `LOG_VAL_GENERATIONS=0`) to preserve host RAM.
- The microbatch-2 long attempt OOMed during actor backward, so the long-run defaults are back to `ppo_micro_batch_size_per_gpu=1` and `log_prob_micro_batch_size_per_gpu=1`.
- The 2026-04-23 long microbatch-1/no-gradient-checkpointing attempt also OOMed during `actor_rollout_update_actor`, inside Qwen3.5 linear attention (`torch_chunk_gated_delta_rule`), with actor using about 77.69 GiB and vLLM asleep at about 1.51 GiB. This confirms the remaining bottleneck is actor activation memory, not rollout KV cache.
- Do not use `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` with this vLLM setup; it conflicts with vLLM's CuMem memory pool.
