#!/usr/bin/env bash
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export WANDB_DIR="${WANDB_DIR:-/workspace/eval_results/wandb}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export VERL_FSDP_CKPT_OFFLOAD_TO_CPU="${VERL_FSDP_CKPT_OFFLOAD_TO_CPU:-1}"
mkdir -p "${WANDB_DIR}"

TRAIN_FILES=${TRAIN_FILES:-/workspace/typst_apps_data/train.parquet}
VAL_FILES=${VAL_FILES:-/workspace/typst_apps_data/validation.parquet}
MODEL_PATH=${MODEL_PATH:-/workspace/typst_universe_scrape/outputs/qwen35-9b-merged}
TOKENIZER_PATH=${TOKENIZER_PATH:-/workspace/hf_cache/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a}
LORA_ADAPTER_PATH=${LORA_ADAPTER_PATH:-}
HARNESS_DIR=${HARNESS_DIR:-/workspace/typst_harness}
LORA_RANK=${LORA_RANK:-64}
LORA_ALPHA=${LORA_ALPHA:-128}
LR=${LR:-3e-6}
WARMUP_STEPS=${WARMUP_STEPS:-5}
WSD_DECAY_RATIO=${WSD_DECAY_RATIO:-0.1}
ACTOR_MODEL_DTYPE=${ACTOR_MODEL_DTYPE:-bf16}
REF_MODEL_DTYPE=${REF_MODEL_DTYPE:-bf16}
ENABLE_GRADIENT_CHECKPOINTING=${ENABLE_GRADIENT_CHECKPOINTING:-False}
ENABLE_ACTIVATION_OFFLOAD=${ENABLE_ACTIVATION_OFFLOAD:-False}
ULYSSES_SEQUENCE_PARALLEL_SIZE=${ULYSSES_SEQUENCE_PARALLEL_SIZE:-2}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-32768}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-8}
ROLLOUT_N=${ROLLOUT_N:-5}
AGENT_NUM_WORKERS=${AGENT_NUM_WORKERS:-4}
REWARD_NUM_WORKERS=${REWARD_NUM_WORKERS:-4}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
PPO_EPOCHS=${PPO_EPOCHS:-4}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-2}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-2}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.5}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-32}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-32768}
ENFORCE_EAGER=${ENFORCE_EAGER:-False}
DISABLE_LOG_STATS=${DISABLE_LOG_STATS:-False}
SAVE_FREQ=${SAVE_FREQ:-1}
MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-2}
ACTOR_CKPT_SAVE_CONTENTS=${ACTOR_CKPT_SAVE_CONTENTS:-"[lora_model,optimizer,extra]"}
ACTOR_CKPT_LOAD_CONTENTS=${ACTOR_CKPT_LOAD_CONTENTS:-"${ACTOR_CKPT_SAVE_CONTENTS}"}
TEST_FREQ=${TEST_FREQ:-1}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
LOG_VAL_GENERATIONS=${LOG_VAL_GENERATIONS:-16}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-/workspace/eval_results/typst_grpo_real/checkpoints}
ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-/workspace/eval_results/typst_grpo_real/rollouts}
VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR:-/workspace/eval_results/typst_grpo_real/validation}
INCREMENTAL_ROLLOUT_DATA_DIR=${INCREMENTAL_ROLLOUT_DATA_DIR:-/workspace/eval_results/typst_grpo_real/incremental_rollouts}

EXTRA_ARGS=()
if [ -n "${TOTAL_TRAINING_STEPS:-}" ]; then
    EXTRA_ARGS+=(trainer.total_training_steps="${TOTAL_TRAINING_STEPS}")
fi
if [ -n "${TRAIN_MAX_SAMPLES:-}" ]; then
    EXTRA_ARGS+=(data.train_max_samples="${TRAIN_MAX_SAMPLES}")
fi
if [ -n "${VAL_MAX_SAMPLES:-}" ]; then
    EXTRA_ARGS+=(data.val_max_samples="${VAL_MAX_SAMPLES}")
fi

LORA_ADAPTER_ARGS=()
if [ -n "${LORA_ADAPTER_PATH}" ]; then
    LORA_ADAPTER_ARGS+=(actor_rollout_ref.model.lora_adapter_path="${LORA_ADAPTER_PATH}")
fi

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=False \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.val_batch_size="${VAL_BATCH_SIZE}" \
    data.dataloader_num_workers=0 \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    +data.apply_chat_template_kwargs.enable_thinking=True \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.tokenizer_path="${TOKENIZER_PATH}" \
    actor_rollout_ref.model.lora_rank="${LORA_RANK}" \
    actor_rollout_ref.model.lora_alpha="${LORA_ALPHA}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing="${ENABLE_GRADIENT_CHECKPOINTING}" \
    actor_rollout_ref.model.enable_activation_offload="${ENABLE_ACTIVATION_OFFLOAD}" \
    actor_rollout_ref.actor.optim.optimizer_impl=recipe.typst_apps.muon_hybrid \
    actor_rollout_ref.actor.optim.optimizer=MuonWithAdamW \
    actor_rollout_ref.actor.optim.lr="${LR}" \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.optim.lr_warmup_steps="${WARMUP_STEPS}" \
    actor_rollout_ref.actor.optim.lr_scheduler_type=wsd \
    +actor_rollout_ref.actor.optim.wsd_decay_ratio="${WSD_DECAY_RATIO}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.actor.ppo_epochs="${PPO_EPOCHS}" \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size="${ULYSSES_SEQUENCE_PARALLEL_SIZE}" \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm \
    actor_rollout_ref.actor.loss_scale_factor="${MAX_RESPONSE_LENGTH}" \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.model_dtype="${ACTOR_MODEL_DTYPE}" \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.temperature=0.7 \
  actor_rollout_ref.rollout.enforce_eager="${ENFORCE_EAGER}" \
  actor_rollout_ref.rollout.disable_log_stats="${DISABLE_LOG_STATS}" \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
  actor_rollout_ref.rollout.max_num_seqs="${MAX_NUM_SEQS}" \
  actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS}" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.rollout.agent.num_workers="${AGENT_NUM_WORKERS}" \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.language_model_only=True \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.mamba_cache_mode=align \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size="${ULYSSES_SEQUENCE_PARALLEL_SIZE}" \
    actor_rollout_ref.ref.fsdp_config.model_dtype="${REF_MODEL_DTYPE}" \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    reward.num_workers="${REWARD_NUM_WORKERS}" \
    reward.custom_reward_function.path=/workspace/verl/recipe/typst_apps/typst_apps_reward.py \
    reward.custom_reward_function.name=compute_score \
    +reward.custom_reward_function.reward_kwargs.harness_dir="${HARNESS_DIR}" \
    +reward.custom_reward_function.reward_kwargs.timeout=10 \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=typst_apps_grpo \
    trainer.experiment_name=qwen35_9b_typst_apps_grpo_lora \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.default_local_dir="${CHECKPOINT_DIR}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.max_actor_ckpt_to_keep="${MAX_ACTOR_CKPT_TO_KEEP}" \
    actor_rollout_ref.actor.checkpoint.save_contents="${ACTOR_CKPT_SAVE_CONTENTS}" \
    actor_rollout_ref.actor.checkpoint.load_contents="${ACTOR_CKPT_LOAD_CONTENTS}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
    trainer.log_val_generations="${LOG_VAL_GENERATIONS}" \
    trainer.rollout_data_dir="${ROLLOUT_DATA_DIR}" \
    trainer.validation_data_dir="${VALIDATION_DATA_DIR}" \
    +trainer.incremental_rollout_data_dir="${INCREMENTAL_ROLLOUT_DATA_DIR}" \
    trainer.total_epochs=1 \
    "${EXTRA_ARGS[@]}" \
    "${LORA_ADAPTER_ARGS[@]}" \
    "$@"
