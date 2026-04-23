#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

RUN_STAMP=${RUN_STAMP:-$(date -u +%Y%m%d_%H%M%S)}
LOG_DIR=${LOG_DIR:-/workspace/eval_results/typst_grpo_real/logs}
LOG_FILE=${LOG_FILE:-${LOG_DIR}/grpo_real_${RUN_STAMP}.log}

mkdir -p "${LOG_DIR}"

export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_DIR="${WANDB_DIR:-/workspace/eval_results/wandb}"

echo "Logging VERL GRPO run to ${LOG_FILE}"
echo "W&B mode: ${WANDB_MODE}"
echo "W&B dir: ${WANDB_DIR}"

bash recipe/typst_apps/run_qwen35_9b_typst_apps_grpo_lora.sh "$@" 2>&1 | tee -a "${LOG_FILE}"
exit "${PIPESTATUS[0]}"
