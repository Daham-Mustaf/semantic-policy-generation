#!/usr/bin/env bash
set -euo pipefail

# One-click launcher for full evaluation jobs in background.
# Usage:
#   bash run_all_eval_nohup.sh
#   bash run_all_eval_nohup.sh deepseek-chat

MODEL_ID="${1:-deepseek-chat}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${ROOT_DIR}/evaluation/results"
TS="$(date +%Y%m%d_%H%M%S)"

mkdir -p "${RESULTS_DIR}"

echo "Project root: ${ROOT_DIR}"
echo "Model ID: ${MODEL_ID}"
echo "Timestamp: ${TS}"
echo

REASONING_LOG="${RESULTS_DIR}/nohup_evaluate_reasoning_agent_${MODEL_ID}_full_${TS}.log"
REASONING_PID="${RESULTS_DIR}/nohup_evaluate_reasoning_agent_${MODEL_ID}_full_${TS}.pid"

TEXT2TTL_LOG="${RESULTS_DIR}/nohup_evaluate_text2ttl_pipeline_${MODEL_ID}_full_${TS}.log"
TEXT2TTL_PID="${RESULTS_DIR}/nohup_evaluate_text2ttl_pipeline_${MODEL_ID}_full_${TS}.pid"

cd "${ROOT_DIR}"

nohup uv run python evaluation/evaluate_reasoning_agent.py \
  --model-id "${MODEL_ID}" \
  > "${REASONING_LOG}" 2>&1 &
echo $! > "${REASONING_PID}"

nohup uv run python evaluation/evaluate_text2ttl_pipeline.py \
  --model-id "${MODEL_ID}" \
  --dataset-size -1 \
  > "${TEXT2TTL_LOG}" 2>&1 &
echo $! > "${TEXT2TTL_PID}"

echo "Started full evaluation jobs in background."
echo
echo "[1] Reasoning Agent (full)"
echo "  PID file: ${REASONING_PID}"
echo "  Log file: ${REASONING_LOG}"
echo
echo "[2] Text2TTL Pipeline (full)"
echo "  PID file: ${TEXT2TTL_PID}"
echo "  Log file: ${TEXT2TTL_LOG}"
echo
echo "Quick checks:"
echo "  tail -f \"${REASONING_LOG}\""
echo "  tail -f \"${TEXT2TTL_LOG}\""
