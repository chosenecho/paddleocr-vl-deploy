#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/ocr_deploy
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
export PADDLE_PDX_CACHE_HOME="$BASE/cache/paddlex"
export PADDLE_PDX_MODEL_SOURCE="BOS"
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK="True"
export HF_HOME="$BASE/cache/hf"
export HF_ENDPOINT="https://hf-mirror.com"
export MODELSCOPE_CACHE="$BASE/cache/modelscope"
export TMPDIR="$BASE/cache/tmp"
mkdir -p "$BASE/logs" "$TMPDIR"

# Prevent CPU thread oversubscription across 6 workers on 20 vCPUs
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

FD_BIN="$BASE/envs/fd_server/bin"
CL_BIN="$BASE/envs/ocr_client/bin"

VLM_PORT="${VLM_PORT:-8118}"
API_PORT="${OCR_API_PORT:-8080}"
OCR_WORKERS="${OCR_WORKERS:-6}"
VL_REC_MAX_CONCURRENCY="${VL_REC_MAX_CONCURRENCY:-64}"

echo "=== [1/3] Stopping any previous instances (including spawned GPU workers) ==="
pkill -f "uvicorn server_pool:app" 2>/dev/null || true
pkill -f "$BASE/envs/ocr_client/bin/python" 2>/dev/null || true
pkill -f "paddleocr genai_server" 2>/dev/null || true
pkill -f "fastdeploy" 2>/dev/null || true
pkill -f "$BASE/envs/fd_server/bin/python" 2>/dev/null || true
sleep 3
pkill -9 -f "$BASE/envs/ocr_client/bin/python" 2>/dev/null || true
pkill -9 -f "$BASE/envs/fd_server/bin/python" 2>/dev/null || true
sleep 2

echo "=== [2/3] Starting FastDeploy VLM Server (port ${VLM_PORT}, BOS/China mirror) ==="
OMP_NUM_THREADS=1 PATH="$FD_BIN:$PATH" nohup "$FD_BIN/paddleocr" genai_server \
  --model_name PaddleOCR-VL-1.6-0.9B \
  --host 127.0.0.1 \
  --port "${VLM_PORT}" \
  --backend fastdeploy \
  --backend_config "$BASE/config/fastdeploy_config.yml" \
  > "$BASE/logs/fastdeploy.log" 2>&1 &
FD_PID=$!
echo "FastDeploy PID=$FD_PID, waiting for /health (first run downloads ~2GB weights from BOS)..."

for i in $(seq 1 240); do
  if curl -sf "http://127.0.0.1:${VLM_PORT}/health" >/dev/null 2>&1; then
    echo "FastDeploy VLM Server is HEALTHY (${i}x2s)!"
    break
  fi
  if ! kill -0 "$FD_PID" 2>/dev/null; then
    echo "ERROR: FastDeploy exited early! Last 60 lines of log:"
    tail -n 60 "$BASE/logs/fastdeploy.log"
    exit 1
  fi
  sleep 2
done

if ! curl -sf "http://127.0.0.1:${VLM_PORT}/health" >/dev/null 2>&1; then
  echo "ERROR: FastDeploy timed out waiting for /health. Last 60 lines:"
  tail -n 60 "$BASE/logs/fastdeploy.log"
  exit 1
fi

echo "=== [3/3] Starting High-Concurrency PaddleOCR-VL-1.6 Pool Gateway (port ${API_PORT}, gpu_workers=${OCR_WORKERS}) ==="
cd "$BASE/scripts"
export VLM_SERVER_URL="http://127.0.0.1:${VLM_PORT}/v1"
export VLM_BACKEND="fastdeploy-server"
export VL_REC_MAX_CONCURRENCY="${VL_REC_MAX_CONCURRENCY}"
export OCR_WORKERS="${OCR_WORKERS}"
export PIPELINE_VERSION="v1.6"

nohup "$CL_BIN/uvicorn" server_pool:app \
  --host 0.0.0.0 \
  --port "${API_PORT}" \
  --workers 1 \
  > "$BASE/logs/gateway.log" 2>&1 &
GW_PID=$!
echo "Gateway PID=$GW_PID, waiting for /health..."

for i in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:${API_PORT}/health" >/dev/null 2>&1; then
    echo "Gateway is HEALTHY (${i}x2s)!"
    curl -s "http://127.0.0.1:${API_PORT}/health"; echo ""
    nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
    exit 0
  fi
  if ! kill -0 "$GW_PID" 2>/dev/null; then
    echo "ERROR: Gateway exited early! Last 60 lines of log:"
    tail -n 60 "$BASE/logs/gateway.log"
    exit 1
  fi
  sleep 2
done

echo "ERROR: Gateway timed out. Last 60 lines:"
tail -n 60 "$BASE/logs/gateway.log"
exit 1
