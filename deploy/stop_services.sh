#!/usr/bin/env bash
set -euo pipefail
BASE=/root/autodl-tmp/ocr_deploy
pkill -f "uvicorn server_pool:app" 2>/dev/null || true
pkill -f "$BASE/envs/ocr_client/bin/python" 2>/dev/null || true
pkill -f "paddleocr genai_server" 2>/dev/null || true
pkill -f "fastdeploy" 2>/dev/null || true
pkill -f "$BASE/envs/fd_server/bin/python" 2>/dev/null || true
sleep 3
pkill -9 -f "$BASE/envs/ocr_client/bin/python" 2>/dev/null || true
pkill -9 -f "$BASE/envs/fd_server/bin/python" 2>/dev/null || true
sleep 1
echo "Stopped OCR Gateway, all GPU workers, and FastDeploy server."
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
