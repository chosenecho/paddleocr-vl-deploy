#!/usr/bin/env bash
set -euo pipefail

export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
BASE=/root/autodl-tmp/ocr_deploy
mkdir -p "$BASE"/{envs,cache/pip,cache/conda,cache/paddlex,cache/hf,cache/modelscope,cache/tmp,logs,config,scripts}

# Force 100% domestic China mirrors (never hit repo.anaconda.com or pypi.org or huggingface.co)
export CONDA_PKGS_DIRS="$BASE/cache/conda"
export PIP_CACHE_DIR="$BASE/cache/pip"
export PADDLE_PDX_CACHE_HOME="$BASE/cache/paddlex"
export PADDLE_PDX_MODEL_SOURCE="BOS"
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK="True"
export HF_HOME="$BASE/cache/hf"
export HF_ENDPOINT="https://hf-mirror.com"
export MODELSCOPE_CACHE="$BASE/cache/modelscope"
export TMPDIR="$BASE/cache/tmp"

# Global pip.conf -> Tsinghua mirror (so paddleocr install_genai_server_deps also uses China mirror)
mkdir -p ~/.pip
cat > ~/.pip/pip.conf << 'PIP_EOF'
[global]
index-url = https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
trusted-host = mirrors.tuna.tsinghua.edu.cn
timeout = 120
PIP_EOF

# Clean .condarc -> ONLY Tsinghua mirrors (remove defaults that hangs on repo.anaconda.com)
cat > ~/.condarc << 'CONDARC_EOF'
channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r/
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r/
CONDARC_EOF

# Persist env vars in ~/.bashrc
grep -q "PADDLE_PDX_CACHE_HOME" ~/.bashrc || cat >> ~/.bashrc << 'BASHRC_EOF'

# PaddleOCR-VL-1.6 domestic mirrors & cache paths (keep / clean)
export PADDLE_PDX_CACHE_HOME=/root/autodl-tmp/ocr_deploy/cache/paddlex
export PADDLE_PDX_MODEL_SOURCE=BOS
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
export HF_HOME=/root/autodl-tmp/ocr_deploy/cache/hf
export HF_ENDPOINT=https://hf-mirror.com
export MODELSCOPE_CACHE=/root/autodl-tmp/ocr_deploy/cache/modelscope
export PIP_CACHE_DIR=/root/autodl-tmp/ocr_deploy/cache/pip
BASHRC_EOF

echo "=== [1/4] Creating Python 3.10 conda environments (Tsinghua mirror only) ==="
if [ ! -d "$BASE/envs/fd_server" ]; then
  conda create -y --override-channels -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ -p "$BASE/envs/fd_server" python=3.10
fi
if [ ! -d "$BASE/envs/ocr_client" ]; then
  conda create -y --override-channels -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ -p "$BASE/envs/ocr_client" python=3.10
fi

echo "=== [2/4] Installing PaddlePaddle-GPU + PaddleOCR + FastDeploy in fd_server (China mirrors) ==="
FD_BIN="$BASE/envs/fd_server/bin"
FD_PY="$FD_BIN/python"
FD_OCR="$FD_BIN/paddleocr"
"$FD_PY" -m pip install -U pip
"$FD_PY" -m pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/ --extra-index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
"$FD_PY" -m pip install -U "paddleocr[doc-parser]>=3.6.0"
PATH="$FD_BIN:$PATH" "$FD_OCR" install_genai_server_deps fastdeploy
# FastDeploy 2.3.0 requires paddleformers.trl (removed in paddleformers >= 1.0)
# PaddlePaddle 3.2.1 DistributedStrategy breaks under protobuf >= 6
"$FD_PY" -m pip install --no-deps "paddleformers==0.3.2"
"$FD_PY" -m pip install "protobuf==4.25.8"

echo "=== [3/4] Installing PaddlePaddle-GPU + PaddleOCR-VL-1.6 + API deps in ocr_client ==="
CL_BIN="$BASE/envs/ocr_client/bin"
CL_PY="$CL_BIN/python"
"$CL_PY" -m pip install -U pip
"$CL_PY" -m pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/ --extra-index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
"$CL_PY" -m pip install -U "paddleocr[doc-parser]>=3.6.0" safetensors fastapi uvicorn httpx aiohttp pypdf pypdfium2

echo "=== [4/4] Verifying installations & disk usage ==="
"$FD_PY" -c "import paddle, fastdeploy, paddleocr; print('fd_server OK -> paddle:', paddle.__version__, 'fastdeploy:', fastdeploy.__version__, 'paddleocr:', paddleocr.__version__)"
"$CL_PY" -c "import paddle, paddleocr; print('ocr_client OK -> paddle:', paddle.__version__, 'paddleocr:', paddleocr.__version__, 'cuda:', paddle.device.is_compiled_with_cuda())"
df -h / "$BASE" || true
echo "=== INSTALL_ALL_DONE ==="
