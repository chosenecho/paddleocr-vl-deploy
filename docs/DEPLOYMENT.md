# 部署、调优与运维指南

本文档说明如何从零部署本仓库的 PaddleOCR-VL-1.6 服务、按显存规格调优参数、搭建多机集群以及日常排错。

---

## 1. 环境隔离与依赖版本说明

[`deploy/01_install.sh`](../deploy/01_install.sh) 在 `/root/autodl-tmp/ocr_deploy` 下创建两个独立的 Python 3.10 Conda 环境，所有缓存（conda / pip / paddlex / huggingface / modelscope / tmp）均指向数据盘，避免撑爆系统盘 `/`：

| 环境路径 | 职责 | 关键依赖与版本锁定原因 |
| --- | --- | --- |
| `envs/fd_server` | 运行 FastDeploy VLM 推理服务（端口 `8118`） | `paddlepaddle-gpu==3.2.1` (cu126)、`paddleocr[doc-parser]>=3.6.0`、FastDeploy 依赖；锁定 `paddleformers==0.3.2`（FastDeploy 2.3.0 依赖 `paddleformers.trl`，`>=1.0` 已移除）与 `protobuf==4.25.8`（PaddlePaddle 3.2.1 在 `protobuf>=6` 下异常） |
| `envs/ocr_client` | 运行 FastAPI 网关（端口 `8080`）、`bench_test.py`、`batch_runner.py` | `paddlepaddle-gpu==3.2.1` (cu126)、`paddleocr[doc-parser]>=3.6.0`、`safetensors`、`fastapi`、`uvicorn`、`httpx`、`aiohttp`、`pypdf`、`pypdfium2` |

所有下载均强制走国内镜像：
- Conda / PyPI：清华大学开源软件镜像站 (`mirrors.tuna.tsinghua.edu.cn`)
- PaddlePaddle Wheel：`https://www.paddlepaddle.org.cn/packages/stable/cu126/`
- PaddleX 模型权重：百度 BOS (`PADDLE_PDX_MODEL_SOURCE=BOS`)
- HuggingFace：`https://hf-mirror.com`

---

## 2. 显存配比与性能调优

### 默认配置：RTX 4090 D 48GB（或 A6000 / L40S 48GB）

当前 [`deploy/fastdeploy_config.yml`](../deploy/fastdeploy_config.yml) 与 [`deploy/start_services.sh`](../deploy/start_services.sh) 的默认配比：

- **FastDeploy VLM（BF16，不量化，保留最高精度）**：
  - `gpu-memory-utilization: 0.45`（约占 21.6 GB：2 GB 模型权重 + 约 19.5 GB KV Cache）
  - `max-model-len: 16384`，`max-num-batched-tokens: 16384`，`max-num-seqs: 256`，`max-concurrency: 4096`，`workers: 6`
  - `graph-optimization-config: '{"graph_opt_level":0, "use_cudagraph":true}'`
- **Gateway GPU Workers**：
  - `OCR_WORKERS=6`（6 个常驻 PP-DocLayoutV3 进程）
  - `VL_REC_MAX_CONCURRENCY=64`（每个 Worker 最多同时向 FastDeploy 发起 64 路块识别请求）
  - `OMP_NUM_THREADS=2` / `MKL_NUM_THREADS=2` / `OPENBLAS_NUM_THREADS=2`（防止多进程 CPU 线程争抢）
- **实测显存占用**：空闲启动后约 **25.5 GB / 49.1 GB**，高并发批处理下留有充足显存余量，避免 OOM 与 HTTP 429。

### 适配 24GB 显存 GPU（如 RTX 4090 / 3090 24GB）

修改 `fastdeploy_config.yml` 并调小 `OCR_WORKERS`：

```yaml
max-model-len: 16384
max-num-batched-tokens: 8192
gpu-memory-utilization: 0.45
max-num-seqs: 128
max-concurrency: 1024
workers: 2
graph-optimization-config: '{"graph_opt_level":0, "use_cudagraph":true}'
```

启动时指定 2~3 个 GPU Worker：

```bash
OCR_WORKERS=2 VL_REC_MAX_CONCURRENCY=32 bash /root/autodl-tmp/ocr_deploy/scripts/start_services.sh
```

---

## 3. 环境变量清单

### 启动脚本环境变量（`start_services.sh`）

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `VLM_PORT` | `8118` | FastDeploy 内部监听端口（绑定 `127.0.0.1`） |
| `OCR_API_PORT` | `8080` | FastAPI 网关对外端口（绑定 `0.0.0.0`） |
| `OCR_WORKERS` | `6` | 网关 `ProcessPoolExecutor` GPU Worker 进程数 |
| `VL_REC_MAX_CONCURRENCY` | `64` | 单 Worker 并发调用 FastDeploy 的上限 |

### 网关服务端环境变量（`server_pool.py`）

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `PIPELINE_VERSION` | `v1.6` | PaddleOCR-VL 管线版本 |
| `VLM_SERVER_URL` | `http://127.0.0.1:8118/v1` | FastDeploy OpenAI 兼容端点 |
| `VLM_BACKEND` | `fastdeploy-server` | PaddleOCR VLM 后端类型 |
| `VLM_MODEL_NAME` | `""` | 自定义模型名（留空使用默认） |
| `USE_DOC_ORIENTATION` | `false` | 默认文档方向分类开关 |
| `USE_DOC_UNWARPING` | `false` | 默认文本图像矫正开关 |
| `USE_CHART_RECOGNITION` | `false` | 默认图表识别开关 |
| `USE_SEAL_RECOGNITION` | `false` | 默认印章识别开关 |
| `AUTO_RETRY_REPETITION` | `true` | 是否默认开启重复幻觉自动重试 |
| `AUTO_DETECT_LOW_TEXT` | `true` | 是否默认检测低字数页 |
| `AUTO_RETRY_LOW_TEXT` | `true` | 是否默认对非空白低字数页兜底重跑 |
| `LOW_TEXT_THRESHOLD` | `30` | 低字数页默认有效字符阈值 |
| `AUTO_CLOSE_CODE_BLOCKS` | `true` | 是否默认自动闭合未闭合代码块 |

---

## 4. 多节点横向扩展

每个 GPU 节点完全无状态，只需在新增节点上同步 `deploy/` 并执行安装与启动即可（如果在云平台上通过已有实例镜像/磁盘克隆新节点，可直接同步最新脚本并运行 `start_services.sh`，无需重装环境与重下权重）。

本地通过 SSH 隧道映射多个节点端口后，可在业务客户端做轮询、挂一层 Nginx / HAProxy 负载均衡，或直接使用配套的多节点客户端流水线 **[`chosenecho/paddleocr-vl-scan`](https://github.com/chosenecho/paddleocr-vl-scan)**（原生支持逗号分隔多端点负载均衡、`tar | ssh` 预暂存零 HTTP 上传、节点亲和调度与本地原 PDF 彩色裁图）：

```bash
# 映射两个远程节点到本地 8080 / 8081
ssh -N -L 8080:127.0.0.1:8080 ocr-node-1
ssh -N -L 8081:127.0.0.1:8080 ocr-node-2
```

---

## 5. 大规模离线批处理（`batch_runner.py`）

适合处理数万至十数万页 PDF / 扫描图片：

```bash
/root/autodl-tmp/ocr_deploy/envs/ocr_client/bin/python \
  /root/autodl-tmp/ocr_deploy/scripts/batch_runner.py \
  --input-dir /data/input_pdfs \
  --output-dir /data/ocr_output \
  --server-url http://127.0.0.1:8080 \
  --concurrency 12 \
  --chunk-pages 10 \
  --tmp-dir /dev/shm
```

运行特点：
- 自动将大 PDF 按 `--chunk-pages`（默认 10 页）切分到内存盘 `/dev/shm` 后并行提交，单文件不阻塞并发。
- 进度实时记录在 `<output-dir>/progress.db`（SQLite）；进程中断后直接用相同参数重跑即可无缝断点续传。
- 坏文件/超时分片最多重试 `--max-retries`（默认 3）次后记录到 `failed` 状态，不影响整体任务推进。
- 全部完成后自动按原相对路径合并输出到 `<output-dir>/markdown/`；加 `--save-json` 可同时保存结构化 JSON。

---

## 6. 日常运维与排错

### 查看日志与 GPU 状态

```bash
BASE=/root/autodl-tmp/ocr_deploy

# FastDeploy 推理引擎日志
tail -f "$BASE/logs/fastdeploy.log"

# FastAPI 网关与 Worker 池日志
tail -f "$BASE/logs/gateway.log"

# 实时查看显存与利用率
watch -n 1 nvidia-smi
```

### 常见问题

1. **Windows 上传脚本后报 `bash: ... $'\r': No such file or directory`**
   - Windows CRLF 换行符导致，执行一次转换即可：
     ```bash
     find /root/autodl-tmp/ocr_deploy/scripts /root/autodl-tmp/ocr_deploy/config -maxdepth 1 -type f -exec sed -i 's/\r$//' {} +
     ```
2. **首次启动 `start_services.sh` 等待较久**
   - 首次启动需要从百度 BOS 下载 PaddleOCR-VL-1.6 权重（约 2GB）及 PP-DocLayoutV3 权重，通常 1~3 分钟完成；可 `tail -f logs/fastdeploy.log` 观察进度。后续重启一般在 30~40 秒内全部就绪。
3. **重启后显存未释放**
   - `start_services.sh` 与 `stop_services.sh` 会先温和终止再 `pkill -9` 清理所有 `spawn` 出来的子进程。若仍有残留进程占用显存，可执行 `nvidia-smi` 核对 PID 并清理后重新启动。
4. **如何确认所有特性是否生效**
   - 请求 `curl -s http://127.0.0.1:8080/health`，核对返回 JSON 中 `vlm_ok: true`、`gpu_workers` 数量以及 `features` 字段中的开关状态。
