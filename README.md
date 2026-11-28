# PaddleOCR-VL-1.6 High-Concurrency Deployment

面向单卡大显存 GPU（默认按 **RTX 4090 D 48GB** 调优）的 **PaddleOCR-VL-1.6** 高并发生产级部署方案：基于 **FastDeploy VLM 推理后端 + 多进程 GPU Worker 池 FastAPI 网关**，提供兼容 PaddleX `/layout-parsing` 的 HTTP 接口、采样参数透传、质量自动兜底护栏与断点续传批处理脚本。

## 为什么做这套封装

直接使用单进程 `PaddleOCRVL` 或默认 `uvicorn --workers N` 部署时，常见三个瓶颈：

1. **吞吐打不满**：版面分析（PP-DocLayoutV3）单进程串行成为瓶颈，FastDeploy 连续批处理吃不满；改为 **1 个异步 Master + `ProcessPoolExecutor` N 个独立 GPU Worker** 后，请求按空闲 Worker 真实调度，持续给后端 FastDeploy（`max-num-seqs=256` + CUDA Graph）喂满并发。
2. **线上长文档质量异常难拦截**：VLM 偶尔出现重复输出死循环、漏检低字数页、或 Markdown 代码块未闭合。网关内置了单页粒度的检测与兜底重跑，不拖累同批次其他页面。
3. **国内环境依赖冲突多**：FastDeploy 2.3.0 与高版本 `paddleformers`、`protobuf` 存在兼容冲突，且默认会访问海外源。安装脚本采用双 Conda 环境隔离 + 100% 国内镜像源（清华源 / 百度 BOS / hf-mirror）。

## 架构概览

```text
Client / batch_runner.py / paddleocr-vl-scan
          │  HTTP POST /layout-parsing  (port 8080)
          ▼
┌──────────────────────────────────────────────────────────┐
│  FastAPI Master (server_pool.py, uvicorn --workers 1)    │
│  ├── /health (检查 Worker 池与 FastDeploy 健康状态)      │
│  └── ProcessPoolExecutor (spawn, 默认 OCR_WORKERS=6)     │
│        ├── GPU Worker 1 ─┐                               │
│        ├── GPU Worker 2 ─┼─► PaddleOCRVL(v1.6)           │
│        └── GPU Worker N ─┘   (版面分析 + 质量护栏/重试)   │
└──────────────────────────┬───────────────────────────────┘
                           │ OpenAI-compatible HTTP (127.0.0.1:8118/v1)
                           ▼
┌──────────────────────────────────────────────────────────┐
│  FastDeploy VLM Server (PaddleOCR-VL-1.6-0.9B, BF16)     │
│  gpu-memory-utilization=0.45 | max-num-seqs=256 | CUDAGraph
└──────────────────────────────────────────────────────────┘
```

## 核心特性

- **完整参数透传**：`/layout-parsing` 暴露 PaddleX 版面/格式参数及 VLM 采样参数（`repetitionPenalty`、`temperature`、`topP`、`maxNewTokens`、`minPixels`、`maxPixels`、`vlmExtraArgs`）。
- **重复幻觉自动调参重试**：自动识别单字刷屏、连续重复行、长行占比异常、连续子串死循环（跳过正常 Markdown 表格分隔线），仅对异常单页阶梯提升 `repetition_penalty` / `temperature` 重跑；若调用方已显式传入 `repetitionPenalty`，默认尊重调用方参数不重复重试。
- **低字数页 (`LOW_TEXT_PAGE`) 检测与兜底**：统计去除图片/HTML 标签后的有效字数 `effectiveChars`；非空白页低于阈值（默认 `30`）时，自动依次尝试放宽版面检测阈值与整页识别兜底，空白页直接标记跳过以节省算力。
- **代码块自动闭合**：自动检测页尾未闭合的 CommonMark 围栏代码块（```` ``` ```` / `~~~`）并补全闭合标记。
- **可断点续传的大规模批处理**：[`deploy/batch_runner.py`](deploy/batch_runner.py) 支持 PDF 自动分片、SQLite (`progress.db`) 进度落盘、失败重试、死信隔离与最终 Markdown 合并。

## 目录结构

```text
.
├── deploy/
│   ├── 01_install.sh          # 双 Conda 环境隔离安装（国内镜像 + 版本锁定）
│   ├── fastdeploy_config.yml  # FastDeploy 推理配置（默认按 48GB GPU 调优）
│   ├── server_pool.py         # FastAPI 多进程 GPU 池网关 + 质量护栏
│   ├── start_services.sh      # 一键拉起 FastDeploy(8118) + Gateway(8080)
│   ├── stop_services.sh       # 彻底停止网关、Worker 与 FastDeploy
│   ├── bench_test.py          # 单页并发与批处理满载吞吐压测脚本
│   └── batch_runner.py        # 基于 SQLite 断点续传的大规模 PDF/图片批处理客户端
└── docs/
    ├── API.md                 # 接口参数、返回字段、质量标记与调用示例
    └── DEPLOYMENT.md          # 安装部署、显存调优（24GB/48GB）、多节点与排错指南
```

## 快速开始

### 前置要求

- Linux + NVIDIA GPU（CUDA 12.6+ 驱动；默认配置针对 **48GB 显存**，24GB 显存调优见 [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md)）
- Miniconda / Conda（脚本默认位于 `/root/miniconda3/bin`）
- 磁盘空间 `>= 25 GB`（默认安装与缓存目录为 `/root/autodl-tmp/ocr_deploy`，不占用系统盘 `/`）

### 1. 上传并安装环境

将仓库上传到目标服务器后，放置配置文件与脚本并运行安装：

```bash
BASE=/root/autodl-tmp/ocr_deploy
mkdir -p "$BASE"/{config,scripts}
cp deploy/fastdeploy_config.yml "$BASE/config/"
cp deploy/01_install.sh deploy/server_pool.py deploy/start_services.sh \
   deploy/stop_services.sh deploy/batch_runner.py deploy/bench_test.py "$BASE/scripts/"
chmod +x "$BASE/scripts/"*

# 创建 fd_server / ocr_client 两个独立 Conda 环境并安装依赖
bash "$BASE/scripts/01_install.sh"
```

### 2. 启动与停止服务

```bash
# 启动 FastDeploy (127.0.0.1:8118) + OCR 网关 (0.0.0.0:8080)
# 首次启动会自动从百度 BOS 下载约 2GB 权重
bash /root/autodl-tmp/ocr_deploy/scripts/start_services.sh

# 停止全部服务
bash /root/autodl-tmp/ocr_deploy/scripts/stop_services.sh
```

> 可通过环境变量覆盖默认启动参数，例如：`OCR_WORKERS=4 OCR_API_PORT=8080 VLM_PORT=8118 bash start_services.sh`。

### 3. 健康检查与调用示例

```bash
# 健康检查（返回后端状态、Worker 数及已开启特性）
curl -s http://127.0.0.1:8080/health

# 识别单张图片或 PDF（支持本地路径、HTTP(S) URL、Base64）
curl -s -X POST http://127.0.0.1:8080/layout-parsing \
  -H "Content-Type: application/json" \
  -d '{
    "file": "/path/to/demo.png",
    "fileType": 1,
    "repetitionPenalty": 1.15,
    "temperature": 0.1,
    "topP": 0.9,
    "returnPrunedResult": false
  }'
```

通过 SSH 端口转发从本机安全访问单机或多机节点：

```bash
# 节点 1 -> 本地 8080
ssh -N -L 8080:127.0.0.1:8080 <your-ocr-node-1>

# 节点 2 -> 本地 8081（多机并行分流）
ssh -N -L 8081:127.0.0.1:8080 <your-ocr-node-2>
```

### 4. 吞吐压测与批量处理

```bash
PY=/root/autodl-tmp/ocr_deploy/envs/ocr_client/bin/python

# 单页并发 + 批处理满载压测
"$PY" /root/autodl-tmp/ocr_deploy/scripts/bench_test.py --server-url http://127.0.0.1:8080

# 大规模 PDF/图片断点续传批处理（支持中断后直接重跑续传）
"$PY" /root/autodl-tmp/ocr_deploy/scripts/batch_runner.py \
  --input-dir /path/to/pdfs \
  --output-dir /path/to/output \
  --server-url http://127.0.0.1:8080 \
  --concurrency 12 \
  --chunk-pages 10
```

## 详细文档

- **接口与质量护栏说明**：[`docs/API.md`](docs/API.md) — 完整请求/响应字段、采样参数校验、重复幻觉重试策略、`LOW_TEXT_PAGE` 判定与 Python 客户端示例
- **部署、显存调优与排错**：[`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) — 依赖锁定原因、48GB/24GB 显存配比、环境变量清单、多机部署与常见问题

## 配套客户端

- **[`chosenecho/paddleocr-vl-scan`](https://github.com/chosenecho/paddleocr-vl-scan)**：本地多节点批量 PDF 扫描客户端。单台 GPU 本机跑文件直接用 `deploy/batch_runner.py` 即可；需要本地经 SSH 隧道调度多台节点、扫前逐页预检（`book-quality`）、`tar | ssh` 预暂存、本地原 PDF 彩色裁图或 SQLite 质量追踪时，用 `paddleocr-vl-scan`。

## License

[MIT](LICENSE)
