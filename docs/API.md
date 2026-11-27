# API 参考文档

网关由 [`deploy/server_pool.py`](../deploy/server_pool.py) 提供，默认监听 `0.0.0.0:8080`，协议兼容 PaddleX `/layout-parsing` 并在此基础上增加了采样参数透传与质量护栏字段。

---

## 1. `GET /health`

检查 FastDeploy 后端与 GPU Worker 池是否全部初始化完成。

### 成功响应（`HTTP 200`）

```json
{
  "status": "ok",
  "pipeline_version": "v1.6",
  "vlm_backend": "fastdeploy-server",
  "vlm_ok": true,
  "gpu_workers": 6,
  "master_pid": 2172,
  "features": {
    "sampling_params_exposed": [
      "repetitionPenalty",
      "temperature",
      "topP",
      "maxNewTokens",
      "minPixels",
      "maxPixels",
      "vlmExtraArgs"
    ],
    "auto_retry_repetition": true,
    "auto_detect_low_text": true,
    "auto_retry_low_text": true,
    "low_text_threshold": 30,
    "auto_close_code_blocks": true
  }
}
```

若 FastDeploy 不可达或 Worker 池尚未就绪，返回 `HTTP 503`。

---

## 2. `POST /layout-parsing`

对单个或批量 PDF / 图片执行版面解析与文档 OCR，返回每页的 Markdown 文本、可选结构化 JSON 及质量标记。

### 请求参数

#### 输入与输出控制

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `file` | `string` | `null` | 单文件输入：服务器本地路径、`http(s)://` URL 或 Base64（支持 `data:...;base64,` 前缀）。与 `files` 二选一 |
| `files` | `string[]` | `null` | 批量输入数组（同一次请求在单个 GPU Worker 内批处理，吞吐最高） |
| `fileType` | `int` | `null` | `0`=PDF，`1`=图片；省略时根据文件头魔数自动判断 |
| `prettifyMarkdown` | `bool` | `true` | 是否格式化美化 Markdown 输出 |
| `showFormulaNumber` | `bool` | `false` | 是否保留公式编号 |
| `returnMarkdownImages` | `bool` | `false` | 是否在 `markdown.images` 中返回 Base64 编码的插图 |
| `returnPrunedResult` | `bool` | `true` | 是否返回 PaddleX 结构化 `prunedResult`（仅需 Markdown 时设为 `false` 可显著减小响应体积） |
| `logId` | `string` | 自动 UUID | 自定义请求追踪 ID |

#### 版面与识别开关

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `useDocOrientationClassify` | `bool` | 文档方向分类 |
| `useDocUnwarping` | `bool` | 文本图像矫正（去弯曲） |
| `useLayoutDetection` | `bool` | 是否启用版面检测（设为 `false` 时整页直接送入 VLM） |
| `useChartRecognition` | `bool` | 图表解析 |
| `useSealRecognition` | `bool` | 印章识别 |
| `useOcrForImageBlock` | `bool` | 是否对图片块执行 OCR |
| `layoutThreshold` | `float \| object` | 版面检测置信度阈值 |
| `layoutNms` | `bool` | 版面框 NMS 开关 |
| `layoutUnclipRatio` | `float \| float[] \| object` | 版面框外扩比例 |
| `layoutMergeBboxesMode` | `string \| object` | 版面框合并策略 |
| `layoutShapeMode` | `string` | 版面形状模式 |
| `promptLabel` | `string` | 指定 VLM 提示词标签 |
| `formatBlockContent` | `bool` | 是否格式化块内容 |
| `mergeLayoutBlocks` | `bool` | 是否合并版面块 |
| `markdownIgnoreLabels` | `string[]` | 导出 Markdown 时忽略的版面标签列表 |

#### VLM 采样与图像分辨率参数（透传至 FastDeploy）

| 字段 | 类型 | 校验规则 | 说明 |
| --- | --- | --- | --- |
| `repetitionPenalty` | `float` | `> 0` | 重复惩罚系数，如 `1.15`；非法值返回 `HTTP 422` |
| `temperature` | `float` | `>= 0` | 采样温度，如 `0.1`；非法值返回 `HTTP 422` |
| `topP` | `float` | `[0, 1]` | 核采样概率阈值，如 `0.9`；非法值返回 `HTTP 422` |
| `maxNewTokens` | `int` | `> 0` | 单块最大生成 token 数 |
| `minPixels` | `int` | `> 0` | VLM 视觉输入最小像素数 |
| `maxPixels` | `int` | `> 0` | VLM 视觉输入最大像素数 |
| `vlmExtraArgs` | `object` | — | 透传给 VLM 后端的额外参数字典 |

#### 质量护栏覆盖参数（请求级覆盖服务端环境变量）

| 字段 | 类型 | 默认值（服务端） | 说明 |
| --- | --- | --- | --- |
| `autoRetryRepetition` | `bool` | `true`（未显式传 `repetitionPenalty` 时） | 是否开启重复幻觉自动调参重试 |
| `autoDetectLowText` | `bool` | `true` | 是否检测低字数页并打标 `LOW_TEXT_PAGE` |
| `autoRetryLowText` | `bool` | `true` | 非空白低字数页是否自动降级重跑兜底 |
| `lowTextThreshold` | `int` | `30`（`>= 0`） | 低字数页判定阈值（按 `effectiveChars` 计算） |
| `autoCloseCodeBlocks` | `bool` | `true` | 是否自动闭合未闭合的围栏代码块 |

---

### 响应格式

```json
{
  "logId": "a1b2c3d4...",
  "errorCode": 0,
  "errorMsg": "Success",
  "result": {
    "layoutParsingResults": [
      {
        "markdown": {
          "text": "# 文档标题\n...",
          "images": null
        },
        "prunedResult": null,
        "effectiveChars": 2504,
        "qualityFlags": [],
        "retryInfo": null
      }
    ],
    "pageCount": 1,
    "elapsedSec": 2.683,
    "workerPid": 2201
  }
}
```

#### 每页新增字段说明（完全向后兼容 PaddleX 原结构）

- `effectiveChars` (`int`)：剔除 HTML 注释、HTML 标签、Markdown 图片语法及全部空白后的有效正文字数。
- `qualityFlags` (`string[]`)：质量标记集合，可能取值如下：
  - `REPETITION_HALLUCINATION_RETRIED`：检测到重复幻觉，服务端自动调参重跑并成功消除。
  - `REPETITION_HALLUCINATION`：检测到重复幻觉，未开启重试或重试后仍存在残留（此时优先返回较短的非空结果）。
  - `LOW_TEXT_PAGE_RECOVERED`：初始识别字数低于阈值，经兜底重跑后成功恢复至 `>= lowTextThreshold`。
  - `LOW_TEXT_PAGE`：最终有效字数仍低于 `lowTextThreshold`（包括纯空白页或兜底后仍低字数）。
  - `CODE_BLOCK_AUTO_CLOSED`：检测到页尾存在未闭合的 ```` ``` ```` 或 `~~~` 代码块并已自动补齐闭合行。
- `retryInfo` (`object | null`)：触发自动重试时的详细记录（包含触发原因、每次尝试的参数、有效字数与幻觉状态）；未触发重试时为 `null`。

---

## 3. 内置质量护栏机制详解

所有质量检查与重跑均以**单页**为粒度执行（直接提取该页的 `doc_preprocessor_res.output_img`），不会因多页 PDF 中的某一页异常而重跑整个文件。

### 3.1 重复幻觉检测与自动重试

1. **检测规则**（已过滤标准 Markdown 表格分隔线如 `| --- | :---: |`，避免误报）：
   - `char_flood`：单个字母/数字/汉字连续重复 `>= 35` 次
   - `consecutive_line_loop`：长度 `>= 10` 的非空行连续完全相同 `>= 5` 次
   - `line_dominance_loop`：长度 `>= 20` 的行出现 `>= 8` 次且占全页非空字符比例 `>= 35%`
   - `substring_loop` / `tail_substring_loop`：行内或页尾 2000 字符窗口内，存在长度 `3..80` 的非空白子串连续循环 `>= 6` 次
2. **阶梯调参重试策略**：
   - 第 1 次：`repetition_penalty = max(base + 0.15, 1.15)`、`temperature = 0.1`、`top_p = 0.9`
   - 第 2 次：`repetition_penalty = max(base + 0.30, 1.30)`、`temperature = 0.25`、`top_p = 0.9`
3. **兼容调用方自带重试**：
   - 当请求中显式传入 `repetitionPenalty` 且未传 `autoRetryRepetition` 时，服务端默认关闭内部重试，避免“客户端重试 × 服务端重试”放大耗时；如需强制同时开启，显式传 `"autoRetryRepetition": true`。

### 3.2 低字数页 (`LOW_TEXT_PAGE`) 兜底

1. 当 `effectiveChars < lowTextThreshold`（默认 `30`）时，先通过 4 倍下采样灰度分布判断页面是否有视觉内容（非白像素占比 `>= 0.5%` 且标准差 `>= 5.0`）。
2. **纯空白页**：直接标记 `LOW_TEXT_PAGE`，**不浪费重跑算力**。
3. **有视觉内容的页面**：依次执行两步兜底（若第一步已恢复则提前停止）：
   - 第 1 步：`layout_threshold = 0.25`、`use_ocr_for_image_block = true`、`repetition_penalty = max(原值, 1.05)`
   - 第 2 步：`use_layout_detection = false`（整页直出）、`repetition_penalty = max(原值, 1.15)`、`temperature = 0.1`、`top_p = 0.9`
4. 仅在兜底结果**无幻觉**且有效字数恢复至 `>= lowTextThreshold`（或比原结果至少多 10 字）时采纳新结果。

### 3.3 Markdown 代码块自动闭合

逐行跟踪 CommonMark 围栏代码块状态（支持 3 个及以上 `` ` `` 或 `~`），忽略带内联反引号的非围栏行；若页尾仍处于未闭合状态，则在末尾自动补全对应长度与字符的闭合围栏。

---

## 4. 调用示例

### Python：Base64 上传本地 PDF / 图片

```python
import base64
import httpx

def ocr_file(path: str, server_url: str = "http://127.0.0.1:8080") -> dict:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")

    payload = {
        "file": b64,
        "fileType": 0 if path.lower().endswith(".pdf") else 1,
        "returnPrunedResult": False,
        "returnMarkdownImages": False,
    }
    resp = httpx.post(f"{server_url}/layout-parsing", json=payload, timeout=300.0)
    resp.raise_for_status()
    return resp.json()

data = ocr_file("sample.pdf")
for idx, page in enumerate(data["result"]["layoutParsingResults"], 1):
    print(f"=== Page {idx} | chars={page['effectiveChars']} | flags={page['qualityFlags']} ===")
    print(page["markdown"]["text"][:300])
```

### Python：双节点并发轮询调用

当本机通过 SSH 隧道将两台服务器分别映射到 `8080` 和 `8081` 时：

```python
import asyncio
import base64
import itertools
import httpx

ENDPOINTS = ["http://127.0.0.1:8080", "http://127.0.0.1:8081"]

async def process_files(paths: list[str], concurrency: int = 16):
    cycle = itertools.cycle(ENDPOINTS)
    sem = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(timeout=300.0) as client:
        async def one(p: str):
            url = next(cycle)
            b64 = base64.b64encode(open(p, "rb").read()).decode("ascii")
            async with sem:
                r = await client.post(
                    f"{url}/layout-parsing",
                    json={"file": b64, "returnPrunedResult": False},
                )
                r.raise_for_status()
                return p, r.json()

        return await asyncio.gather(*(one(p) for p in paths))
```
