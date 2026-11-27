#!/usr/bin/env python3
"""
True Least-Busy Multi-Process GPU Pool FastAPI Gateway for PaddleOCR-VL-1.6 + FastDeploy.

Architecture:
  1 async FastAPI Master Process (uvicorn --workers 1)
    └── ProcessPoolExecutor(max_workers=OCR_WORKERS, spawn)
          ├── GPU Worker 1: PaddleOCRVL(v1.6, use_queues=True, vl_rec_max_concurrency=64)
          ├── GPU Worker 2: ...
          └── GPU Worker N (default 6 on 48GB 4090D)
                └── FastDeploy VLM Server (127.0.0.1:8118/v1, max-num-seqs=256, CUDA Graph)

Features:
- Exposes full PaddleX /layout-parsing parameters including repetitionPenalty, temperature, topP,
  maxNewTokens, layoutThreshold, useOcrForImageBlock, vlmExtraArgs, etc.
- Automatic repetition hallucination detection & escalating sampling-parameter retry per page.
- Automatic LOW_TEXT_PAGE detection & fallback recovery for missed-layout pages (skips blank pages).
- Automatic Markdown fenced code block closure (``` / ~~~).
"""

import asyncio
import base64
import io
import multiprocessing as mp
import os
import re
import tempfile
import time
import uuid
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

VLM_SERVER_URL = os.environ.get("VLM_SERVER_URL", "http://127.0.0.1:8118/v1")
VLM_BACKEND = os.environ.get("VLM_BACKEND", "fastdeploy-server")
VLM_MODEL_NAME = os.environ.get("VLM_MODEL_NAME", "")
VL_REC_MAX_CONCURRENCY = int(os.environ.get("VL_REC_MAX_CONCURRENCY", "64"))
OCR_WORKERS = int(os.environ.get("OCR_WORKERS", "6"))
PIPELINE_VERSION = os.environ.get("PIPELINE_VERSION", "v1.6")
USE_DOC_ORIENTATION = os.environ.get("USE_DOC_ORIENTATION", "false").lower() == "true"
USE_DOC_UNWARPING = os.environ.get("USE_DOC_UNWARPING", "false").lower() == "true"
USE_CHART_RECOGNITION = os.environ.get("USE_CHART_RECOGNITION", "false").lower() == "true"
USE_SEAL_RECOGNITION = os.environ.get("USE_SEAL_RECOGNITION", "false").lower() == "true"

# Quality guardrails (enabled by default; override via env or per-request fields)
AUTO_CLOSE_CODE_BLOCKS = os.environ.get("AUTO_CLOSE_CODE_BLOCKS", "true").lower() == "true"
AUTO_DETECT_LOW_TEXT = os.environ.get("AUTO_DETECT_LOW_TEXT", "true").lower() == "true"
AUTO_RETRY_LOW_TEXT = os.environ.get("AUTO_RETRY_LOW_TEXT", "true").lower() == "true"
LOW_TEXT_THRESHOLD = int(os.environ.get("LOW_TEXT_THRESHOLD", "30"))
AUTO_RETRY_REPETITION = os.environ.get("AUTO_RETRY_REPETITION", "true").lower() == "true"

# Per-worker process globals
_worker_pipeline = None
_executor: Optional[ProcessPoolExecutor] = None
_workers_ready: bool = False

_RE_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_RE_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_RE_HTML_TAG = re.compile(r"<[^>]+>")
_RE_TABLE_SEP = re.compile(r"^[\|\s:\-]+$")
_RE_CHAR_FLOOD = re.compile(r"([\w\u4e00-\u9fff])\1{34,}")
_RE_HAS_CONTENT = re.compile(r"[\w\u4e00-\u9fff]")
_RE_OPEN_FENCE = re.compile(r"^[ ]{0,3}(`{3,}|~{3,})(.*)$")
_RE_CLOSE_FENCE = re.compile(r"^[ ]{0,3}(`{3,}|~{3,})[ \t]*$")


def _count_effective_chars(md_text: str) -> int:
    """Count meaningful non-whitespace characters excluding HTML/Markdown image tags."""
    if not md_text:
        return 0
    s = _RE_HTML_COMMENT.sub("", md_text)
    s = _RE_MD_IMAGE.sub("", s)
    s = _RE_HTML_TAG.sub("", s)
    s = re.sub(r"\s+", "", s)
    return len(s)


def _close_unclosed_code_blocks(md_text: str) -> Tuple[str, bool]:
    """Detect and close any unclosed CommonMark fenced code block (``` or ~~~)."""
    if not md_text:
        return md_text, False

    in_fence = False
    fence_char = ""
    fence_len = 0

    for line in md_text.splitlines():
        if not in_fence:
            m = _RE_OPEN_FENCE.match(line)
            if m:
                marker, info = m.group(1), m.group(2)
                if marker[0] == "`" and "`" in info:
                    continue
                in_fence = True
                fence_char = marker[0]
                fence_len = len(marker)
        else:
            m = _RE_CLOSE_FENCE.match(line)
            if m:
                marker = m.group(1)
                if marker[0] == fence_char and len(marker) >= fence_len:
                    in_fence = False
                    fence_char = ""
                    fence_len = 0

    if not in_fence:
        return md_text, False

    closing = fence_char * fence_len
    sep = "" if md_text.endswith("\n") else "\n"
    return f"{md_text}{sep}{closing}", True


def _has_period_loop(s: str, min_period: int = 3, max_period: int = 80, min_repeats: int = 6) -> bool:
    """
    O(L * log(max_period)) check for any substring of length p in [min_period, max_period]
    repeating >= min_repeats times consecutively (Pigeonhole Principle aligned-block scan).
    """
    n = len(s)
    if n < min_period * min_repeats:
        return False
    need_aligned = min_repeats - 1
    upper_p = min(max_period, n // min_repeats)
    for p in range(min_period, upper_p + 1):
        prev = None
        run = 0
        for i in range(0, n - p + 1, p):
            blk = s[i : i + p]
            if blk == prev:
                run += 1
                if run >= need_aligned and _RE_HAS_CONTENT.search(blk):
                    # Verify full consecutive repeat count around aligned blocks
                    if (blk * min_repeats) in s[max(0, i - p * min_repeats) : i + p * 2]:
                        return True
            else:
                prev = blk
                run = 1
    return False


def _detect_repetition_hallucination(md_text: str) -> Optional[str]:
    """Detect degenerate VLM repetition loops while avoiding false positives on normal tables."""
    if not md_text or len(md_text) < 36:
        return None

    clean = _RE_HTML_TAG.sub(" ", _RE_MD_IMAGE.sub(" ", md_text))

    # 1. Single alphanumeric/CJK character flood (>= 35 consecutive)
    if _RE_CHAR_FLOOD.search(clean):
        return "char_flood"

    # 2. Line-level consecutive repetition & high-frequency line dominance
    lines = [ln.strip() for ln in clean.splitlines() if ln.strip()]
    consec_prev = None
    consec_run = 0
    long_line_counts: Counter = Counter()
    total_non_ws = 0

    for ln in lines:
        total_non_ws += len(ln)
        if _RE_TABLE_SEP.match(ln):
            consec_prev = None
            consec_run = 0
            continue
        if len(ln) >= 10:
            if ln == consec_prev:
                consec_run += 1
                if consec_run >= 5:
                    return "consecutive_line_loop"
            else:
                consec_prev = ln
                consec_run = 1
        else:
            consec_prev = None
            consec_run = 0

        if len(ln) >= 20:
            long_line_counts[ln] += 1

    if total_non_ws > 0 and long_line_counts:
        top_line, top_cnt = long_line_counts.most_common(1)[0]
        if top_cnt >= 8 and (len(top_line) * top_cnt) / total_non_ws >= 0.35:
            return "line_dominance_loop"

    # 3. Substring period loop on lines and tail window
    for ln in lines:
        if len(ln) >= 36 and not _RE_TABLE_SEP.match(ln):
            if _has_period_loop(ln):
                return "substring_loop"

    tail_collapsed = re.sub(r"\s+", " ", clean[-2000:]).strip()
    if len(tail_collapsed) >= 36 and _has_period_loop(tail_collapsed):
        return "tail_substring_loop"

    return None


def _get_page_image(res: Any) -> Any:
    """Extract per-page numpy BGR image from PaddleOCRVLResult for isolated single-page retry."""
    if not hasattr(res, "get"):
        return None
    doc_prep = res.get("doc_preprocessor_res")
    if doc_prep is not None and hasattr(doc_prep, "get"):
        img = doc_prep.get("output_img")
        if img is not None and hasattr(img, "shape"):
            return img
    return None


def _has_visual_content(img: Any) -> bool:
    """Return True if page image is visually non-blank (avoids retrying truly blank pages)."""
    if img is None or not hasattr(img, "shape"):
        return False
    try:
        sub = img[::4, ::4]
        if sub.size == 0:
            return False
        gray = sub.mean(axis=2) if sub.ndim == 3 else sub
        non_white_ratio = float((gray < 245).mean())
        std_val = float(gray.std())
        return non_white_ratio >= 0.005 and std_val >= 5.0
    except Exception:
        return True


def _worker_init(config: Dict[str, Any]):
    global _worker_pipeline
    warnings.filterwarnings("ignore")
    from paddleocr import PaddleOCRVL

    kwargs: Dict[str, Any] = {
        "pipeline_version": config["pipeline_version"],
        "vl_rec_backend": config["vlm_backend"],
        "vl_rec_server_url": config["vlm_server_url"],
        "vl_rec_max_concurrency": config["vl_rec_max_concurrency"],
        "use_queues": True,
        "use_doc_orientation_classify": config["use_doc_orientation"],
        "use_doc_unwarping": config["use_doc_unwarping"],
        "use_chart_recognition": config["use_chart_recognition"],
        "use_seal_recognition": config["use_seal_recognition"],
    }
    if config.get("vlm_model_name"):
        kwargs["vl_rec_api_model_name"] = config["vlm_model_name"]

    print(f"[gpu-worker pid={os.getpid()}] Loading PaddleOCRVL({config['pipeline_version']})...", flush=True)
    _worker_pipeline = PaddleOCRVL(**kwargs)
    print(f"[gpu-worker pid={os.getpid()}] Ready.", flush=True)


def _worker_ping() -> int:
    return os.getpid()


def _resolve_one_file(file_val: str, file_type: Optional[int], tmp_dir: Path) -> str:
    if not file_val.startswith(("http://", "https://")) and len(file_val) < 1024:
        p = Path(file_val)
        if p.exists() and p.is_file():
            return str(p)

    if file_val.startswith(("http://", "https://")):
        return file_val

    raw = file_val
    if "," in raw[:100] and raw.startswith("data:"):
        raw = raw.split(",", 1)[1]
    try:
        data = base64.b64decode(raw)
    except Exception as e:
        raise ValueError(f"Invalid file path, URL, or base64 content: {e}")

    if file_type == 0 or data[:4] == b"%PDF":
        ext = ".pdf"
    elif data[:8] == b"\x89PNG\r\n\x1a\n":
        ext = ".png"
    elif data[:2] in (b"II", b"MM"):
        ext = ".tiff"
    else:
        ext = ".jpg"

    out_path = tmp_dir / f"input_{uuid.uuid4().hex}{ext}"
    out_path.write_bytes(data)
    return str(out_path)


def _extract_page_raw(
    res: Any,
    return_md_images: bool,
    return_pruned: bool,
    prettify_markdown: bool,
    show_formula_number: bool,
) -> Tuple[str, Optional[Dict[str, str]], Any]:
    md_text = ""
    md_images: Optional[Dict[str, str]] = None

    md_obj = None
    if hasattr(res, "_to_markdown"):
        try:
            md_obj = res._to_markdown(
                pretty=prettify_markdown,
                show_formula_number=show_formula_number,
            )
        except Exception:
            md_obj = getattr(res, "markdown", None)
    else:
        md_obj = getattr(res, "markdown", None)

    if isinstance(md_obj, dict):
        md_text = md_obj.get("markdown_texts", "") or md_obj.get("text", "")
        if return_md_images:
            imgs = md_obj.get("markdown_images") or md_obj.get("images") or {}
            md_images = {}
            for k, v in imgs.items():
                if isinstance(v, str):
                    md_images[k] = v
                else:
                    buf = io.BytesIO()
                    v.save(buf, format="PNG")
                    md_images[k] = base64.b64encode(buf.getvalue()).decode("ascii")

    pruned = None
    if return_pruned:
        json_obj = getattr(res, "json", None)
        if isinstance(json_obj, dict):
            pruned = json_obj.get("res", json_obj)

    return md_text, md_images, pruned


_API_TO_PREDICT_KWARGS = [
    ("useDocOrientationClassify", "use_doc_orientation_classify"),
    ("useDocUnwarping", "use_doc_unwarping"),
    ("useLayoutDetection", "use_layout_detection"),
    ("useChartRecognition", "use_chart_recognition"),
    ("useSealRecognition", "use_seal_recognition"),
    ("useOcrForImageBlock", "use_ocr_for_image_block"),
    ("layoutThreshold", "layout_threshold"),
    ("layoutNms", "layout_nms"),
    ("layoutUnclipRatio", "layout_unclip_ratio"),
    ("layoutMergeBboxesMode", "layout_merge_bboxes_mode"),
    ("layoutShapeMode", "layout_shape_mode"),
    ("promptLabel", "prompt_label"),
    ("formatBlockContent", "format_block_content"),
    ("repetitionPenalty", "repetition_penalty"),
    ("temperature", "temperature"),
    ("topP", "top_p"),
    ("minPixels", "min_pixels"),
    ("maxPixels", "max_pixels"),
    ("maxNewTokens", "max_new_tokens"),
    ("mergeLayoutBlocks", "merge_layout_blocks"),
    ("markdownIgnoreLabels", "markdown_ignore_labels"),
    ("vlmExtraArgs", "vlm_extra_args"),
]


def _predict_single_page_retry(
    page_img: Any,
    base_kwargs: Dict[str, Any],
    overrides: Dict[str, Any],
    return_md_images: bool,
    return_pruned: bool,
    prettify_markdown: bool,
    show_formula_number: bool,
) -> Optional[Tuple[str, Optional[Dict[str, str]], Any]]:
    retry_kwargs = dict(base_kwargs)
    retry_kwargs.update(overrides)
    retry_outputs = list(_worker_pipeline.predict(page_img, **retry_kwargs))
    if not retry_outputs:
        return None
    return _extract_page_raw(
        retry_outputs[0],
        return_md_images=return_md_images,
        return_pruned=return_pruned,
        prettify_markdown=prettify_markdown,
        show_formula_number=show_formula_number,
    )


def _worker_predict(req_dict: Dict[str, Any]) -> Dict[str, Any]:
    global _worker_pipeline
    t0 = time.perf_counter()
    log_id = req_dict.get("logId") or uuid.uuid4().hex

    return_md_images = bool(req_dict.get("returnMarkdownImages", False))
    return_pruned = bool(req_dict.get("returnPrunedResult", True))
    prettify_markdown = True if req_dict.get("prettifyMarkdown") is None else bool(req_dict["prettifyMarkdown"])
    show_formula_number = bool(req_dict.get("showFormulaNumber", False))

    auto_close_code = (
        AUTO_CLOSE_CODE_BLOCKS
        if req_dict.get("autoCloseCodeBlocks") is None
        else bool(req_dict["autoCloseCodeBlocks"])
    )
    auto_detect_low_text = (
        AUTO_DETECT_LOW_TEXT
        if req_dict.get("autoDetectLowText") is None
        else bool(req_dict["autoDetectLowText"])
    )
    auto_retry_low_text = (
        AUTO_RETRY_LOW_TEXT
        if req_dict.get("autoRetryLowText") is None
        else bool(req_dict["autoRetryLowText"])
    )
    low_text_threshold = (
        LOW_TEXT_THRESHOLD
        if req_dict.get("lowTextThreshold") is None
        else int(req_dict["lowTextThreshold"])
    )

    caller_set_rp = req_dict.get("repetitionPenalty") is not None
    req_auto_retry_rep = req_dict.get("autoRetryRepetition")
    if req_auto_retry_rep is not None:
        auto_retry_rep = bool(req_auto_retry_rep)
    else:
        # Auto-retry by default when caller didn't explicitly supply repetitionPenalty;
        # if caller explicitly supplies repetitionPenalty (e.g. caller-side retry loop), respect it.
        auto_retry_rep = AUTO_RETRY_REPETITION and not caller_set_rp

    with tempfile.TemporaryDirectory(prefix="ocr_req_") as td:
        td_path = Path(td)
        file_type = req_dict.get("fileType")

        if req_dict.get("files"):
            inputs: Union[str, List[str]] = [
                _resolve_one_file(f, file_type, td_path) for f in req_dict["files"]
            ]
        else:
            inputs = _resolve_one_file(req_dict["file"], file_type, td_path)

        predict_kwargs: Dict[str, Any] = {}
        for api_key, kw in _API_TO_PREDICT_KWARGS:
            val = req_dict.get(api_key)
            if val is not None:
                if api_key == "layoutUnclipRatio" and isinstance(val, list):
                    val = tuple(val)
                predict_kwargs[kw] = val

        outputs = list(_worker_pipeline.predict(inputs, **predict_kwargs))

        pages: List[Dict[str, Any]] = []
        for res in outputs:
            md_text, md_images, pruned = _extract_page_raw(
                res,
                return_md_images=return_md_images,
                return_pruned=return_pruned,
                prettify_markdown=prettify_markdown,
                show_formula_number=show_formula_number,
            )
            page_img = _get_page_image(res)
            quality_flags: List[str] = []
            retry_info: Optional[Dict[str, Any]] = None

            # 1. Repetition hallucination detection & escalating sampling-param retry
            rep_reason = _detect_repetition_hallucination(md_text)
            if rep_reason:
                if auto_retry_rep and page_img is not None:
                    base_rp = float(req_dict.get("repetitionPenalty") or 1.0)
                    schedule = [
                        {"repetition_penalty": round(max(base_rp + 0.15, 1.15), 2), "temperature": 0.1, "top_p": 0.9},
                        {"repetition_penalty": round(max(base_rp + 0.30, 1.30), 2), "temperature": 0.25, "top_p": 0.9},
                    ]
                    resolved = False
                    attempts = []
                    for attempt_idx, params in enumerate(schedule, 1):
                        cand = _predict_single_page_retry(
                            page_img,
                            predict_kwargs,
                            params,
                            return_md_images,
                            return_pruned,
                            prettify_markdown,
                            show_formula_number,
                        )
                        if cand is None:
                            continue
                        cand_md, cand_imgs, cand_pruned = cand
                        cand_rep = _detect_repetition_hallucination(cand_md)
                        attempts.append(
                            {
                                "attempt": attempt_idx,
                                "params": params,
                                "hallucination": cand_rep,
                                "chars": len(cand_md),
                            }
                        )
                        if cand_rep is None:
                            md_text, md_images, pruned = cand_md, cand_imgs, cand_pruned
                            resolved = True
                            break
                        # If still hallucinating, prefer shorter non-empty result
                        if cand_md and len(cand_md) < len(md_text):
                            md_text, md_images, pruned = cand_md, cand_imgs, cand_pruned

                    retry_info = {
                        "reason": f"repetition_hallucination:{rep_reason}",
                        "resolved": resolved,
                        "attempts": attempts,
                    }
                    quality_flags.append(
                        "REPETITION_HALLUCINATION_RETRIED" if resolved else "REPETITION_HALLUCINATION"
                    )
                else:
                    quality_flags.append("REPETITION_HALLUCINATION")

            # 2. Low-text page (LOW_TEXT_PAGE) detection & fallback recovery
            effective_chars = _count_effective_chars(md_text)
            if auto_detect_low_text and effective_chars < low_text_threshold:
                if (
                    auto_retry_low_text
                    and predict_kwargs.get("use_layout_detection") is not False
                    and page_img is not None
                    and _has_visual_content(page_img)
                ):
                    low_attempts = []
                    fallback_plan = [
                        {
                            "layout_threshold": 0.25,
                            "use_ocr_for_image_block": True,
                            "repetition_penalty": float(predict_kwargs.get("repetition_penalty") or 1.05),
                        },
                        {
                            "use_layout_detection": False,
                            "repetition_penalty": float(predict_kwargs.get("repetition_penalty") or 1.15),
                            "temperature": 0.1,
                            "top_p": 0.9,
                        },
                    ]
                    best_eff = effective_chars
                    best_cand = None
                    for attempt_idx, overrides in enumerate(fallback_plan, 1):
                        cand = _predict_single_page_retry(
                            page_img,
                            predict_kwargs,
                            overrides,
                            return_md_images,
                            return_pruned,
                            prettify_markdown,
                            show_formula_number,
                        )
                        if cand is None:
                            continue
                        cand_md, cand_imgs, cand_pruned = cand
                        cand_rep = _detect_repetition_hallucination(cand_md)
                        cand_eff = _count_effective_chars(cand_md)
                        low_attempts.append(
                            {
                                "attempt": attempt_idx,
                                "overrides": overrides,
                                "effectiveChars": cand_eff,
                                "hallucination": cand_rep,
                            }
                        )
                        if cand_rep is None and cand_eff > best_eff:
                            best_eff = cand_eff
                            best_cand = (cand_md, cand_imgs, cand_pruned)
                            if best_eff >= low_text_threshold:
                                break

                    recovered = False
                    if best_cand is not None and (
                        best_eff >= low_text_threshold or best_eff >= effective_chars + 10
                    ):
                        md_text, md_images, pruned = best_cand
                        effective_chars = best_eff
                        recovered = effective_chars >= low_text_threshold

                    low_info = {
                        "reason": "low_text_page",
                        "recovered": recovered,
                        "effectiveChars": effective_chars,
                        "attempts": low_attempts,
                    }
                    retry_info = low_info if retry_info is None else {"repetition": retry_info, "lowText": low_info}
                    if recovered:
                        quality_flags.append("LOW_TEXT_PAGE_RECOVERED")

                if effective_chars < low_text_threshold:
                    quality_flags.append("LOW_TEXT_PAGE")

            # 3. Auto-close unclosed Markdown fenced code blocks (``` / ~~~)
            if auto_close_code:
                md_text, was_closed = _close_unclosed_code_blocks(md_text)
                if was_closed:
                    quality_flags.append("CODE_BLOCK_AUTO_CLOSED")

            pages.append(
                {
                    "markdown": {
                        "text": md_text,
                        "images": md_images if return_md_images else None,
                    },
                    "prunedResult": pruned,
                    "effectiveChars": effective_chars,
                    "qualityFlags": quality_flags,
                    "retryInfo": retry_info,
                }
            )

    elapsed = round(time.perf_counter() - t0, 3)
    return {
        "logId": log_id,
        "errorCode": 0,
        "errorMsg": "Success",
        "result": {
            "layoutParsingResults": pages,
            "pageCount": len(pages),
            "elapsedSec": elapsed,
            "workerPid": os.getpid(),
        },
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _executor, _workers_ready
    cfg = {
        "pipeline_version": PIPELINE_VERSION,
        "vlm_backend": VLM_BACKEND,
        "vlm_server_url": VLM_SERVER_URL,
        "vlm_model_name": VLM_MODEL_NAME,
        "vl_rec_max_concurrency": VL_REC_MAX_CONCURRENCY,
        "use_doc_orientation": USE_DOC_ORIENTATION,
        "use_doc_unwarping": USE_DOC_UNWARPING,
        "use_chart_recognition": USE_CHART_RECOGNITION,
        "use_seal_recognition": USE_SEAL_RECOGNITION,
    }
    print(f"[master pid={os.getpid()}] Spawning {OCR_WORKERS} GPU worker processes...", flush=True)
    ctx = mp.get_context("spawn")
    _executor = ProcessPoolExecutor(
        max_workers=OCR_WORKERS,
        mp_context=ctx,
        initializer=_worker_init,
        initargs=(cfg,),
    )
    # Force eager initialization of all N workers before reporting /health OK
    loop = asyncio.get_running_loop()
    futs = [loop.run_in_executor(_executor, _worker_ping) for _ in range(OCR_WORKERS)]
    pids = await asyncio.gather(*futs)
    _workers_ready = True
    print(f"[master pid={os.getpid()}] All {OCR_WORKERS} GPU workers initialized: {sorted(set(pids))}", flush=True)
    yield
    _workers_ready = False
    if _executor is not None:
        _executor.shutdown(wait=False, cancel_futures=True)


app = FastAPI(title="PaddleOCR-VL-1.6 High-Concurrency FastDeploy Pool Gateway", lifespan=lifespan)


class LayoutParsingRequest(BaseModel):
    file: Optional[str] = Field(None, description="Single local path, HTTP(S) URL, or Base64 string")
    files: Optional[List[str]] = Field(None, description="Batch of local paths/URLs/Base64 strings")
    fileType: Optional[int] = Field(None, description="0=PDF, 1=Image; auto-detected if omitted")
    useDocOrientationClassify: Optional[bool] = None
    useDocUnwarping: Optional[bool] = None
    useLayoutDetection: Optional[bool] = None
    useChartRecognition: Optional[bool] = None
    useSealRecognition: Optional[bool] = None
    useOcrForImageBlock: Optional[bool] = None
    layoutThreshold: Optional[Union[float, Dict[str, Any]]] = None
    layoutNms: Optional[bool] = None
    layoutUnclipRatio: Optional[Union[float, List[float], Dict[str, Any]]] = None
    layoutMergeBboxesMode: Optional[Union[str, Dict[str, Any]]] = None
    layoutShapeMode: Optional[str] = None
    promptLabel: Optional[str] = None
    formatBlockContent: Optional[bool] = None
    repetitionPenalty: Optional[float] = Field(None, description="VLM repetition_penalty (> 0), e.g. 1.15")
    temperature: Optional[float] = Field(None, description="VLM sampling temperature (>= 0), e.g. 0.1")
    topP: Optional[float] = Field(None, description="VLM top_p in [0, 1], e.g. 0.9")
    minPixels: Optional[int] = Field(None, gt=0)
    maxPixels: Optional[int] = Field(None, gt=0)
    maxNewTokens: Optional[int] = Field(None, gt=0)
    mergeLayoutBlocks: Optional[bool] = None
    markdownIgnoreLabels: Optional[List[str]] = None
    vlmExtraArgs: Optional[Dict[str, Any]] = None
    prettifyMarkdown: Optional[bool] = True
    showFormulaNumber: Optional[bool] = False
    returnMarkdownImages: Optional[bool] = False
    returnPrunedResult: Optional[bool] = True
    logId: Optional[str] = None

    # Quality guardrail overrides (defaults enabled via server env)
    autoCloseCodeBlocks: Optional[bool] = Field(
        None, description="Auto-close unclosed ``` / ~~~ code blocks (default: server AUTO_CLOSE_CODE_BLOCKS=true)"
    )
    autoDetectLowText: Optional[bool] = Field(
        None, description="Tag LOW_TEXT_PAGE when effectiveChars < lowTextThreshold (default: true)"
    )
    autoRetryLowText: Optional[bool] = Field(
        None, description="Auto-retry non-blank LOW_TEXT_PAGE with relaxed layout/full-page OCR (default: true)"
    )
    lowTextThreshold: Optional[int] = Field(
        None, ge=0, description="Effective char threshold for LOW_TEXT_PAGE (default: 30)"
    )
    autoRetryRepetition: Optional[bool] = Field(
        None,
        description="Auto-retry on repetition hallucination with escalating repetitionPenalty/temperature "
        "(default: true when repetitionPenalty is omitted)",
    )

    @field_validator("repetitionPenalty")
    @classmethod
    def _check_rp(cls, v):
        if v is not None and v <= 0:
            raise ValueError(f"`repetitionPenalty` must be > 0; got {v}")
        return v

    @field_validator("temperature")
    @classmethod
    def _check_temp(cls, v):
        if v is not None and v < 0:
            raise ValueError(f"`temperature` must be >= 0; got {v}")
        return v

    @field_validator("topP")
    @classmethod
    def _check_top_p(cls, v):
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError(f"`topP` must be in [0, 1]; got {v}")
        return v


@app.get("/health")
async def health():
    vlm_health_url = VLM_SERVER_URL.rstrip("/")
    if vlm_health_url.endswith("/v1"):
        vlm_health_url = vlm_health_url[:-3]
    vlm_ok = False
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{vlm_health_url}/health")
            vlm_ok = r.status_code == 200
    except Exception:
        vlm_ok = False

    if not vlm_ok or not _workers_ready or _executor is None:
        raise HTTPException(
            status_code=503,
            detail={"status": "unhealthy", "vlm_ok": vlm_ok, "workers_ready": _workers_ready},
        )
    return {
        "status": "ok",
        "pipeline_version": PIPELINE_VERSION,
        "vlm_backend": VLM_BACKEND,
        "vlm_ok": vlm_ok,
        "gpu_workers": OCR_WORKERS,
        "master_pid": os.getpid(),
        "features": {
            "sampling_params_exposed": [
                "repetitionPenalty",
                "temperature",
                "topP",
                "maxNewTokens",
                "minPixels",
                "maxPixels",
                "vlmExtraArgs",
            ],
            "auto_retry_repetition": AUTO_RETRY_REPETITION,
            "auto_detect_low_text": AUTO_DETECT_LOW_TEXT,
            "auto_retry_low_text": AUTO_RETRY_LOW_TEXT,
            "low_text_threshold": LOW_TEXT_THRESHOLD,
            "auto_close_code_blocks": AUTO_CLOSE_CODE_BLOCKS,
        },
    }


@app.post("/layout-parsing")
async def layout_parsing(req: LayoutParsingRequest):
    if not req.file and not req.files:
        raise HTTPException(status_code=400, detail="Either 'file' or 'files' must be provided.")
    if _executor is None or not _workers_ready:
        raise HTTPException(status_code=503, detail="GPU worker pool not ready.")
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(_executor, _worker_predict, req.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
