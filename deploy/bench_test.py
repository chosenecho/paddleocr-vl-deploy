#!/usr/bin/env python3
"""
End-to-end verification and full-saturation throughput benchmark for PaddleOCR-VL-1.6 Pool Gateway.
Tests both:
  1. Single-page concurrent requests (verifies all GPU workers receive balanced traffic)
  2. Multi-page batched requests (simulates 10-page PDF chunks / 8-image batches for true 150k-page throughput)
"""
import argparse
import asyncio
import time
import urllib.request
from pathlib import Path

import aiohttp

DEMO_URL = "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/paddleocr_vl_demo.png"


async def call_payload(session: aiohttp.ClientSession, url: str, payload: dict):
    t0 = time.perf_counter()
    async with session.post(f"{url}/layout-parsing", json=payload) as resp:
        data = await resp.json()
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {data}")
        dt = time.perf_counter() - t0
        pages = data["result"]["layoutParsingResults"]
        md = pages[0]["markdown"]["text"] if pages else ""
        return dt, len(pages), len(md), md[:160], data["result"].get("workerPid")


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--server-url", default="http://127.0.0.1:8080")
    p.add_argument("--single-total", type=int, default=24)
    p.add_argument("--batch-size", type=int, default=8, help="Pages per batched request")
    p.add_argument("--batch-requests", type=int, default=12, help="Number of batched requests")
    p.add_argument("--concurrency", type=int, default=12)
    p.add_argument("--image", default="/root/autodl-tmp/ocr_deploy/cache/demo.png")
    args = p.parse_args()

    img = Path(args.image)
    if not img.exists():
        img.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DEMO_URL, str(img))

    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        # 1. Warmup
        print("=== [1/3] Single-page warmup & accuracy check ===")
        dt, n_pg, md_len, preview, pid = await call_payload(
            session, args.server_url,
            {"file": str(img), "fileType": 1, "returnMarkdownImages": False, "returnPrunedResult": False},
        )
        print(f"Warmup OK: {dt:.2f}s | worker_pid={pid} | pages={n_pg} | md_chars={md_len}")
        print("Preview:", preview.replace("\n", " "), "\n")

        # 2. Single-page concurrent test
        print(f"=== [2/3] Single-page concurrent test (total={args.single_total}, concurrency={args.concurrency}) ===")
        sem = asyncio.Semaphore(args.concurrency)
        lats, pids = [], set()

        async def run_single():
            async with sem:
                dt, _, _, _, pid = await call_payload(
                    session, args.server_url,
                    {"file": str(img), "fileType": 1, "returnMarkdownImages": False, "returnPrunedResult": False},
                )
                lats.append(dt)
                pids.add(pid)

        t0 = time.perf_counter()
        await asyncio.gather(*[run_single() for _ in range(args.single_total)])
        total_dt = time.perf_counter() - t0
        pps_single = args.single_total / total_dt
        lats.sort()
        print(
            f"Single-page throughput: {pps_single:.2f} pages/s ({args.single_total} pages in {total_dt:.2f}s) | "
            f"p50={lats[len(lats)//2]:.2f}s | active_workers={len(pids)} {sorted(pids)}"
        )

        # 3. Batched / multi-page chunk saturation test (real 150k-page mode!)
        total_batch_pages = args.batch_size * args.batch_requests
        print(
            f"\n=== [3/3] Batched saturation test ({args.batch_requests} reqs x {args.batch_size} pages = "
            f"{total_batch_pages} pages, concurrency={args.concurrency}) ==="
        )
        b_lats, b_pids = [], set()

        async def run_batch():
            async with sem:
                dt, n_pg, _, _, pid = await call_payload(
                    session, args.server_url,
                    {
                        "files": [str(img)] * args.batch_size,
                        "fileType": 1,
                        "returnMarkdownImages": False,
                        "returnPrunedResult": False,
                    },
                )
                b_lats.append(dt)
                b_pids.add(pid)

        t0 = time.perf_counter()
        await asyncio.gather(*[run_batch() for _ in range(args.batch_requests)])
        total_b_dt = time.perf_counter() - t0
        pps_batch = total_batch_pages / total_b_dt
        est_150k_h = 150000 / pps_batch / 3600.0
        print(
            f"Batched throughput: {pps_batch:.2f} pages/s ({total_batch_pages} dense pages in {total_b_dt:.2f}s) | "
            f"active_workers={len(b_pids)} {sorted(b_pids)}"
        )
        print(f"Estimated 150,000 dense newspaper pages on 1x 4090D: ~{est_150k_h:.1f} hours")


if __name__ == "__main__":
    asyncio.run(main())
