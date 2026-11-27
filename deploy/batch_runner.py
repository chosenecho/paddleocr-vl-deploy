#!/usr/bin/env python3
"""
Resumable, high-concurrency 150k-page batch runner for PaddleOCR-VL-1.6 Gateway.

Features:
- Scans input dir for PDFs and images (.pdf, .png, .jpg, .jpeg, .tif, .tiff)
- Splits large PDFs into N-page chunks (default 10) for smooth parallel throughput
- SQLite checkpointing (progress.db) for exact resume after crashes/restarts
- Automatic retries + dead-letter isolation for corrupted files
- Live pages/sec & ETA telemetry
- Merges PDF chunks into clean per-document .md and .json outputs
"""

import argparse
import asyncio
import json
import os
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Tuple

import aiohttp
from pypdf import PdfReader, PdfWriter

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def init_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS tasks (
            task_id TEXT PRIMARY KEY,
            src_path TEXT NOT NULL,
            rel_path TEXT NOT NULL,
            file_type INTEGER NOT NULL,
            chunk_idx INTEGER NOT NULL,
            start_page INTEGER NOT NULL,
            end_page INTEGER NOT NULL,
            num_pages INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            retries INTEGER NOT NULL DEFAULT 0,
            elapsed_sec REAL,
            error_msg TEXT,
            updated_at REAL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status)")
    conn.commit()
    return conn


def discover_and_enqueue(conn: sqlite3.Connection, input_dir: Path, chunk_pages: int) -> Tuple[int, int]:
    existing = {row[0] for row in conn.execute("SELECT task_id FROM tasks")}
    new_tasks = 0
    total_pages = 0

    files = sorted(p for p in input_dir.rglob("*") if p.is_file())
    rows_to_insert = []

    for f in files:
        ext = f.suffix.lower()
        rel = f.relative_to(input_dir).as_posix()
        if ext == ".pdf":
            try:
                reader = PdfReader(str(f))
                n_pages = len(reader.pages)
            except Exception as e:
                tid = f"{rel}#chunk0"
                if tid not in existing:
                    rows_to_insert.append((tid, str(f), rel, 0, 0, 0, 0, 0, "failed", 99, 0.0, f"PdfReadError: {e}", time.time()))
                continue

            total_pages += n_pages
            for chunk_idx, start in enumerate(range(0, n_pages, chunk_pages)):
                end = min(start + chunk_pages, n_pages)
                tid = f"{rel}#chunk{chunk_idx:05d}"
                if tid not in existing:
                    rows_to_insert.append((tid, str(f), rel, 0, chunk_idx, start, end, end - start, "pending", 0, None, None, time.time()))
                    new_tasks += 1
        elif ext in IMAGE_EXTS:
            total_pages += 1
            tid = f"{rel}#img"
            if tid not in existing:
                rows_to_insert.append((tid, str(f), rel, 1, 0, 0, 1, 1, "pending", 0, None, None, time.time()))
                new_tasks += 1

    if rows_to_insert:
        conn.executemany(
            """
            INSERT OR IGNORE INTO tasks
            (task_id, src_path, rel_path, file_type, chunk_idx, start_page, end_page, num_pages, status, retries, elapsed_sec, error_msg, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows_to_insert,
        )
        conn.commit()

    # Reset any stale 'running' tasks from a prior interrupted run
    conn.execute("UPDATE tasks SET status='pending' WHERE status='running'")
    conn.commit()
    return new_tasks, total_pages


def extract_pdf_chunk(src_path: str, start_page: int, end_page: int, out_file: Path) -> str:
    reader = PdfReader(src_path)
    writer = PdfWriter()
    for p in range(start_page, end_page):
        writer.add_page(reader.pages[p])
    with open(out_file, "wb") as fp:
        writer.write(fp)
    return str(out_file)


async def wait_for_health(session: aiohttp.ClientSession, base_url: str, max_wait: int = 300):
    t0 = time.time()
    while time.time() - t0 < max_wait:
        try:
            async with session.get(f"{base_url}/health", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        await asyncio.sleep(3)
    return False


async def run_batch(args):
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    chunks_out_dir = output_dir / "_chunks"
    chunks_out_dir.mkdir(parents=True, exist_ok=True)

    db_path = output_dir / "progress.db"
    conn = init_db(db_path)

    print(f"[scan] Scanning {input_dir} (chunk_pages={args.chunk_pages})...")
    new_tasks, discovered_pages = discover_and_enqueue(conn, input_dir, args.chunk_pages)

    pending_rows = conn.execute(
        """
        SELECT task_id, src_path, rel_path, file_type, chunk_idx, start_page, end_page, num_pages, retries
        FROM tasks
        WHERE status = 'pending' OR (status = 'failed' AND retries < ?)
        ORDER BY src_path, chunk_idx
        """,
        (args.max_retries,),
    ).fetchall()

    total_pending_pages = sum(r[7] for r in pending_rows)
    done_pages_initial = conn.execute("SELECT COALESCE(SUM(num_pages), 0) FROM tasks WHERE status='done'").fetchone()[0]
    print(
        f"[queue] New tasks={new_tasks}, pending_tasks={len(pending_rows)} ({total_pending_pages} pages), "
        f"already_done={done_pages_initial} pages, total_discovered={discovered_pages} pages"
    )
    if not pending_rows:
        print("[done] No pending tasks.")
        return

    queue: asyncio.Queue = asyncio.Queue()
    for row in pending_rows:
        queue.put_nowait(row)

    stats = {"done_pages": 0, "done_tasks": 0, "failed_tasks": 0, "start_time": time.perf_counter()}
    base_url = args.server_url.rstrip("/")

    connector = aiohttp.TCPConnector(limit=args.concurrency + 8)
    timeout = aiohttp.ClientTimeout(total=args.timeout)

    with tempfile.TemporaryDirectory(prefix="ocr_chunks_", dir=args.tmp_dir) as tmp_root:
        tmp_root_path = Path(tmp_root)

        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            if not await wait_for_health(session, base_url, max_wait=60):
                raise RuntimeError(f"OCR Gateway {base_url}/health is not healthy!")

            async def worker(wid: int):
                while True:
                    try:
                        row = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        return

                    task_id, src_path, rel_path, file_type, chunk_idx, start_p, end_p, num_pages, retries = row
                    conn.execute("UPDATE tasks SET status='running', updated_at=? WHERE task_id=?", (time.time(), task_id))
                    conn.commit()

                    chunk_tmp: Optional[Path] = None
                    try:
                        if file_type == 0:
                            chunk_tmp = tmp_root_path / f"w{wid}_{chunk_idx}_{os.getpid()}_{time.time_ns()}.pdf"
                            send_path = await asyncio.to_thread(extract_pdf_chunk, src_path, start_p, end_p, chunk_tmp)
                        else:
                            send_path = src_path

                        payload = {
                            "file": send_path,
                            "fileType": file_type,
                            "useDocOrientationClassify": args.use_orientation,
                            "useDocUnwarping": args.use_unwarping,
                            "useChartRecognition": args.use_chart,
                            "useSealRecognition": args.use_seal,
                            "returnMarkdownImages": args.save_images,
                            "returnPrunedResult": args.save_json,
                        }

                        t0 = time.perf_counter()
                        async with session.post(f"{base_url}/layout-parsing", json=payload) as resp:
                            if resp.status in (502, 503, 504):
                                text = await resp.text()
                                raise ConnectionError(f"HTTP {resp.status}: {text[:200]}")
                            if resp.status != 200:
                                text = await resp.text()
                                raise RuntimeError(f"HTTP {resp.status}: {text[:400]}")
                            data = await resp.json()

                        elapsed = time.perf_counter() - t0
                        pages = data["result"]["layoutParsingResults"]

                        # Save chunk output atomically
                        safe_rel = Path(rel_path)
                        chunk_dir = chunks_out_dir / safe_rel.parent / safe_rel.stem
                        chunk_dir.mkdir(parents=True, exist_ok=True)

                        md_combined = "\n\n---\n\n".join(p["markdown"]["text"] for p in pages)
                        md_tmp = chunk_dir / f"chunk_{chunk_idx:05d}.md.tmp"
                        md_final = chunk_dir / f"chunk_{chunk_idx:05d}.md"
                        md_tmp.write_text(md_combined, encoding="utf-8")
                        md_tmp.replace(md_final)

                        if args.save_json:
                            js_tmp = chunk_dir / f"chunk_{chunk_idx:05d}.json.tmp"
                            js_final = chunk_dir / f"chunk_{chunk_idx:05d}.json"
                            js_tmp.write_text(
                                json.dumps([p.get("prunedResult") for p in pages], ensure_ascii=False),
                                encoding="utf-8",
                            )
                            js_tmp.replace(js_final)

                        conn.execute(
                            "UPDATE tasks SET status='done', elapsed_sec=?, error_msg=NULL, updated_at=? WHERE task_id=?",
                            (round(elapsed, 3), time.time(), task_id),
                        )
                        conn.commit()
                        stats["done_pages"] += num_pages
                        stats["done_tasks"] += 1

                    except Exception as e:
                        new_retries = retries + 1
                        err_str = f"{type(e).__name__}: {e}"[:500]
                        if new_retries < args.max_retries:
                            conn.execute(
                                "UPDATE tasks SET status='pending', retries=?, error_msg=?, updated_at=? WHERE task_id=?",
                                (new_retries, err_str, time.time(), task_id),
                            )
                            conn.commit()
                            if isinstance(e, (ConnectionError, aiohttp.ClientError)):
                                await wait_for_health(session, base_url, max_wait=180)
                            queue.put_nowait((task_id, src_path, rel_path, file_type, chunk_idx, start_p, end_p, num_pages, new_retries))
                        else:
                            conn.execute(
                                "UPDATE tasks SET status='failed', retries=?, error_msg=?, updated_at=? WHERE task_id=?",
                                (new_retries, err_str, time.time(), task_id),
                            )
                            conn.commit()
                            stats["failed_tasks"] += 1
                            print(f"[FAILED] {task_id}: {err_str}", flush=True)
                    finally:
                        if chunk_tmp and chunk_tmp.exists():
                            try:
                                chunk_tmp.unlink()
                            except OSError:
                                pass
                        queue.task_done()

            async def reporter():
                while True:
                    await asyncio.sleep(10)
                    dt = max(time.perf_counter() - stats["start_time"], 0.001)
                    pps = stats["done_pages"] / dt
                    rem = max(total_pending_pages - stats["done_pages"], 0)
                    eta_h = (rem / pps / 3600.0) if pps > 0 else float("inf")
                    print(
                        f"[progress] pages={stats['done_pages']}/{total_pending_pages} "
                        f"({stats['done_pages']/max(total_pending_pages,1)*100:.1f}%) | "
                        f"speed={pps:.2f} pages/s | failed={stats['failed_tasks']} | ETA={eta_h:.2f}h",
                        flush=True,
                    )

            rep_task = asyncio.create_task(reporter())
            workers = [asyncio.create_task(worker(i)) for i in range(args.concurrency)]
            await asyncio.gather(*workers)
            rep_task.cancel()

    # Merge completed PDF chunks into final markdown files
    if args.merge_md:
        merged_dir = output_dir / "markdown"
        merged_dir.mkdir(parents=True, exist_ok=True)
        docs = conn.execute(
            "SELECT DISTINCT rel_path FROM tasks WHERE status='done'"
        ).fetchall()
        for (rel_path,) in docs:
            incomplete = conn.execute(
                "SELECT COUNT(*) FROM tasks WHERE rel_path=? AND status!='done'", (rel_path,)
            ).fetchone()[0]
            if incomplete > 0:
                continue
            safe_rel = Path(rel_path)
            chunk_dir = chunks_out_dir / safe_rel.parent / safe_rel.stem
            md_files = sorted(chunk_dir.glob("chunk_*.md"))
            if md_files:
                out_md = merged_dir / safe_rel.with_suffix(".md")
                out_md.parent.mkdir(parents=True, exist_ok=True)
                out_md.write_text(
                    "\n\n".join(f.read_text(encoding="utf-8") for f in md_files),
                    encoding="utf-8",
                )

    dt = max(time.perf_counter() - stats["start_time"], 0.001)
    print(
        f"[finished] processed={stats['done_pages']} pages in {dt:.1f}s "
        f"({stats['done_pages']/dt:.2f} pages/s), failed_tasks={stats['failed_tasks']}"
    )


def main():
    p = argparse.ArgumentParser(description="Resumable high-concurrency batch OCR runner")
    p.add_argument("--input-dir", required=True, help="Directory containing PDFs / images")
    p.add_argument("--output-dir", required=True, help="Directory for outputs & progress.db")
    p.add_argument("--server-url", default="http://127.0.0.1:8080", help="OCR Gateway base URL")
    p.add_argument("--concurrency", type=int, default=8, help="Concurrent HTTP requests (recommend 2x gateway workers)")
    p.add_argument("--chunk-pages", type=int, default=10, help="Pages per PDF chunk")
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--tmp-dir", default="/dev/shm", help="Fast temp dir for split PDF chunks")
    p.add_argument("--save-json", action="store_true", help="Also save pruned JSON results")
    p.add_argument("--save-images", action="store_true", help="Also return markdown embedded images")
    p.add_argument("--merge-md", action="store_true", default=True, help="Merge chunks into final per-doc .md")
    p.add_argument("--use-orientation", action="store_true")
    p.add_argument("--use-unwarping", action="store_true")
    p.add_argument("--use-chart", action="store_true")
    p.add_argument("--use-seal", action="store_true")
    args = p.parse_args()
    asyncio.run(run_batch(args))


if __name__ == "__main__":
    main()
