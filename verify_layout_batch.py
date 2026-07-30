"""
Verifies that PPDocLayoutBackend.detect_batch() (batched, one PaddleX call
for all pages) returns the SAME results as calling detect() once per page
(the old sequential way) -- and reports the speedup from batching.

Runs BOTH on the exact same rendered pages of a real PDF, then compares,
page by page and box by box:
    - same number of boxes
    - same label, in the same order
    - same bbox coordinates and score (exact match expected -- both paths
      run the identical model on the identical image, batching only changes
      how many images go through PaddleX per call, not the math)

If detect_batch() ever reorders results relative to the input image list,
this is exactly what would show up here as a mismatch (wrong label/bbox
for a page), so a clean run of this script is a real correctness check --
not just "it didn't crash".

Usage:
    python verify_layout_batch.py path/to/file.pdf
"""

from __future__ import annotations

import sys
import time
from pathlib import Path


def _mismatch(pn: int, reason: str) -> str:
    return f"Page {pn + 1}: MISMATCH -- {reason}"


def main():
    if len(sys.argv) < 2:
        print("Usage: python verify_layout_batch.py <file.pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    if not Path(pdf_path).is_file():
        print(f"Not a file: {pdf_path}")
        sys.exit(1)

    print("Importing project modules...", flush=True)
    from module.layout import get_layout_backend
    from module.pipeline.config import load_pipeline_conf
    from module.pipeline.loader import load_pdf_pages

    conf = load_pipeline_conf()
    threshold = conf.get("layout", {}).get("threshold", 0.2)

    print("Building layout backend...", flush=True)
    layout = get_layout_backend(conf)

    print(f"Rendering pages from {pdf_path}...", flush=True)
    pages = load_pdf_pages(pdf_path, dpi=conf.get("pdf", {}).get("dpi", 200))
    print(f"Rendered {len(pages)} page(s).", flush=True)

    print("\nRunning SEQUENTIAL detect() once per page (old way)...", flush=True)
    t0 = time.time()
    sequential_results = [layout.detect(img, threshold) for img in pages]
    t_sequential = time.time() - t0
    print(f"  done in {t_sequential:.2f}s", flush=True)

    print("\nRunning BATCHED detect_batch() for all pages in one call (new way)...", flush=True)
    t0 = time.time()
    batched_results = layout.detect_batch(pages, threshold)
    t_batched = time.time() - t0
    print(f"  done in {t_batched:.2f}s", flush=True)

    print("\n" + "=" * 60)
    print("COMPARING RESULTS")
    print("=" * 60)

    mismatches = []

    if len(sequential_results) != len(batched_results):
        mismatches.append(
            f"page COUNT differs: sequential={len(sequential_results)} "
            f"batched={len(batched_results)}"
        )
    else:
        for pn, (seq_blocks, batch_blocks) in enumerate(zip(sequential_results, batched_results)):
            if len(seq_blocks) != len(batch_blocks):
                mismatches.append(_mismatch(
                    pn, f"box count differs: sequential={len(seq_blocks)} batched={len(batch_blocks)}"
                ))
                continue

            for i, (sb, bb) in enumerate(zip(seq_blocks, batch_blocks)):
                if sb["type"] != bb["type"]:
                    mismatches.append(_mismatch(
                        pn, f"box {i}: label differs: sequential={sb['type']!r} batched={bb['type']!r}"
                    ))
                    continue
                if sb["bbox"] != bb["bbox"]:
                    mismatches.append(_mismatch(
                        pn, f"box {i} ({sb['type']}): bbox differs: sequential={sb['bbox']} batched={bb['bbox']}"
                    ))
                if abs(sb["score"] - bb["score"]) > 1e-6:
                    mismatches.append(_mismatch(
                        pn, f"box {i} ({sb['type']}): score differs: sequential={sb['score']:.4f} batched={bb['score']:.4f}"
                    ))

    if mismatches:
        print(f"\n{len(mismatches)} MISMATCH(ES) FOUND:\n")
        for m in mismatches:
            print(" -", m)
        print("\n=> detect_batch() is NOT safe to use as-is -- do not switch the real pipeline over yet.")
        sys.exit(1)

    print(f"\nAll {len(sequential_results)} page(s) match EXACTLY (same boxes, labels, bboxes, scores).")
    print(f"\nTiming: sequential={t_sequential:.2f}s  batched={t_batched:.2f}s  "
          f"speedup={t_sequential / t_batched if t_batched > 0 else float('inf'):.2f}x")
    print("\n=> detect_batch() is safe to use -- results are identical to the old per-page path.")


if __name__ == "__main__":
    main()
