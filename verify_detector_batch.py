"""
Verifies that OCR.detect_sorted_batch() (batched -- uses the detector's own
.batch_predict() when available, e.g. ocr.det_backend: mineru) returns the
SAME results as calling detect_sorted() once per page (the old sequential
way) -- and reports the speedup from batching.

Runs BOTH on the exact same rendered pages of a real PDF (converted to BGR
the same way document_pipeline.py does before detecting), then compares,
page by page and box by box: same number of boxes, same coordinates.

If detect_sorted_batch() ever reordered results relative to the input
image list, or a backend's batch_predict() mis-cropped/mis-shaped
something, this would show up here as a mismatch, so a clean run is a
real correctness check -- not just "it didn't crash".

Usage:
    python verify_detector_batch.py path/to/file.pdf
"""

from __future__ import annotations

import sys
import time
from pathlib import Path


def _mismatch(pn: int, reason: str) -> str:
    return f"Page {pn + 1}: MISMATCH -- {reason}"


def main():
    if len(sys.argv) < 2:
        print("Usage: python verify_detector_batch.py <file.pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    if not Path(pdf_path).is_file():
        print(f"Not a file: {pdf_path}")
        sys.exit(1)

    print("Importing project modules...", flush=True)
    import cv2
    import numpy as np

    from module.ocr import get_ocr_engine
    from module.pipeline.config import load_pipeline_conf
    from module.pipeline.loader import load_pdf_pages

    conf = load_pipeline_conf()

    print(f"Building OCR engine (det_backend={conf.get('ocr', {}).get('det_backend', 'onnx')})...", flush=True)
    ocr = get_ocr_engine(conf)

    print(f"Rendering pages from {pdf_path}...", flush=True)
    pages = load_pdf_pages(pdf_path, dpi=conf.get("pdf", {}).get("dpi", 200))
    print(f"Rendered {len(pages)} page(s).", flush=True)

    imgs_bgr = [cv2.cvtColor(np.array(p.convert("RGB")), cv2.COLOR_RGB2BGR) for p in pages]

    print("\nRunning SEQUENTIAL detect_sorted() once per page (old way)...", flush=True)
    t0 = time.time()
    sequential_results = [ocr.detect_sorted(img) for img in imgs_bgr]
    t_sequential = time.time() - t0
    print(f"  done in {t_sequential:.2f}s", flush=True)

    print("\nRunning BATCHED detect_sorted_batch() for all pages in one call (new way)...", flush=True)
    t0 = time.time()
    batched_results = ocr.detect_sorted_batch(imgs_bgr)
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
        for pn, (seq_boxes, batch_boxes) in enumerate(zip(sequential_results, batched_results)):
            seq_none = seq_boxes is None
            batch_none = batch_boxes is None
            if seq_none != batch_none:
                mismatches.append(_mismatch(
                    pn, f"one is None and the other isn't: sequential={seq_none} batched={batch_none}"
                ))
                continue
            if seq_none:
                continue

            if len(seq_boxes) != len(batch_boxes):
                mismatches.append(_mismatch(
                    pn, f"box count differs: sequential={len(seq_boxes)} batched={len(batch_boxes)}"
                ))
                continue

            for i, (sb, bb) in enumerate(zip(seq_boxes, batch_boxes)):
                if not np.allclose(sb, bb, atol=1e-3):
                    mismatches.append(_mismatch(
                        pn, f"box {i}: coords differ: sequential={sb.tolist()} batched={bb.tolist()}"
                    ))

    if mismatches:
        print(f"\n{len(mismatches)} MISMATCH(ES) FOUND:\n")
        for m in mismatches:
            print(" -", m)
        print("\n=> detect_sorted_batch() is NOT safe to use as-is.")
        sys.exit(1)

    print(f"\nAll {len(sequential_results)} page(s) match EXACTLY (same boxes, same coords).")
    print(f"\nTiming: sequential={t_sequential:.2f}s  batched={t_batched:.2f}s  "
          f"speedup={t_sequential / t_batched if t_batched > 0 else float('inf'):.2f}x")
    print("\n=> detect_sorted_batch() is safe to use -- results are identical to the old per-page path.")


if __name__ == "__main__":
    main()
