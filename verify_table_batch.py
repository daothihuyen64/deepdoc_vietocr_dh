"""
Verifies that MinerUTableProcessor.process_batch() (batches Surya's
TableRecPredictor + DetectionPredictor across MULTIPLE table crops in one
call each) returns the SAME content as calling the processor once per crop
via __call__() (the old way) -- and reports the speedup from batching.

Runs BOTH over every image in a folder of table crops, then compares the
final HTML content string image by image.

NOTE: unlike verify_layout_batch.py/verify_detector_batch.py, exact string
equality here is a slightly stronger bar -- Surya's predictors are
transformer-based and batch composition CAN shift floating-point results
by a hair on GPU (different attention/conv kernel selected for a different
batch size). If you see mismatches that are only 1-2 characters different
in a low-confidence cell (not whole rows/columns missing or shuffled),
that's most likely this, not a real bug -- re-run
verify_detector_batch.py-style reasoning applies: a few tiny mismatches on
GPU are a known acceptable cost of batching, wholesale wrong/reordered
tables are not.

Usage:
    python verify_table_batch.py [input_dir]

    input_dir defaults to data_test/ (same convention as test_table_models.py).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def main():
    input_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data_test")
    if not input_dir.is_dir():
        print(f"Not a directory: {input_dir}")
        sys.exit(1)

    images = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        print(f"No images found in {input_dir}")
        sys.exit(1)
    print(f"Found {len(images)} image(s) in {input_dir}")

    print("Importing project modules...", flush=True)
    from PIL import Image

    from module.ocr import get_ocr_engine
    from module.pipeline.config import load_pipeline_conf
    from module.table.mineru_processor import MinerUTableProcessor

    conf = load_pipeline_conf()

    print("Building OCR engine + MinerUTableProcessor...", flush=True)
    ocr = get_ocr_engine(conf)
    processor = MinerUTableProcessor(ocr)
    print("Ready.", flush=True)

    crops = [Image.open(p) for p in images]

    print(f"\nRunning SEQUENTIAL __call__() once per crop (old way) -- {len(crops)} crop(s)...", flush=True)
    t0 = time.time()
    sequential_results = [
        processor(crop, pn=0, tno=i) for i, crop in enumerate(crops)
    ]
    t_sequential = time.time() - t0
    print(f"  done in {t_sequential:.2f}s", flush=True)

    print(f"\nRunning BATCHED process_batch() for all {len(crops)} crop(s) in one call (new way)...", flush=True)
    t0 = time.time()
    batched_results = processor.process_batch([(0, i, crop) for i, crop in enumerate(crops)])
    t_batched = time.time() - t0
    print(f"  done in {t_batched:.2f}s", flush=True)

    print("\n" + "=" * 60)
    print("COMPARING RESULTS")
    print("=" * 60)

    mismatches = []
    if len(sequential_results) != len(batched_results):
        mismatches.append(f"result COUNT differs: sequential={len(sequential_results)} batched={len(batched_results)}")
    else:
        for i, (seq_html, batch_html) in enumerate(zip(sequential_results, batched_results)):
            if seq_html != batch_html:
                mismatches.append(
                    f"{images[i].name}: MISMATCH\n"
                    f"    sequential: {seq_html[:200]!r}\n"
                    f"    batched:    {batch_html[:200]!r}"
                )

    if mismatches:
        print(f"\n{len(mismatches)} MISMATCH(ES) FOUND:\n")
        for m in mismatches:
            print(" -", m)
        print("\nSee module docstring above -- tiny 1-2 char diffs in a single cell can be")
        print("expected GPU batch-composition float drift; whole rows/cols wrong is a real bug.")
        sys.exit(1)

    print(f"\nAll {len(sequential_results)} image(s) match EXACTLY (identical HTML content).")
    print(f"\nTiming: sequential={t_sequential:.2f}s  batched={t_batched:.2f}s  "
          f"speedup={t_sequential / t_batched if t_batched > 0 else float('inf'):.2f}x")
    print("\n=> process_batch() is safe to use -- results are identical to the old per-crop path.")


if __name__ == "__main__":
    main()
