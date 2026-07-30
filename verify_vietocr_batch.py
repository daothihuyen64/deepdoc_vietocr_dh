"""
Verifies that Predictor.predict_batch() (batches VietOCR's encoder+decoder
across MULTIPLE variable-width text-line crops in one forward pass, using
pack_padded_sequence + attention masking so padding never contaminates a
shorter crop's result -- see vietocr/model/seqmodel/seq2seq.py) returns the
SAME text as calling predict() once per crop (the old way) -- and reports
the speedup from batching.

Runs BOTH over every image in a folder of text-line crops, then compares
the recognized text string image by image.

NOTE: unlike verify_layout_batch.py/verify_detector_batch.py (deterministic
detectors), a text MISMATCH here is a stronger signal of a real bug than for
verify_table_batch.py's Surya comparison -- VietOCR's own greedy decoding
is a plain argmax over softmax probabilities, so batch-composition
floating-point drift on GPU could in principle flip a single LOW-CONFIDENCE
character right at a decision boundary (same class of noise documented in
verify_table_batch.py), but whole words/characters wrong, or text that only
degrades for crops SHORTER than the widest one in their batch, points
directly at the padding/masking implementation (pack_padded_sequence in
Encoder.forward / the mask in Attention.forward) rather than incidental
kernel noise.

Usage:
    python verify_vietocr_batch.py [input_dir]

    input_dir defaults to data_test/ (same convention as the other verify_*.py
    scripts) -- but for a meaningful check, point it at a folder of actual
    TEXT-LINE crops (small, wide images), not whole table/page images. A
    good real source: pass crop_debug_dir=... to run_ocr_page() (see
    module/pipeline/ocr_page.py) once to dump real per-box crops from an
    actual document, then point this script at that folder.
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

    conf = load_pipeline_conf()

    print("Building OCR engine (VietOCR predictor)...", flush=True)
    ocr = get_ocr_engine(conf)
    predictor = ocr.text_recognizer[0].detector
    print("Ready.", flush=True)

    crops = [Image.open(p).convert("RGB") for p in images]

    print(f"\nRunning SEQUENTIAL predict() once per crop (old way) -- {len(crops)} crop(s)...", flush=True)
    t0 = time.time()
    sequential_results = [predictor.predict(crop, return_prob=True) for crop in crops]
    t_sequential = time.time() - t0
    print(f"  done in {t_sequential:.2f}s", flush=True)

    print(f"\nRunning BATCHED predict_batch() for all {len(crops)} crop(s) in one call (new way)...", flush=True)
    t0 = time.time()
    batched_results = predictor.predict_batch(crops)
    t_batched = time.time() - t0
    print(f"  done in {t_batched:.2f}s", flush=True)

    print("\n" + "=" * 60)
    print("COMPARING RESULTS")
    print("=" * 60)

    mismatches = []
    if len(sequential_results) != len(batched_results):
        mismatches.append(f"result COUNT differs: sequential={len(sequential_results)} batched={len(batched_results)}")
    else:
        for i, ((seq_text, seq_prob), (batch_text, batch_prob)) in enumerate(zip(sequential_results, batched_results)):
            if seq_text != batch_text:
                mismatches.append(
                    f"{images[i].name}: TEXT MISMATCH\n"
                    f"    sequential: {seq_text!r} (prob={seq_prob:.3f})\n"
                    f"    batched:    {batch_text!r} (prob={batch_prob:.3f})"
                )

    if mismatches:
        print(f"\n{len(mismatches)} MISMATCH(ES) FOUND:\n")
        for m in mismatches:
            print(" -", m)
        print("\nSee module docstring above -- a single low-confidence character flipped by")
        print("GPU batch-composition float drift can be expected noise; wrong words, or a")
        print("pattern where only SHORTER crops in a batch degrade, points at a real bug in")
        print("the pack_padded_sequence/attention-mask implementation instead.")
        sys.exit(1)

    print(f"\nAll {len(sequential_results)} image(s) match EXACTLY (identical recognized text).")
    print(f"\nTiming: sequential={t_sequential:.2f}s  batched={t_batched:.2f}s  "
          f"speedup={t_sequential / t_batched if t_batched > 0 else float('inf'):.2f}x")
    print("\n=> predict_batch() is safe to use -- results are identical to the old per-crop path.")


if __name__ == "__main__":
    main()
