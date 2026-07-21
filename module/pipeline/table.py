import re

import cv2
import numpy as np
from PIL import Image

from ..ocr import OCREngine
from ..recognizer import Recognizer
from ..tsr import TableStructureRecognizer

# ── Deskew the cropped table image before TSR + OCR ──────────────────────────
# Detects the skew angle via the table's ruling lines (Hough Transform) and
# rotates it straight BEFORE running TSR and OCR -- both then run on the
# same deskewed crop so bbox coordinates still line up with each other.


def _detect_skew_angle(img_bgr: np.ndarray, angle_tolerance: float = 25.0, min_line_len_ratio: float = 0.3) -> float:
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    min_line_len = int(w * min_line_len_ratio)
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 360, threshold=80,
        minLineLength=min_line_len, maxLineGap=10,
    )
    if lines is None:
        return 0.0
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) <= angle_tolerance:
            angles.append(angle)
    return float(np.median(angles)) if angles else 0.0


def _rotate_image(img_bgr: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    return cv2.warpAffine(img_bgr, M, (new_w, new_h),
                           flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def deskew_crop(crop_img_pil: Image.Image, min_angle: float = 0.1) -> tuple[Image.Image, float]:
    """Returns (rotated-straight PIL image, angle rotated -- 0.0 if negligible)."""
    img_bgr = cv2.cvtColor(np.array(crop_img_pil), cv2.COLOR_RGB2BGR)
    angle = _detect_skew_angle(img_bgr)
    if abs(angle) < min_angle:
        return crop_img_pil, 0.0
    rotated_bgr = _rotate_image(img_bgr, angle)
    rotated_rgb = cv2.cvtColor(rotated_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rotated_rgb), angle


# ── Table -> Markdown via TSR ─────────────────────────────────────────────────


def raw_ocr(img_pil: Image.Image, ocr: OCREngine, crop_debug_dir: str | None = None):
    result = ocr(np.array(img_pil), crop_debug_dir=crop_debug_dir)
    return result if isinstance(result, list) else []


def table_to_markdown(crop_img: Image.Image, tsr_cpns: list[dict], ocr: OCREngine, raw=None) -> str:
    if raw is None:
        raw = raw_ocr(crop_img, ocr)
    if not raw:
        return ""
    boxes = Recognizer.sort_Y_firstly(
        [{"x0": b[0][0], "x1": b[1][0],
          "top": b[0][1], "text": t[0],
          "bottom": b[-1][1],
          "layout_type": "table", "page_number": 0}
         for b, t in raw if b[0][0] <= b[1][0] and b[0][1] <= b[-1][1]],
        np.mean([b[-1][1] - b[0][1] for b, _ in raw]) / 3
    )

    def gather(kwd, fzy=10, ption=0.6):
        nonlocal boxes
        eles = Recognizer.sort_Y_firstly(
            [r for r in tsr_cpns if re.match(kwd, r["label"])], fzy)
        eles = Recognizer.layouts_cleanup(boxes, eles, 5, ption)
        return Recognizer.sort_Y_firstly(eles, 0)

    # Rows gather -- excludes header-labels already covered > 0.5 by a real
    # "table row", BEFORE feeding into layouts_cleanup. Avoids a chain
    # reaction that would wrongly delete both the real table-row AND the
    # table-column-header when both overlap each other (2-line headers).
    def gather_rows(fzy=10, ption=0.6):
        nonlocal boxes
        real_rows = [r for r in tsr_cpns if re.match(r"table row$", r["label"])]
        candidates = [r for r in tsr_cpns if re.match(r".* (row|header)", r["label"])]
        filtered = []
        for r in candidates:
            if re.match(r"table row$", r["label"]):
                filtered.append(r)
                continue
            covered = any(
                Recognizer.overlapped_area(r, rr) > 0.5 or Recognizer.overlapped_area(rr, r) > 0.5
                for rr in real_rows
            )
            if not covered:
                filtered.append(r)
        eles = Recognizer.sort_Y_firstly(filtered, fzy)
        eles = Recognizer.layouts_cleanup(boxes, eles, 5, ption)
        return Recognizer.sort_Y_firstly(eles, 0)

    headers = gather(r".*header$")
    rows = gather_rows()
    spans = gather(r".*spanning")

    clmns = sorted([r for r in tsr_cpns if re.match(r"table column$", r["label"])],
                    key=lambda x: x["x0"])
    clmns = Recognizer.layouts_cleanup(boxes, clmns, 5, 0.5)
    for b in boxes:
        ii = Recognizer.find_overlapped_with_threashold(b, rows, thr=0.3)
        if ii is not None:
            b["R"] = ii
            b["R_top"] = rows[ii]["top"]
            b["R_bott"] = rows[ii]["bottom"]
        ii = Recognizer.find_overlapped_with_threashold(b, headers, thr=0.3)
        if ii is not None:
            b["H_top"] = headers[ii]["top"]
            b["H_bott"] = headers[ii]["bottom"]
            b["H_left"] = headers[ii]["x0"]
            b["H_right"] = headers[ii]["x1"]
            b["H"] = ii
        ii = Recognizer.find_horizontally_tightest_fit(b, clmns)
        if ii is not None:
            b["C"] = ii
            b["C_left"] = clmns[ii]["x0"]
            b["C_right"] = clmns[ii]["x1"]
        ii = Recognizer.find_overlapped_with_threashold(b, spans, thr=0.3)
        if ii is not None:
            b["H_top"] = spans[ii]["top"]
            b["H_bott"] = spans[ii]["bottom"]
            b["H_left"] = spans[ii]["x0"]
            b["H_right"] = spans[ii]["x1"]
            b["SP"] = ii
    return TableStructureRecognizer.construct_table(boxes, markdown=True)
