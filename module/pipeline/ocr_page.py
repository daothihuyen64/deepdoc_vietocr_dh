import copy
import math
import os
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

from ..ocr import OCREngine
from .types import OCRBox


@dataclass
class PageOcrPrep:
    """Per-page detection state, held between prepare_ocr_page() (detect +
    crop, no VietOCR call) and finish_ocr_page() (assembles the final
    OCRBox list) -- split apart so process_pdf() can recognize every
    page's crops in ONE VietOCR call spanning the WHOLE document, instead
    of one call per page (see DocumentPipeline.process_pdf()'s Phase 3a/3b).
    """

    dt_boxes: np.ndarray | None
    prerecognized: dict[int, tuple[str, float]] = field(default_factory=dict)
    # Which positions in dt_boxes still need FRESH recognition (i.e. were
    # NOT already in `prerecognized`) -- same order as the crops this page
    # contributed to process_pdf()'s whole-document crop list, so
    # finish_ocr_page() can zip them back together 1:1.
    crop_indices: list[int] = field(default_factory=list)


def prepare_ocr_page(
    img_pil: Image.Image,
    ocr: OCREngine,
    crop_debug_dir: str | None = None,
    dt_boxes: np.ndarray | None = None,
    prerecognized: dict[int, tuple[str, float]] | None = None,
) -> tuple[PageOcrPrep, list[np.ndarray]]:
    """Detects (or reuses) this page's boxes and crops every box that still
    needs fresh recognition -- does NOT call the recognizer itself.

    `dt_boxes`/`prerecognized`, when given, come from page_orientation.py having
    already detected (and, for a sampled subset, recognized) boxes on this EXACT
    image while scoring 0/90/270 candidates -- skips re-running the detector
    entirely, and only recognizes whichever boxes weren't in that sample.

    Returns (prep, crops) -- crops is a flat list of the boxes needing
    fresh recognition (same order as prep.crop_indices). Pass `crops` into
    process_pdf()'s whole-document crop list, and pass `prep` + that
    page's slice of the resulting recognition output into finish_ocr_page().
    """
    img_bgr = cv2.cvtColor(np.array(img_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    ori_im = img_bgr.copy()

    if dt_boxes is None:
        dt_boxes = ocr.detect_sorted(img_bgr)

    prerecognized = prerecognized or {}
    crop_indices: list[int] = []
    crops: list[np.ndarray] = []
    has_boxes = dt_boxes is not None and len(dt_boxes) > 0
    if has_boxes:
        for bno in range(len(dt_boxes)):
            if bno in prerecognized:
                continue
            tmp_box = copy.deepcopy(dt_boxes[bno])
            crops.append(ocr.get_rotate_crop_image(ori_im, tmp_box))
            crop_indices.append(bno)

        # Dumps the EXACT crops that need fresh recognition (post crop
        # padding/rotate-90 decision) -- for debugging what VietOCR
        # actually sees per box, not a reconstruction of it. Boxes reused
        # from page_orientation.py's sample aren't re-cropped, so they
        # won't appear here.
        if crop_debug_dir:
            os.makedirs(crop_debug_dir, exist_ok=True)
            for bno, img_crop in zip(crop_indices, crops):
                cv2.imwrite(os.path.join(crop_debug_dir, f"{bno}.png"), img_crop)

    return PageOcrPrep(dt_boxes=dt_boxes, prerecognized=prerecognized, crop_indices=crop_indices), crops


def finish_ocr_page(prep: PageOcrPrep, fresh_rec_res: list[tuple[str, float]], ocr: OCREngine) -> list[OCRBox]:
    """Assembles the final OCRBox list for one page, given this page's own
    slice of recognition results (`fresh_rec_res`, in the SAME ORDER as
    prep.crop_indices -- see prepare_ocr_page).

    Returns ocr_boxes with 'bbox' (raw), 'bbox_row' (virtual bbox counter-
    rotated by the page's estimated skew angle -- used as a row-grouping
    fallback in reading_order.py) and 'quad' (raw 4-point box, used for the
    per-box local-anchor row grouping). The page image itself is never
    warped -- only these derived coordinates are.
    """
    dt_boxes = prep.dt_boxes
    if dt_boxes is None or len(dt_boxes) == 0:
        return []

    rec_res: list[tuple[str, float] | None] = [None] * len(dt_boxes)
    for bno, res in prep.prerecognized.items():
        rec_res[bno] = res
    for bno, res in zip(prep.crop_indices, fresh_rec_res):
        rec_res[bno] = res

    def _skew_of_quad(box, min_width=100):
        p0, p1, p2, p3 = box
        w = p1[0] - p0[0]
        if w < min_width:
            return None
        t_top = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
        t_bot = math.atan2(p2[1] - p3[1], p2[0] - p3[0])
        return (t_top + t_bot) / 2

    def _is_reliable_skew_sample(text):
        # Only trust boxes containing at least 1 letter -- excludes pure
        # numeric coordinates/measurements which are unreliable for
        # estimating the page's overall skew angle.
        return any(ch.isalpha() for ch in text)

    _thetas = [
        _skew_of_quad(box)
        for box, (text, score) in zip(dt_boxes, rec_res)
        if _is_reliable_skew_sample(text)
    ]
    _thetas = [t for t in _thetas if t is not None]
    _page_theta = sorted(_thetas)[len(_thetas) // 2] if _thetas else 0.0

    if len(dt_boxes) > 0:
        _all_pts = np.concatenate([np.array(b) for b in dt_boxes], axis=0)
        _cx, _cy = float(_all_pts[:, 0].mean()), float(_all_pts[:, 1].mean())
    else:
        _cx, _cy = 0.0, 0.0

    def _rotate_pt(x, y, theta, cx, cy):
        dx, dy = x - cx, y - cy
        c, s = math.cos(-theta), math.sin(-theta)
        return cx + dx * c - dy * s, cy + dx * s + dy * c

    ocr_boxes: list[OCRBox] = []
    for box, (text, score) in zip(dt_boxes, rec_res):
        if score >= ocr.drop_score and text.strip():
            box_np = np.array(box)
            x0, y0 = float(box_np[:, 0].min()), float(box_np[:, 1].min())
            x1, y1 = float(box_np[:, 0].max()), float(box_np[:, 1].max())
            rot_pts = [_rotate_pt(float(px), float(py), _page_theta, _cx, _cy) for px, py in box_np]
            rx0 = min(p[0] for p in rot_pts)
            ry0 = min(p[1] for p in rot_pts)
            rx1 = max(p[0] for p in rot_pts)
            ry1 = max(p[1] for p in rot_pts)
            ocr_boxes.append({
                "bbox": [x0, y0, x1, y1],
                "bbox_row": [rx0, ry0, rx1, ry1],
                "quad": [[float(px), float(py)] for px, py in box_np],
                "text": text,
            })

    return ocr_boxes


def run_ocr_page(
    img_pil: Image.Image,
    ocr: OCREngine,
    crop_debug_dir: str | None = None,
    dt_boxes: np.ndarray | None = None,
    prerecognized: dict[int, tuple[str, float]] | None = None,
) -> list[OCRBox]:
    """Runs detection + recognition ONCE for a single page -- convenience
    wrapper over prepare_ocr_page()/finish_ocr_page() for callers that
    don't need to batch recognition across multiple pages (e.g.
    DocumentPipeline.process_pdf() calls the two halves directly instead,
    to recognize every page's crops in one whole-document VietOCR call).
    """
    prep, crops = prepare_ocr_page(img_pil, ocr, crop_debug_dir, dt_boxes, prerecognized)
    fresh_rec_res, _ = ocr.text_recognizer[0](crops) if crops else ([], 0.0)
    return finish_ocr_page(prep, fresh_rec_res, ocr)
