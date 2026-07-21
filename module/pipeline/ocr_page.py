import copy
import math
import os

import cv2
import numpy as np
from PIL import Image

from ..ocr import OCREngine
from .types import OCRBox


def run_ocr_page(img_pil: Image.Image, ocr: OCREngine, crop_debug_dir: str | None = None) -> list[OCRBox]:
    """Runs detection + recognition ONCE for the whole page.

    Returns ocr_boxes with 'bbox' (raw), 'bbox_row' (virtual bbox counter-
    rotated by the page's estimated skew angle -- used as a row-grouping
    fallback in reading_order.py) and 'quad' (raw 4-point box, used for the
    per-box local-anchor row grouping). The page image itself is never
    warped -- only these derived coordinates are.
    """
    img_bgr = cv2.cvtColor(np.array(img_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    ori_im = img_bgr.copy()

    dt_boxes, _ = ocr.text_detector[0](img_bgr)
    if dt_boxes is None or len(dt_boxes) == 0:
        return []
    dt_boxes = ocr.sorted_boxes(dt_boxes)

    img_crop_list = []
    for bno in range(len(dt_boxes)):
        tmp_box = copy.deepcopy(dt_boxes[bno])
        img_crop_list.append(ocr.get_rotate_crop_image(ori_im, tmp_box))

    # Dumps the EXACT crops handed to the recognizer below (post crop
    # padding/rotate-90 decision) -- for debugging what VietOCR actually
    # sees per box, not a reconstruction of it.
    if crop_debug_dir:
        os.makedirs(crop_debug_dir, exist_ok=True)
        for bno, img_crop in enumerate(img_crop_list):
            cv2.imwrite(os.path.join(crop_debug_dir, f"{bno}.png"), img_crop)

    rec_res, _ = ocr.text_recognizer[0](img_crop_list)

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
