import cv2
import numpy as np
from PIL import Image

from ..ocr import OCREngine


# ── small-tilt deskew ─────────────────────────────────────────────────────────


def _top_edge_angle(box) -> float:
    """Angle (degrees) of the TL->TR edge of a quad box [TL, TR, BR, BL]."""
    pts = np.asarray(box, dtype=np.float32)
    dx = float(pts[1, 0] - pts[0, 0])
    dy = float(pts[1, 1] - pts[0, 1])
    return float(np.degrees(np.arctan2(dy, dx)))


def deskew_page(
    img: Image.Image,
    ocr: OCREngine,
    min_boxes: int = 5,
    angle_threshold: float = 0.5,
    max_angle: float = 10.0,
) -> tuple[Image.Image, float]:
    """Corrects small page tilt from the median angle of detected text-line
    boxes (shared `ocr.text_detector[0]` -- no separate model loaded).

    Returns (corrected_img, angle_degrees) -- angle=0.0 means unchanged.
    """
    img_bgr = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    dt_boxes, _ = ocr.text_detector[0](img_bgr)
    if dt_boxes is None or len(dt_boxes) < min_boxes:
        return img, 0.0

    angles = [a for a in (_top_edge_angle(box) for box in dt_boxes) if abs(a) <= max_angle]
    if len(angles) < min_boxes:
        return img, 0.0

    angle = float(np.median(angles))
    if abs(angle) < angle_threshold:
        return img, 0.0

    w, h = img.size
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    cos_a, sin_a = abs(M[0, 0]), abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    M[0, 2] += (new_w - w) / 2.0
    M[1, 2] += (new_h - h) / 2.0

    rotated_bgr = cv2.warpAffine(
        img_bgr, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )
    return Image.fromarray(cv2.cvtColor(rotated_bgr, cv2.COLOR_BGR2RGB)), angle


# ── 0/90/270 orientation correction ───────────────────────────────────────────


def _detect_boxes(img_bgr: np.ndarray, ocr: OCREngine) -> np.ndarray | None:
    """Detection only -- no recognition. Used both for the geometry gate
    (which never needs text) and as the first half of scoring a candidate."""
    dt_boxes, _ = ocr.text_detector[0](img_bgr)
    if dt_boxes is None or len(dt_boxes) == 0:
        return None
    return ocr.sorted_boxes(dt_boxes)


def _sample_recognize(
    img_bgr: np.ndarray,
    dt_boxes: np.ndarray,
    ocr: OCREngine,
    sample_max: int,
    min_scores: int,
) -> tuple[float, dict[int, tuple[str, float]]]:
    """Recognizes an evenly spaced SAMPLE of up to `sample_max` already-detected
    boxes (real return_prob confidence, not the hardcoded-1.0 path) and scores
    the candidate as their mean prob.

    Returns (mean_score, {box_index: (text, score)}) -- the recognized sample
    is handed back so the caller can reuse it for the winning candidate
    instead of re-recognizing it later.
    """
    n = len(dt_boxes)
    idxs = np.linspace(0, n - 1, sample_max, dtype=int) if n > sample_max else range(n)

    recognized: dict[int, tuple[str, float]] = {}
    scores = []
    for i in idxs:
        i = int(i)
        crop_bgr = ocr.get_rotate_crop_image(img_bgr, dt_boxes[i])
        if crop_bgr is None or crop_bgr.size == 0:
            continue
        crop_pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        text, score = ocr.text_recognizer[0].detector.predict(crop_pil, return_prob=True)
        recognized[i] = (text, score)
        scores.append(score)

    if len(scores) < min_scores:
        return 0.0, recognized
    return float(np.mean(scores)), recognized


def _score_orientation_candidate(
    img_bgr: np.ndarray,
    ocr: OCREngine,
    sample_max: int,
    min_scores: int,
) -> tuple[float, np.ndarray | None, dict[int, tuple[str, float]]]:
    """Detect + sample-recognize a candidate rotation in one call. Returns
    (mean_score, sorted_dt_boxes, {box_index: (text, score)})."""
    dt_boxes = _detect_boxes(img_bgr, ocr)
    if dt_boxes is None:
        return 0.0, None, {}
    score, recognized = _sample_recognize(img_bgr, dt_boxes, ocr, sample_max, min_scores)
    return score, dt_boxes, recognized


def _box_wh(box) -> tuple[float, float]:
    """Width/height of a quad box [TL, TR, BR, BL], in the box's own local axes."""
    pts = np.asarray(box, dtype=np.float32)
    w = float(np.linalg.norm(pts[1] - pts[0]))
    h = float(np.linalg.norm(pts[3] - pts[0]))
    return w, h


def _looks_sideways(dt_boxes, min_count: int, min_ratio: float) -> bool:
    """True if enough detected boxes are TALLER than they are WIDE to suspect
    the page might actually be rotated 90/270 -- gates whether 90/270 ever get
    OCR-scored at all. Normal horizontal text lines are wide, not tall, so an
    already-upright page essentially never clears this gate, which is what
    stops it from occasionally losing to a garbled 90/270 OCR read just from
    model noise (the original bug this gate fixes)."""
    if dt_boxes is None or len(dt_boxes) == 0:
        return False
    tall = sum(1 for box in dt_boxes if _box_wh(box)[1] > _box_wh(box)[0])
    return tall > min_count and (tall / len(dt_boxes)) > min_ratio


def correct_page_orientation(
    img: Image.Image,
    ocr: OCREngine,
    score_threshold: float = 0.8,
    sample_max: int = 20,
    min_scores: int = 5,
    sideways_min_count: int = 4,
    sideways_min_ratio: float = 0.3,
) -> tuple[Image.Image, str, float, np.ndarray | None, dict[int, tuple[str, float]]]:
    """Detects 0 deg's boxes first -- detection ONLY, no recognition yet.

    If they don't geometrically look plausibly sideways (see
    `_looks_sideways` -- more than `sideways_min_count` AND more than
    `sideways_min_ratio` of them taller than wide), 90/270 are ruled out
    without ever recognizing/rotating for them -- geometry alone can't tell
    90 from 270 (needs real content), but it CAN cheaply rule out "rotated a
    quarter turn" from box shapes alone. 180 is a different story: a page
    rotated 180 degrees still has ordinary wide-not-tall text-line boxes
    (rotating 180 doesn't change a box's aspect ratio, only which way the
    characters inside it face), so this geometry gate can never catch it --
    0 always gets sample-recognized for real here, and 180 gets tried too
    whenever 0 doesn't read confidently, comparing the two scores to pick
    a winner.

    When the gate DOES trip (looks plausibly quarter-turned), 0 gets
    sample-recognized (real score) and the OCR-confidence early-exit run:
    90 deg -- stop immediately if it clears `score_threshold` -- otherwise
    270 deg (always the last candidate), then take whichever of the (up to)
    3 scores tried is highest. 180 is not tried on this path -- a page
    that's genuinely quarter-turned is not also upside down.

    Returns (final_img, label, score, dt_boxes, recognized) -- `dt_boxes`/
    `recognized` are the WINNING candidate's already-computed detection
    (all boxes) and recognition (sampled subset) -- reusable by
    run_ocr_page() instead of redoing that work on the image this function
    hands back.
    """
    img_bgr_0 = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    dt_boxes_0 = _detect_boxes(img_bgr_0, ocr)

    if not _looks_sideways(dt_boxes_0, sideways_min_count, sideways_min_ratio):
        score_0, recognized_0 = _sample_recognize(img_bgr_0, dt_boxes_0, ocr, sample_max, min_scores)
        if score_0 > score_threshold:
            return img, "0", score_0, dt_boxes_0, recognized_0

        img_bgr_180 = cv2.rotate(img_bgr_0, cv2.ROTATE_180)
        score_180, dt_boxes_180, recognized_180 = _score_orientation_candidate(
            img_bgr_180, ocr, sample_max, min_scores
        )
        if score_180 > score_0:
            final_img = Image.fromarray(cv2.cvtColor(img_bgr_180, cv2.COLOR_BGR2RGB))
            return final_img, "180", score_180, dt_boxes_180, recognized_180
        return img, "0", score_0, dt_boxes_0, recognized_0

    score_0, recognized_0 = _sample_recognize(img_bgr_0, dt_boxes_0, ocr, sample_max, min_scores)
    best_score, best_label = score_0, "0"
    best_img_bgr, best_dt_boxes, best_recognized = img_bgr_0, dt_boxes_0, recognized_0

    if best_score <= score_threshold:
        for label, rotate_flag in (("90", cv2.ROTATE_90_COUNTERCLOCKWISE), ("270", cv2.ROTATE_90_CLOCKWISE)):
            img_bgr = cv2.rotate(img_bgr_0, rotate_flag)
            score, dt_boxes, recognized = _score_orientation_candidate(img_bgr, ocr, sample_max, min_scores)
            if score > best_score:
                best_score, best_label = score, label
                best_img_bgr, best_dt_boxes, best_recognized = img_bgr, dt_boxes, recognized
            if score > score_threshold:
                break

    if best_label == "0":
        return img, "0", best_score, best_dt_boxes, best_recognized
    final_img = Image.fromarray(cv2.cvtColor(best_img_bgr, cv2.COLOR_BGR2RGB))
    return final_img, best_label, best_score, best_dt_boxes, best_recognized
