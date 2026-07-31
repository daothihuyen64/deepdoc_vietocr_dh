from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

from ..ocr import OCREngine

# Sentinel for correct_page_orientation()'s `dt_boxes_0` param -- lets a
# caller distinguish "not supplied, detect it yourself" (default) from
# "here's the answer, there really are zero boxes" (explicit None).
_UNSET = object()


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
    boxes (shared detector via `ocr.detect_raw()` -- no separate model loaded).

    Returns (corrected_img, angle_degrees) -- angle=0.0 means unchanged.
    """
    img_bgr = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    dt_boxes = ocr.detect_raw(img_bgr)
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


def _batch_recognize_samples(
    entries: list[tuple[np.ndarray, np.ndarray]],
    ocr: OCREngine,
    sample_max: int,
    min_scores: int,
    max_pages_per_batch: int = 15,
) -> list[tuple[float, dict[int, tuple[str, float]]]]:
    """Batched counterpart of the old per-page sample-and-recognize step --
    samples up to `sample_max` already-detected boxes from EACH (img_bgr,
    dt_boxes) entry (one entry per PAGE), flattens up to
    `max_pages_per_batch` entries' worth of sample crops into ONE list at a
    time, recognizes them in a SINGLE FastOCR predict_batch() call per
    chunk, then regroups results back per entry via explicit owner-index
    bookkeeping (same pattern as DocumentPipeline.process_pdf()'s
    whole-document OCR batching) -- so results can never get assigned to
    the wrong entry even though every entry's crops in a chunk were
    recognized together. Chunking by PAGE COUNT (not raw crop count) keeps
    a document with many pages needing the same orientation round from
    handing every one of their samples to a single unbounded batch call
    and risking a GPU OOM.

    Returns one (mean_score, {box_index: (text, score)}) tuple per entry,
    same order as `entries`. `dt_boxes` must not be None/empty for the
    entries passed in -- callers filter those out beforehand (a candidate
    rotation with zero detected boxes scores 0.0 without ever reaching here).
    """
    recognized_per_entry: list[dict[int, tuple[str, float]]] = [{} for _ in entries]
    scores_per_entry: list[list[float]] = [[] for _ in entries]

    for chunk_start in range(0, len(entries), max_pages_per_batch):
        chunk = entries[chunk_start : chunk_start + max_pages_per_batch]

        all_crops: list[Image.Image] = []
        owner_entry: list[int] = []
        owner_box_idx: list[int] = []
        for local_i, (img_bgr, dt_boxes) in enumerate(chunk):
            entry_i = chunk_start + local_i
            n = len(dt_boxes)
            idxs = np.linspace(0, n - 1, sample_max, dtype=int) if n > sample_max else range(n)
            for i in idxs:
                i = int(i)
                crop_bgr = ocr.get_rotate_crop_image(img_bgr, dt_boxes[i])
                if crop_bgr is None or crop_bgr.size == 0:
                    continue
                all_crops.append(Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)))
                owner_entry.append(entry_i)
                owner_box_idx.append(i)

        all_results = ocr.text_recognizer[0].detector.predict_batch(all_crops) if all_crops else []

        for entry_i, box_i, (text, score) in zip(owner_entry, owner_box_idx, all_results):
            recognized_per_entry[entry_i][box_i] = (text, score)
            scores_per_entry[entry_i].append(score)

    results = []
    for entry_i in range(len(entries)):
        scores = scores_per_entry[entry_i]
        if len(scores) < min_scores:
            results.append((0.0, recognized_per_entry[entry_i]))
        else:
            results.append((float(np.mean(scores)), recognized_per_entry[entry_i]))
    return results


@dataclass
class _OrientState:
    """Per-page working state for batch_correct_page_orientation()'s
    multi-round pipeline -- `best_*` fields track the highest-scoring
    candidate tried SO FAR (starts as the 0-degree candidate), exactly
    mirroring the single-page algorithm's best_score/best_label/etc local
    variables, just carried across batched rounds instead of a single
    page's sequential loop."""

    pn: int
    img: Image.Image
    img_bgr_0: np.ndarray
    sideways: bool
    done: bool
    best_score: float = 0.0
    best_label: str = "0"
    best_img_bgr: np.ndarray = None
    best_dt_boxes: np.ndarray | None = None
    best_recognized: dict[int, tuple[str, float]] = field(default_factory=dict)


def _try_candidate_round(
    states: list[_OrientState],
    ocr: OCREngine,
    sample_max: int,
    min_scores: int,
    label: str,
    rotate_flag: int,
    max_pages_per_batch: int = 15,
) -> None:
    """Batch-detects + batch-recognizes ONE rotation candidate (180/90/270)
    for every state in `states` (already filtered to just the pages that
    still need this candidate tried this round), updating each state's
    `best_*` fields in place if this candidate scores higher -- the
    batched equivalent of the single-page algorithm's
    `if score > best_score: best_score, best_label = score, label; ...`
    step, just done for a whole group of pages' detect+recognize calls at
    once instead of one page at a time.
    """
    if not states:
        return

    rotated_bgrs = [cv2.rotate(st.img_bgr_0, rotate_flag) for st in states]
    dt_boxes_list = ocr.detect_sorted_batch(rotated_bgrs)

    # A candidate rotation with zero detected boxes scores 0.0 without ever
    # being recognized (matches the old _score_orientation_candidate()'s
    # `if dt_boxes is None: return 0.0, None, {}`) -- only states with at
    # least one detected box need a recognition call at all.
    recognize_idxs = [i for i, db in enumerate(dt_boxes_list) if db is not None]
    scores = [0.0] * len(states)
    recognized_list: list[dict[int, tuple[str, float]]] = [{} for _ in states]
    if recognize_idxs:
        entries = [(rotated_bgrs[i], dt_boxes_list[i]) for i in recognize_idxs]
        results = _batch_recognize_samples(entries, ocr, sample_max, min_scores, max_pages_per_batch)
        for i, (score, recognized) in zip(recognize_idxs, results):
            scores[i] = score
            recognized_list[i] = recognized

    for i, st in enumerate(states):
        if scores[i] > st.best_score:
            st.best_score = scores[i]
            st.best_label = label
            st.best_img_bgr = rotated_bgrs[i]
            st.best_dt_boxes = dt_boxes_list[i]
            st.best_recognized = recognized_list[i]


def batch_correct_page_orientation(
    imgs: list[Image.Image],
    dt_boxes_0_per_page: list,
    ocr: OCREngine,
    score_threshold: float = 0.8,
    sample_max: int = 20,
    min_scores: int = 5,
    sideways_min_count: int = 4,
    sideways_min_ratio: float = 0.3,
    max_pages_per_batch: int = 15,
) -> list[tuple[Image.Image, str, float, np.ndarray | None, dict[int, tuple[str, float]]]]:
    """Batched, whole-document counterpart of correct_page_orientation() --
    produces IDENTICAL (img, label, score, dt_boxes, recognized) results
    per page as calling correct_page_orientation() once per page (SAME
    decision tree, SAME score_threshold/early-exit behavior -- see that
    function's docstring for the full algorithm description), but every
    detect/recognize call a given round needs is batched across every page
    that still needs that round, instead of looping pages one at a time:

      Round 1 (0deg):  EVERY page with detected boxes -- unconditional.
      Round 2 (180):   non-sideways pages whose 0deg score didn't clear
                       score_threshold.
      Round 2 (90):    sideways pages whose 0deg score didn't clear
                       score_threshold.
      Round 3 (270):   sideways pages whose 90deg score STILL didn't clear
                       score_threshold (270 is always the last candidate
                       tried, matching the single-page version).

    `dt_boxes_0_per_page[pn]` must be this page's ALREADY-DETECTED 0-degree
    boxes (e.g. from OCR.detect_sorted_batch() -- see
    DocumentPipeline.process_pdf()'s Phase 1b), one entry per page in
    `imgs`, `None` meaning "this page genuinely has zero detected boxes"
    (not "please detect it for me" -- unlike correct_page_orientation()'s
    own `dt_boxes_0` param, there is no lazy-detect fallback here, since
    this function's entire purpose is to be called from a pipeline that
    already has every page's 0-degree boxes on hand).

    Returns one (final_img, label, score, dt_boxes, recognized) tuple per
    input image, in the SAME ORDER as `imgs`.
    """
    states: list[_OrientState] = []
    for pn, img in enumerate(imgs):
        img_bgr_0 = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
        boxes_0 = dt_boxes_0_per_page[pn]
        sideways = _looks_sideways(boxes_0, sideways_min_count, sideways_min_ratio)
        states.append(_OrientState(
            pn=pn, img=img, img_bgr_0=img_bgr_0, sideways=sideways,
            # No boxes at all (blank/image-only page) -- nothing to score in
            # either orientation, nothing to recognize, matches the
            # single-page function's immediate `return img, "0", 0.0, boxes_0, {}`.
            done=(boxes_0 is None),
            best_img_bgr=img_bgr_0, best_dt_boxes=boxes_0,
        ))

    # ---- Round 1: 0-degree sample recognition, unconditional for every
    # page that has boxes_0.
    round1_states = [st for st in states if not st.done]
    if round1_states:
        entries = [(st.img_bgr_0, st.best_dt_boxes) for st in round1_states]
        results = _batch_recognize_samples(entries, ocr, sample_max, min_scores, max_pages_per_batch)
        for st, (score, recognized) in zip(round1_states, results):
            st.best_score = score
            st.best_recognized = recognized
            # best_label/best_img_bgr/best_dt_boxes are already "0"/img_bgr_0/boxes_0.
            st.done = score > score_threshold

    # ---- Round 2: non-sideways pages try 180; sideways pages try 90.
    need_180 = [st for st in states if not st.done and not st.sideways]
    need_90 = [st for st in states if not st.done and st.sideways]

    _try_candidate_round(need_180, ocr, sample_max, min_scores, "180", cv2.ROTATE_180, max_pages_per_batch)
    for st in need_180:
        # Non-sideways pages always terminate after the 180 attempt,
        # win or lose -- matches the single-page function's non-sideways
        # branch, which never tries anything past 180.
        st.done = True

    _try_candidate_round(need_90, ocr, sample_max, min_scores, "90", cv2.ROTATE_90_COUNTERCLOCKWISE, max_pages_per_batch)
    for st in need_90:
        st.done = st.best_score > score_threshold

    # ---- Round 3: sideways pages whose 90-degree attempt still didn't
    # clear score_threshold try 270 (always the last candidate).
    need_270 = [st for st in states if not st.done]
    _try_candidate_round(need_270, ocr, sample_max, min_scores, "270", cv2.ROTATE_90_CLOCKWISE, max_pages_per_batch)
    for st in need_270:
        st.done = True

    results_out = []
    for st in states:
        if st.best_label == "0":
            results_out.append((st.img, "0", st.best_score, st.best_dt_boxes, st.best_recognized))
        else:
            final_img = Image.fromarray(cv2.cvtColor(st.best_img_bgr, cv2.COLOR_BGR2RGB))
            results_out.append((final_img, st.best_label, st.best_score, st.best_dt_boxes, st.best_recognized))
    return results_out


def correct_page_orientation(
    img: Image.Image,
    ocr: OCREngine,
    score_threshold: float = 0.8,
    sample_max: int = 20,
    min_scores: int = 5,
    sideways_min_count: int = 4,
    sideways_min_ratio: float = 0.3,
    dt_boxes_0=_UNSET,
) -> tuple[Image.Image, str, float, np.ndarray | None, dict[int, tuple[str, float]]]:
    """Single-page convenience wrapper over batch_correct_page_orientation()
    -- callers processing many pages should call that directly (with
    dt_boxes_0_per_page for every page) to batch every round's
    detect/recognize calls across the WHOLE document instead of one page
    at a time (see DocumentPipeline.process_pdf()'s Phase 1c).

    `dt_boxes_0`, when given, is used AS-IS instead of detecting here --
    lets a caller reuse an already-detected 0-degree candidate instead of
    paying for a separate detect call. Must be `ocr.sorted_boxes()`-sorted
    already, exactly what `ocr.detect_sorted(img_bgr_0)` (the default when
    omitted) itself returns -- `_UNSET` sentinel default (not None) so a
    caller can still explicitly pass `dt_boxes_0=None` to mean "there
    really are zero boxes" without that being confused with "not
    supplied, detect it yourself".

    If the page doesn't geometrically look plausibly sideways (see
    `_looks_sideways` -- more than `sideways_min_count` AND more than
    `sideways_min_ratio` of its boxes taller than wide), 90/270 are ruled
    out without ever recognizing/rotating for them -- geometry alone can't
    tell 90 from 270 (needs real content), but it CAN cheaply rule out
    "rotated a quarter turn" from box shapes alone. 180 is a different
    story: a page rotated 180 degrees still has ordinary wide-not-tall
    text-line boxes (rotating 180 doesn't change a box's aspect ratio,
    only which way the characters inside it face), so this geometry gate
    can never catch it -- 0 always gets sample-recognized for real, and
    180 gets tried too whenever 0 doesn't read confidently, comparing the
    two scores to pick a winner.

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
    if dt_boxes_0 is _UNSET:
        img_bgr_0 = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
        dt_boxes_0 = ocr.detect_sorted(img_bgr_0)
    return batch_correct_page_orientation(
        [img], [dt_boxes_0], ocr,
        score_threshold=score_threshold, sample_max=sample_max, min_scores=min_scores,
        sideways_min_count=sideways_min_count, sideways_min_ratio=sideways_min_ratio,
    )[0]
