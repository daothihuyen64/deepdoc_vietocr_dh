from .types import OCRBox, PageBlock


def overlap_ratio(text_bbox: list[float], layout_bbox: list[float]) -> float:
    ax0, ay0, ax1, ay1 = text_bbox
    bx0, by0, bx1, by1 = layout_bbox
    inter = max(0., min(ax1, bx1) - max(ax0, bx0)) * max(0., min(ay1, by1) - max(ay0, by0))
    area = max(1., (ax1 - ax0) * (ay1 - ay0))
    return inter / area


def block_area(bbox: list[float]) -> float:
    return max(0., bbox[2] - bbox[0]) * max(0., bbox[3] - bbox[1])


def best_match_block(
    text_bbox: list[float],
    layout_blocks: list[PageBlock],
    skip_types: frozenset[str],
    overlap_threshold: float,
) -> PageBlock | None:
    best_non_skip, best_non_skip_s, best_non_skip_area = None, overlap_threshold, None
    best_skip, best_skip_s, best_skip_area = None, overlap_threshold, None

    for b in layout_blocks:
        s = overlap_ratio(text_bbox, b["bbox"])
        if s <= overlap_threshold:
            continue
        area = block_area(b["bbox"])
        is_skip = b["type"].lower() in skip_types

        if is_skip:
            if best_skip is None or s > best_skip_s + 1e-6:
                best_skip, best_skip_s, best_skip_area = b, s, area
            elif abs(s - best_skip_s) <= 1e-6 and area < best_skip_area:
                best_skip, best_skip_s, best_skip_area = b, s, area
        else:
            if best_non_skip is None or s > best_non_skip_s + 1e-6:
                best_non_skip, best_non_skip_s, best_non_skip_area = b, s, area
            elif abs(s - best_non_skip_s) <= 1e-6 and area < best_non_skip_area:
                best_non_skip, best_non_skip_s, best_non_skip_area = b, s, area

    # Non-skip always wins if any candidate clears the threshold, even if its
    # overlap is lower than the best skip-type candidate.
    return best_non_skip if best_non_skip is not None else best_skip


def dedup_nested_blocks(blocks: list[PageBlock], containment_thr: float = 0.98) -> list[PageBlock]:
    """If a block CONTAINS another block (same content_type) almost entirely,
    drop the LARGER (container) block -- this is usually the layout model
    detecting one redundant region that wraps several already-tighter
    detections. Any OCR box left outside all remaining blocks becomes its
    own block via unmatched_to_blocks() later -- no content is lost."""
    remove_ids = set()
    for i, a in enumerate(blocks):
        for j, b in enumerate(blocks):
            if i == j or a["content_type"] != b["content_type"]:
                continue
            if overlap_ratio(b["bbox"], a["bbox"]) >= containment_thr:
                remove_ids.add(id(a))
                break
    return [b for b in blocks if id(b) not in remove_ids]


def map_text_to_blocks(
    ocr_boxes: list[OCRBox],
    layout_blocks: list[PageBlock],
    skip_types: frozenset[str],
    overlap_threshold: float,
) -> list[OCRBox]:
    """First mapping pass for ALL ocr_boxes into layout_blocks (resets
    text_items on every block first). Returns the OCR boxes that matched no
    layout block -- likely regions the layout model missed (the OCR detector
    still found text there, but no layout bbox covers it)."""
    for b in layout_blocks:
        b["text_items"] = []
    unmatched = []
    for o in ocr_boxes:
        best = best_match_block(o["bbox"], layout_blocks, skip_types, overlap_threshold)
        if best is not None:
            best["text_items"].append(o)
        else:
            unmatched.append(o)
    return unmatched


def reassign_ocr_boxes(
    ocr_boxes: list[OCRBox],
    layout_blocks: list[PageBlock],
    skip_types: frozenset[str],
    overlap_threshold: float,
) -> list[OCRBox]:
    """Re-maps a SUBSET of ocr_boxes (e.g. boxes freed after dropping a
    'figure' block that absorbed too many OCR boxes) into the REMAINING
    layout_blocks, using the exact same overlap + tie-break rule as
    map_text_to_blocks() above.

    UNLIKE map_text_to_blocks(): this does NOT reset text_items on
    layout_blocks -- it only appends, so it doesn't lose the correct mapping
    already established for other blocks (text/table/...)."""
    unmatched = []
    for o in ocr_boxes:
        best = best_match_block(o["bbox"], layout_blocks, skip_types, overlap_threshold)
        if best is not None:
            best["text_items"].append(o)
        else:
            unmatched.append(o)
    return unmatched


def unmatched_to_blocks(unmatched: list[OCRBox]) -> list[PageBlock]:
    """Turns each unmatched OCR box into its own independent block, type =
    'text' like a normal block (no separate type) -- no clustering, each box
    = 1 block, bbox = that OCR box's exact bbox. Goes through the same
    content-building / sort_reading_order steps as any other 'text' block,
    no special-casing needed downstream.

    score = 0.0 since this isn't a layout model confidence (the layout model
    never detected this region at all)."""
    blocks: list[PageBlock] = []
    for o in unmatched:
        blocks.append({
            "type": "text",
            "bbox": list(o["bbox"]),
            "score": 0.0,
            "content_type": "text",
            "content": None,
            "text_items": [o],
        })
    return blocks
