import math

from .types import PageBlock


def is_double_page(img_width: float, img_height: float, single_page_ratio_max: float = 0.85) -> bool:
    """True if the page is unusually wide relative to a single sheet of
    paper -> likely an image of 2 sheets scanned side by side (back of the
    previous sheet + front of the next). Uses the width/height ratio (not
    pixel gaps) so it isn't thrown off by page-border watermarks/patterns."""
    return (img_width / img_height) > single_page_ratio_max


def _quad_geom(b: dict, anchor_min_width: float = 100):
    """Returns (cx, cy, w, h, theta) from the raw 'quad' (4 points, NOT yet
    rotated) if present. theta = None if the box has no quad or is too
    narrow to measure a reliable angle (short boxes get their corners
    rounded toward 0 by the detector, so a measured angle wouldn't be
    trustworthy). No 'quad' (e.g. a layout block in sort_reading_order, or a
    box built manually elsewhere) -> derive from 'bbox'/'bbox_row', theta is
    always None."""
    q = b.get("quad")
    if q:
        p0, p1, p2, p3 = q
        cx = sum(p[0] for p in q) / 4
        cy = sum(p[1] for p in q) / 4
        w = p1[0] - p0[0]
        h = (math.hypot(p3[0] - p0[0], p3[1] - p0[1]) + math.hypot(p2[0] - p1[0], p2[1] - p1[1])) / 2
        theta = None
        if w >= anchor_min_width:
            t_top = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
            t_bot = math.atan2(p2[1] - p3[1], p2[0] - p3[0])
            theta = (t_top + t_bot) / 2
        return cx, cy, w, h, theta
    x0, y0, x1, y1 = b.get("bbox_row", b["bbox"])
    return (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0, None


def _geo(b: dict) -> list[float]:
    """Axis-aligned bbox used for the fallback path (no reliable
    quad/theta) -- prefers 'bbox_row' (already deskewed by the page's
    overall angle) if present, falls back to the raw 'bbox'."""
    return b.get("bbox_row", b["bbox"])


def _tb_rows_fallback(lst: list[dict], overlap_min: float = 0.5) -> list[list[dict]]:
    """Groups into rows via axis-aligned y-overlap (best-fit, fixed anchor)
    -- used for boxes that no local anchor picked up (e.g. every box in a
    block is too short to self-measure an angle, or called for LAYOUT
    blocks in sort_reading_order, which have no 'quad')."""
    lst_sorted = sorted(lst, key=lambda b: _geo(b)[1])
    rows = []
    for item in lst_sorted:
        y0, y1 = _geo(item)[1], _geo(item)[3]
        best_row, best_ratio = None, overlap_min
        for row in rows:
            ry0, ry1 = row["anchor_y0"], row["anchor_y1"]
            inter = max(0., min(y1, ry1) - max(y0, ry0))
            min_h = min(y1 - y0, ry1 - ry0)
            if min_h > 0:
                ratio = inter / min_h
                if ratio >= overlap_min and ratio > best_ratio:
                    best_ratio, best_row = ratio, row
        if best_row is not None:
            best_row["items"].append(item)
        else:
            rows.append({"anchor_y0": y0, "anchor_y1": y1, "items": [item]})
    rows.sort(key=lambda r: r["anchor_y0"])
    return [sorted(r["items"], key=lambda b: _geo(b)[0]) for r in rows]


def tb_rows(
    lst: list[dict],
    overlap_min: float = 0.5,
    band_mult: float = 1.0,
    anchor_min_width: float = 100,
) -> list[list[dict]]:
    if not lst:
        return []

    infos = [(_quad_geom(b, anchor_min_width), b) for b in lst]
    infos.sort(key=lambda x: -x[0][2])  # widest/most-reliable boxes considered as anchors first

    n = len(infos)
    assigned = [False] * n
    lines = []  # list of (cy_anchor, [items left->right])

    for i in range(n):
        if assigned[i]:
            continue
        cx, cy, w, h, theta = infos[i][0]
        if theta is None:
            continue  # not reliable enough to be an anchor, leave for the fallback
        c, s = math.cos(theta), math.sin(theta)
        half_h = max(h, 1.0) / 2
        picked = []
        for j in range(n):
            if assigned[j]:
                continue
            cx2, cy2, *_ = infos[j][0]
            dx, dy = cx2 - cx, cy2 - cy
            perp = -dx * s + dy * c  # perpendicular distance from the anchor's skew direction
            along = dx * c + dy * s  # position along that direction (for left->right sort)
            if abs(perp) <= half_h * band_mult:
                picked.append((along, infos[j][1]))
                assigned[j] = True
        picked.sort(key=lambda x: x[0])
        lines.append((cy, [b for _, b in picked]))

    leftover = [infos[i][1] for i in range(n) if not assigned[i]]
    if leftover:
        for row in _tb_rows_fallback(leftover, overlap_min):
            fy = sum(_geo(b)[1] for b in row) / len(row)
            lines.append((fy, row))

    lines.sort(key=lambda x: x[0])
    return [items for _, items in lines]


def _tb(
    lst: list[dict],
    overlap_min: float = 0.5,
    band_mult: float = 1.0,
    anchor_min_width: float = 100,
) -> list[dict]:
    out = []
    for row in tb_rows(lst, overlap_min, band_mult, anchor_min_width):
        out.extend(row)
    return out


def split_side(bbox: list[float], split_x: float, min_spanning_ratio: float = 0.15) -> str:
    """Determines whether a block belongs to the left / right column, or is
    'spanning' (crosses the seam between 2 pages) based on the RATIO of the
    bbox width on each side of split_x. Only 'spanning' when BOTH sides
    carry a meaningful share (>= min_spanning_ratio) -- avoids mistaking a
    long line of text from one side that just slightly overflows the seam
    for something genuinely straddling both sheets."""
    x0, y0, x1, y1 = bbox
    width = max(1e-6, x1 - x0)

    left_w = max(0., min(x1, split_x) - x0)
    right_w = max(0., x1 - max(x0, split_x))

    left_ratio = left_w / width
    right_ratio = right_w / width

    if left_ratio >= min_spanning_ratio and right_ratio >= min_spanning_ratio:
        return "spanning"
    return "left" if left_ratio >= right_ratio else "right"


def _merge_by_y0(ordered: list[PageBlock], extra: list[PageBlock]) -> list[PageBlock]:
    """Splices `extra` (sorted by bbox y0) into `ordered` at the position
    matching its own y0, without disturbing `ordered`'s existing relative
    order."""
    if not extra:
        return ordered
    result, ei = [], 0
    for b in ordered:
        while ei < len(extra) and extra[ei]["bbox"][1] <= b["bbox"][1]:
            result.append(extra[ei])
            ei += 1
        result.append(b)
    result.extend(extra[ei:])
    return result


def sort_reading_order(
    blocks: list[PageBlock],
    img_width: float,
    img_height: float,
    overlap_min: float = 0.5,
    band_mult: float = 1.0,
    anchor_min_width: float = 100,
    single_page_ratio_max: float = 0.85,
    spanning_min_ratio: float = 0.15,
) -> list[PageBlock]:
    """Orders blocks for reading on a single page.

    'skip' blocks (image/figure/formula/seal/... -- never have content, see
    label_schema.skip_types) are pulled out before row-grouping and spliced
    back in by their own y0 afterward: a page-spanning image/background
    block would otherwise Y-overlap with nearly every other block and drag
    them all into a single reading-order "row", collapsing the whole page's
    order down to a left-to-right sort by x0 alone.
    """
    skip_blocks = [b for b in blocks if b.get("content_type") == "skip"]
    orderable = [b for b in blocks if b.get("content_type") != "skip"]

    if not is_double_page(img_width, img_height, single_page_ratio_max):
        ordered = _tb(orderable, overlap_min, band_mult, anchor_min_width)
    else:
        split_x = img_width / 2
        left, right, spanning = [], [], []
        for b in orderable:
            side = split_side(b["bbox"], split_x, spanning_min_ratio)
            if side == "spanning":
                spanning.append(b)
            elif side == "left":
                left.append(b)
            else:
                right.append(b)

        col_ordered = _tb(left, overlap_min, band_mult, anchor_min_width) + _tb(right, overlap_min, band_mult, anchor_min_width)
        spanning_sorted = sorted(spanning, key=lambda b: b["bbox"][1])
        ordered = _merge_by_y0(col_ordered, spanning_sorted)

    skip_sorted = sorted(skip_blocks, key=lambda b: b["bbox"][1])
    return _merge_by_y0(ordered, skip_sorted)
