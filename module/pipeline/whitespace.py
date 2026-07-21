import cv2
import numpy as np
from PIL import Image


def crop_whitespace_before_layout(
    pil_img: Image.Image,
    threshold: int = 200,
    dilate_kernel: tuple[int, int] = (15, 15),
    dilate_iter: int = 3,
    margin: int = 15,
    open_kernel: tuple[int, int] = (3, 3),
    min_contour_area_ratio: float = 0.0008,
    max_content_area_ratio: float = 0.5,
) -> Image.Image:
    """Crops the whitespace border around a scanned page's content, using the
    union bbox of its (denoised) contours. Ported from
    test_crop_whitespace.ipynb after parameter tuning on sample scans.

    Only crops if the content bbox covers <= max_content_area_ratio of the
    page. PP-DocLayout was trained on full pages with natural margins --
    text-dense pages (content already covering most of the page) get their
    aspect ratio distorted by cropping for no benefit, which measurably hurt
    layout detection. The crop is only worth it for sparse-content pages
    (e.g. 2 small ID-card scans on an otherwise blank A4 sheet), where the
    real content would otherwise be too small relative to the frame for the
    layout model to pick up reliably."""
    img = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY_INV)

    # Remove small scan-dust specks before merging regions together.
    open_k = cv2.getStructuringElement(cv2.MORPH_RECT, open_kernel)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, open_k)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, dilate_kernel)
    dilated = cv2.dilate(opened, kernel, iterations=dilate_iter)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return pil_img

    min_area = min_contour_area_ratio * W * H
    rects = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= min_area]
    if not rects:  # every contour got filtered out as noise -- keep them all rather than crop to nothing
        rects = [cv2.boundingRect(c) for c in contours]

    x0 = min(r[0] for r in rects)
    y0 = min(r[1] for r in rects)
    x1 = max(r[0] + r[2] for r in rects)
    y1 = max(r[1] + r[3] for r in rects)

    x0 = max(0, x0 - margin)
    y0 = max(0, y0 - margin)
    x1 = min(W, x1 + margin)
    y1 = min(H, y1 + margin)

    content_area_ratio = ((x1 - x0) * (y1 - y0)) / (W * H)
    if content_area_ratio > max_content_area_ratio:
        return pil_img

    return pil_img.crop((x0, y0, x1, y1))
