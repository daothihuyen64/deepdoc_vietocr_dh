import os

from PIL import Image, ImageDraw, ImageFont

from .types import OCRBox
from ..layout.base import LayoutBlock

_BOX_COLOR = (255, 0, 0)
_LABEL_BG = (255, 255, 0)
_LABEL_FG = (0, 0, 0)
_LEGEND_LINE_HEIGHT = 20

_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "C:/Windows/Fonts/arial.ttf",
]


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _draw_tagged_box(draw: ImageDraw.ImageDraw, bbox: list[float], label: str, font) -> None:
    x0, y0, x1, y1 = [round(v) for v in bbox]
    draw.rectangle([x0, y0, x1, y1], outline=_BOX_COLOR, width=2)
    tag_w = draw.textlength(label, font=font)
    tag_y0 = max(0, y0 - _LEGEND_LINE_HEIGHT)
    draw.rectangle([x0, tag_y0, x0 + tag_w + 4, y0], fill=_LABEL_BG)
    draw.text((x0 + 2, tag_y0), label, fill=_LABEL_FG, font=font)


def save_input_image(img: Image.Image, out_dir: str, pn: int) -> None:
    """Saves the raw page image exactly as it's about to be fed into the layout model."""
    img.convert("RGB").save(os.path.join(out_dir, f"page_{pn + 1}_input.png"))


def save_layout_debug(img: Image.Image, raw_blocks: list[LayoutBlock], out_dir: str, pn: int) -> None:
    """Draws every layout bbox with its index + predicted type."""
    canvas = img.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    font = _load_font(16)
    for i, b in enumerate(raw_blocks):
        _draw_tagged_box(draw, b["bbox"], f"{i}:{b['type']}", font)
    canvas.save(os.path.join(out_dir, f"page_{pn + 1}_layout.png"))


def save_table_crop(crop: Image.Image, out_dir: str, pn: int, tno: int) -> None:
    """Saves a table crop as fed into TSR, after deskewing."""
    crop.convert("RGB").save(os.path.join(out_dir, f"page_{pn + 1}_table_{tno}_deskewed.png"))


def save_table_ocr_debug(crop: Image.Image, raw_ocr_crop: list, out_dir: str, pn: int, tno: int) -> None:
    """Draws every OCR bbox detected inside a table crop (the same 'raw'
    result fed into table_to_markdown), tagged with its index, plus a
    legend at the bottom mapping each index to its recognized text -- for
    debugging column/row mis-assignment in table parsing."""
    base = crop.convert("RGB")
    w, h = base.size
    font = _load_font(14)

    legend_h = 10 + _LEGEND_LINE_HEIGHT * len(raw_ocr_crop)
    canvas = Image.new("RGB", (w, h + legend_h), "white")
    canvas.paste(base, (0, 0))
    draw = ImageDraw.Draw(canvas)

    boxes_texts = [(box, text) for box, (text, _score) in raw_ocr_crop]

    for i, (box, _text) in enumerate(boxes_texts):
        x0 = min(p[0] for p in box)
        y0 = min(p[1] for p in box)
        x1 = max(p[0] for p in box)
        y1 = max(p[1] for p in box)
        _draw_tagged_box(draw, [x0, y0, x1, y1], str(i), font)

    y = h + 5
    for i, (_box, text) in enumerate(boxes_texts):
        draw.text((5, y), f"{i}: {text}", fill=(0, 0, 0), font=font)
        y += _LEGEND_LINE_HEIGHT

    canvas.save(os.path.join(out_dir, f"page_{pn + 1}_table_{tno}_ocr.png"))


def save_table_backend_compare(mineru_content: str, tsr_content: str, out_dir: str, pn: int, tno: int) -> None:
    """Dumps the mineru (wireless) vs tsr content for the SAME table crop
    side by side, for manual quality comparison only -- never read back by
    the pipeline itself."""
    path = os.path.join(out_dir, f"page_{pn + 1}_table_{tno}_wireless_compare.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("## mineru (wireless)\n\n")
        f.write(mineru_content + "\n\n")
        f.write("## tsr\n\n")
        f.write(tsr_content + "\n")


def save_ocr_debug(img: Image.Image, ocr_boxes: list[OCRBox], out_dir: str, pn: int) -> None:
    """Draws every OCR bbox tagged with its index, plus a legend at the
    bottom of the image mapping each index to its recognized text."""
    base = img.convert("RGB")
    w, h = base.size
    font = _load_font(14)

    legend_h = 10 + _LEGEND_LINE_HEIGHT * len(ocr_boxes)
    canvas = Image.new("RGB", (w, h + legend_h), "white")
    canvas.paste(base, (0, 0))
    draw = ImageDraw.Draw(canvas)

    for i, b in enumerate(ocr_boxes):
        _draw_tagged_box(draw, b["bbox"], str(i), font)

    y = h + 5
    for i, b in enumerate(ocr_boxes):
        draw.text((5, y), f"{i}: {b['text']}", fill=(0, 0, 0), font=font)
        y += _LEGEND_LINE_HEIGHT

    canvas.save(os.path.join(out_dir, f"page_{pn + 1}_ocr.png"))
