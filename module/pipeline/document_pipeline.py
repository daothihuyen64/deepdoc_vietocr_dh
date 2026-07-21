import logging
import os
import time
from pathlib import Path

from PIL import Image

from ..layout import LayoutBackend, get_layout_backend
from ..ocr import OCREngine, get_ocr_engine
from ..tsr import TSRBackend, get_tsr_backend
from .config import PipelineConfig, load_pipeline_conf
from .content import build_block_content, build_json, build_markdown
from .debug import save_input_image, save_layout_debug, save_ocr_debug, save_table_crop, save_table_ocr_debug
from .loader import load_pdf_pages
from .ocr_page import run_ocr_page
from .reading_order import sort_reading_order, tb_rows
from .table import deskew_crop, raw_ocr, table_to_markdown
from .text_mapping import (
    dedup_nested_blocks,
    map_text_to_blocks,
    reassign_ocr_boxes,
    unmatched_to_blocks,
)
from .types import PageBlock
from .whitespace import crop_whitespace_before_layout

logger = logging.getLogger(__name__)


class DocumentPipeline:
    """Orchestrates layout detection, OCR, table structure recognition and
    reading-order/content assembly for a single PDF. Components are
    dependency-injected so any of them can be swapped for a different
    implementation without touching this class."""

    def __init__(self, layout: LayoutBackend, ocr: OCREngine, tsr: TSRBackend, config: PipelineConfig):
        self.layout = layout
        self.ocr = ocr
        self.tsr = tsr
        self.config = config

    def process_pdf(self, pdf_path: str, source_filename: str | None = None) -> dict:
        t_render = time.time()
        pages = load_pdf_pages(pdf_path, dpi=self.config.pdf_dpi)
        label = source_filename or os.path.basename(pdf_path)
        logger.info("Processing %s (%d pages) | pdf_render=%.2fs", label, len(pages), time.time() - t_render)

        debug_dir = None
        if self.config.debug_enabled:
            debug_dir = os.path.join(self.config.debug_dir, Path(label).stem)
            os.makedirs(debug_dir, exist_ok=True)

        pages_blocks = [self._process_page(pn, img, debug_dir) for pn, img in enumerate(pages)]

        markdown = build_markdown(pages_blocks, self.layout.label_schema)

        if debug_dir:
            with open(os.path.join(debug_dir, "output.md"), "w", encoding="utf-8") as f:
                f.write(markdown)

        return {
            "json": build_json(label, pages_blocks),
            "markdown": markdown,
        }

    def _process_page(self, pn: int, img: Image.Image, debug_dir: str | None = None) -> list[PageBlock]:
        cfg = self.config
        label_schema = self.layout.label_schema
        t0 = time.time()
        timings: dict[str, float] = {}
        t_stage = t0

        def _mark(name: str) -> None:
            nonlocal t_stage
            now = time.time()
            timings[name] = now - t_stage
            t_stage = now

        if cfg.crop_whitespace_enabled:
            orig_size = img.size
            img = crop_whitespace_before_layout(
                img,
                threshold=cfg.crop_whitespace_threshold,
                dilate_kernel=cfg.crop_whitespace_dilate_kernel,
                dilate_iter=cfg.crop_whitespace_dilate_iter,
                margin=cfg.crop_whitespace_margin,
                open_kernel=cfg.crop_whitespace_open_kernel,
                min_contour_area_ratio=cfg.crop_whitespace_min_contour_area_ratio,
                max_content_area_ratio=cfg.crop_whitespace_max_content_area_ratio,
            )
            if img.size != orig_size:
                logger.debug("Page %d: whitespace-cropped %s -> %s", pn + 1, orig_size, img.size)
        _mark("whitespace_crop")

        w, h = img.size

        if debug_dir:
            save_input_image(img, debug_dir, pn)

        raw_blocks = self.layout.detect(img, cfg.layout_threshold)
        _mark("layout_detect")
        page_crop_debug_dir = os.path.join(debug_dir, f"page_{pn + 1}_vietocr_crops") if debug_dir else None
        ocr_boxes = run_ocr_page(img, self.ocr, crop_debug_dir=page_crop_debug_dir)
        _mark("ocr_page")

        if debug_dir:
            save_layout_debug(img, raw_blocks, debug_dir, pn)
            save_ocr_debug(img, ocr_boxes, debug_dir, pn)

        blocks: list[PageBlock] = []
        tno = 0
        for b in raw_blocks:
            btype = b["type"].lower()
            if btype in label_schema.table_types:
                x0, y0, x1, y1 = map(int, b["bbox"])
                crop = img.crop((max(0, x0 - 2), max(0, y0 - 2), min(w, x1 + 2), min(h, y1 + 2)))
                crop, skew_angle = deskew_crop(crop)
                if abs(skew_angle) > 0.1:
                    logger.debug("Page %d: table deskewed by %.2f°", pn + 1, skew_angle)
                if debug_dir:
                    save_table_crop(crop, debug_dir, pn, tno)
                tno += 1
                tsr_result = self.tsr([crop], thr=cfg.tsr_threshold)
                cpns = tsr_result[0] if tsr_result else []
                crop_debug_dir = (
                    os.path.join(debug_dir, f"page_{pn + 1}_table_{tno - 1}_vietocr_crops")
                    if debug_dir else None
                )
                raw_ocr_crop = raw_ocr(crop, self.ocr, crop_debug_dir=crop_debug_dir)
                if debug_dir:
                    save_table_ocr_debug(crop, raw_ocr_crop, debug_dir, pn, tno - 1)
                content = table_to_markdown(crop, cpns, self.ocr, raw=raw_ocr_crop)
                blocks.append({**b, "content_type": "table", "content": content})
            elif btype in label_schema.skip_types:
                blocks.append({**b, "content_type": "skip", "content": None})
            else:
                blocks.append({**b, "content_type": "text", "content": None})
        _mark("tables")

        blocks = dedup_nested_blocks(blocks)
        unmatched = map_text_to_blocks(ocr_boxes, blocks, label_schema.skip_types, cfg.map_overlap_threshold)

        n_ocr_total = len(ocr_boxes)
        if n_ocr_total > 0:
            figure_to_drop = [
                b for b in blocks
                if b["type"].lower() in cfg.figure_drop_types
                and len(b.get("text_items", [])) / n_ocr_total > cfg.figure_drop_ratio
            ]
            if figure_to_drop:
                reclaim_boxes = []
                for b in figure_to_drop:
                    reclaim_boxes.extend(b["text_items"])
                    logger.debug(
                        "Page %d: dropping block '%s' bbox=%s (absorbed %d/%d = %.0f%% of OCR boxes)",
                        pn + 1, b["type"], [round(v) for v in b["bbox"]],
                        len(b["text_items"]), n_ocr_total, 100 * len(b["text_items"]) / n_ocr_total,
                    )

                drop_ids = {id(b) for b in figure_to_drop}
                blocks = [b for b in blocks if id(b) not in drop_ids]

                unmatched.extend(reassign_ocr_boxes(reclaim_boxes, blocks, label_schema.skip_types, cfg.map_overlap_threshold))

        if unmatched:
            extra_blocks = unmatched_to_blocks(unmatched)
            blocks.extend(extra_blocks)
            logger.debug("Page %d: %d OCR boxes unmatched -> %d extra blocks", pn + 1, len(unmatched), len(extra_blocks))
        _mark("mapping")

        for b in blocks:
            if b["content_type"] == "text":
                rows = tb_rows(b["text_items"], cfg.line_overlap_min, cfg.anchor_band_mult, cfg.anchor_min_width)
                line_texts = []
                for row in rows:
                    line = " ".join(t["text"] for t in row if t["text"].strip())
                    if line:
                        line_texts.append(line)
                b["content"] = build_block_content(line_texts)
        _mark("content_build")

        blocks = sort_reading_order(
            blocks, img_width=w, img_height=h,
            overlap_min=cfg.line_overlap_min,
            band_mult=cfg.anchor_band_mult,
            anchor_min_width=cfg.anchor_min_width,
            single_page_ratio_max=cfg.single_page_ratio_max,
            spanning_min_ratio=cfg.spanning_min_ratio,
        )
        _mark("reading_order")

        elapsed = time.time() - t0
        n_tbl = sum(1 for b in blocks if b["content_type"] == "table")
        n_text = sum(1 for b in blocks if b["content_type"] == "text")
        timings_str = " ".join(f"{k}={v:.2f}s" for k, v in timings.items())
        logger.info(
            "Page %d done in %.1fs: layout=%d ocr=%d table=%d text=%d | %s",
            pn + 1, elapsed, len(raw_blocks), len(ocr_boxes), n_tbl, n_text, timings_str,
        )

        return blocks


def build_pipeline(conf: dict | None = None) -> DocumentPipeline:
    """Single source of truth for constructing a DocumentPipeline -- used by
    both the CLI (full_pipeline.py) and the FastAPI server so there is no
    duplicated business logic between the two entry points."""
    conf = conf or load_pipeline_conf()
    config = PipelineConfig.from_conf(conf)
    return DocumentPipeline(
        get_layout_backend(conf),
        get_ocr_engine(conf),
        get_tsr_backend(conf),
        config,
    )
