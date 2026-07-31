import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ..layout import LayoutBackend, LayoutBlock, get_layout_backend
from ..ocr import OCREngine, get_ocr_engine
from ..table import TableProcessor, get_table_processor
from .config import PipelineConfig, load_pipeline_conf
from .content import build_block_content, build_json, build_markdown
from .debug import save_input_image, save_layout_debug, save_ocr_debug, save_table_crop
from .loader import load_pdf_pages
from .ocr_page import PageOcrPrep, finish_ocr_page, prepare_ocr_page
from .page_orientation import batch_correct_page_orientation, deskew_page
from .reading_order import sort_reading_order, tb_rows
from .table import deskew_crop
from .text_mapping import (
    dedup_nested_blocks,
    map_text_to_blocks,
    reassign_ocr_boxes,
    unmatched_to_blocks,
)
from .types import PageBlock
from .whitespace import crop_whitespace_before_layout

logger = logging.getLogger(__name__)


@dataclass
class _PrePage:
    """Per-page state after whitespace crop + deskew, before orientation
    correction -- held just long enough for process_pdf() to batch-detect
    the 0-degree candidate for every page in one call (see
    OCR.detect_sorted_batch) before running orientation correction, which
    needs that detection as its very first step but is otherwise
    per-page early-exit branching logic that doesn't itself batch cleanly."""

    img: Image.Image
    timings: dict[str, float] = field(default_factory=dict)


@dataclass
class _PreparedPage:
    """Per-page state carried from the pre-layout prep phase (whitespace
    crop, deskew, orientation) into the post-layout phase (OCR, tables,
    assembly). Layout detection itself now runs ONCE for the whole
    document, batched across all pages, in between these two phases --
    see DocumentPipeline.process_pdf()."""

    img: Image.Image
    dt_boxes_reuse: np.ndarray | None
    prerecognized_reuse: dict[int, tuple[str, float]] | None
    timings: dict[str, float] = field(default_factory=dict)


class DocumentPipeline:
    """Orchestrates layout detection, OCR, table structure recognition and
    reading-order/content assembly for a single PDF. Components are
    dependency-injected so any of them can be swapped for a different
    implementation without touching this class."""

    def __init__(self, layout: LayoutBackend, ocr: OCREngine, table_processor: TableProcessor, config: PipelineConfig):
        self.layout = layout
        self.ocr = ocr
        self.table_processor = table_processor
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

        pages_blocks = self._process_pages_batch(pages, debug_dir)

        t_build = time.time()
        markdown = build_markdown(pages_blocks, self.layout.label_schema)
        json_out = build_json(label, pages_blocks)
        logger.info("Built JSON/markdown output in %.2fs", time.time() - t_build)

        if debug_dir:
            with open(os.path.join(debug_dir, "output.md"), "w", encoding="utf-8") as f:
                f.write(markdown)

        out_dir = os.path.join(self.config.output_dir, Path(label).stem)
        os.makedirs(out_dir, exist_ok=True)
        shutil.copyfile(pdf_path, os.path.join(out_dir, label if label.lower().endswith(".pdf") else f"{label}.pdf"))
        with open(os.path.join(out_dir, "output.json"), "w", encoding="utf-8") as f:
            json.dump(json_out, f, ensure_ascii=False, indent=2)
        with open(os.path.join(out_dir, "output.md"), "w", encoding="utf-8") as f:
            f.write(markdown)

        return {
            "json": json_out,
            "markdown": markdown,
        }

    def process_pdf_group(self, items: list[tuple[str, str]]) -> list[dict]:
        """Batch-processes multiple PDFs' pages together as ONE combined
        document (concatenating every PDF's pages into a single list before
        running them through _process_pages_batch), then splits the result
        back per-source-PDF for separate json/markdown/output-dir writes --
        see server/routes.py's /pdfs endpoint, which groups uploads into
        chunks of <=200 total pages before calling this once per chunk.

        `items` is `[(pdf_path, label), ...]`, already grouped by the caller.
        A PDF that fails to render is skipped (result marked "error") without
        aborting the rest of the group. If the shared batched call itself
        fails (rare -- e.g. a crash triggered by pathological page content),
        every PDF in this group is marked "error" with the same message,
        since a shared-model failure mid-batch can't be cheaply attributed
        to one specific input.
        """
        results: list[dict] = []
        all_pages: list[Image.Image] = []
        # (label, pdf_path, start, end) -- end is exclusive, indexes into all_pages/pages_blocks.
        boundaries: list[tuple[str, str, int, int]] = []
        for pdf_path, label in items:
            try:
                pages = load_pdf_pages(pdf_path, dpi=self.config.pdf_dpi)
            except Exception as e:
                logger.exception("Failed to render %s, skipping", label)
                results.append({"file": label, "status": "error", "error": str(e)})
                continue
            start = len(all_pages)
            all_pages.extend(pages)
            boundaries.append((label, pdf_path, start, len(all_pages)))

        if not boundaries:
            return results

        try:
            pages_blocks = self._process_pages_batch(all_pages, debug_dir=None)
        except Exception as e:
            logger.exception("Batched processing failed for a %d-PDF group", len(boundaries))
            for label, _pdf_path, _start, _end in boundaries:
                results.append({"file": label, "status": "error", "error": str(e)})
            return results

        for label, pdf_path, start, end in boundaries:
            sub_blocks = pages_blocks[start:end]
            markdown = build_markdown(sub_blocks, self.layout.label_schema)
            json_out = build_json(label, sub_blocks)
            out_dir = os.path.join(self.config.output_dir, Path(label).stem)
            os.makedirs(out_dir, exist_ok=True)
            shutil.copyfile(pdf_path, os.path.join(out_dir, label if label.lower().endswith(".pdf") else f"{label}.pdf"))
            with open(os.path.join(out_dir, "output.json"), "w", encoding="utf-8") as f:
                json.dump(json_out, f, ensure_ascii=False, indent=2)
            with open(os.path.join(out_dir, "output.md"), "w", encoding="utf-8") as f:
                f.write(markdown)
            results.append({"file": label, "status": "success", "output_dir": out_dir})

        return results

    def process_image_group(self, items: list[tuple[str, str]]) -> list[dict]:
        """Same as process_pdf_group, but each item is one image (1 image ==
        1 "page") instead of a multi-page PDF -- `items` is
        `[(image_path, label), ...]`. See server/routes.py's /images endpoint.
        """
        results: list[dict] = []
        all_pages: list[Image.Image] = []
        loaded: list[tuple[str, str]] = []  # (label, image_path), 1:1 with all_pages
        for image_path, label in items:
            try:
                img = Image.open(image_path)
                img.load()
            except Exception as e:
                logger.exception("Failed to open image %s, skipping", label)
                results.append({"file": label, "status": "error", "error": str(e)})
                continue
            all_pages.append(img)
            loaded.append((label, image_path))

        if not loaded:
            return results

        try:
            pages_blocks = self._process_pages_batch(all_pages, debug_dir=None)
        except Exception as e:
            logger.exception("Batched processing failed for a %d-image group", len(loaded))
            for label, _image_path in loaded:
                results.append({"file": label, "status": "error", "error": str(e)})
            return results

        for (label, image_path), blocks in zip(loaded, pages_blocks):
            sub_blocks = [blocks]
            markdown = build_markdown(sub_blocks, self.layout.label_schema)
            json_out = build_json(label, sub_blocks)
            out_dir = os.path.join(self.config.output_dir, Path(label).stem)
            os.makedirs(out_dir, exist_ok=True)
            shutil.copyfile(image_path, os.path.join(out_dir, label))
            with open(os.path.join(out_dir, "output.json"), "w", encoding="utf-8") as f:
                json.dump(json_out, f, ensure_ascii=False, indent=2)
            with open(os.path.join(out_dir, "output.md"), "w", encoding="utf-8") as f:
                f.write(markdown)
            results.append({"file": label, "status": "success", "output_dir": out_dir})

        return results

    def _process_pages_batch(self, pages: list[Image.Image], debug_dir: str | None) -> list[list[PageBlock]]:
        """Runs layout/OCR/table processing (batched across the whole
        `pages` list) and returns each page's assembled blocks. Extracted
        from process_pdf() so process_pdf_group()/process_image_group() can
        feed it a page list concatenated from MULTIPLE source files (for
        cross-file batching) and split the result back afterward -- this
        method itself has no notion of "which file" a page came from.
        """
        # Phase 1a: whitespace crop + deskew -- genuinely page-specific/
        # sequential, each one lightweight.
        t_prep = time.time()
        pre_pages = [self._prepare_page_pre_orientation(pn, img) for pn, img in enumerate(pages)]
        logger.info("Whitespace/deskew for %d pages in %.2fs", len(pages), time.time() - t_prep)

        # Phase 1b: batch-detect the 0-degree candidate orientation
        # correction needs as its first step, for ALL pages in ONE call --
        # see OCR.detect_sorted_batch / correct_page_orientation's
        # dt_boxes_0 param. Only needed when page_orientation is enabled;
        # correct_page_orientation() never even looks at dt_boxes_0 otherwise.
        dt_boxes_0_per_page: list = [None] * len(pages)
        if self.config.page_orientation_enabled:
            t_detect = time.time()
            imgs_bgr_0 = [
                cv2.cvtColor(np.array(p.img.convert("RGB")), cv2.COLOR_RGB2BGR)
                for p in pre_pages
            ]
            dt_boxes_0_per_page = self.ocr.detect_sorted_batch(imgs_bgr_0)
            logger.info(
                "Batch-detected orientation baseline for %d pages in %.2fs",
                len(pages), time.time() - t_detect,
            )

        # Phase 1c: orientation correction for ALL pages, batched across the
        # whole document -- see page_orientation.py's
        # batch_correct_page_orientation() (each round -- 0-degree score,
        # then 180 for non-sideways pages or 90/270 for sideways pages that
        # still need it -- is batched across every page still needing that
        # round, instead of looping pages one at a time).
        t_orient = time.time()
        if self.config.page_orientation_enabled:
            orient_results = batch_correct_page_orientation(
                [p.img for p in pre_pages],
                dt_boxes_0_per_page,
                self.ocr,
                score_threshold=self.config.page_orientation_score_threshold,
                sample_max=self.config.page_orientation_sample_max,
                min_scores=self.config.page_orientation_min_scores,
                sideways_min_count=self.config.page_orientation_sideways_min_count,
                sideways_min_ratio=self.config.page_orientation_sideways_min_ratio,
                max_pages_per_batch=self.config.page_orientation_max_pages_per_batch,
            )
        else:
            orient_results = [(p.img, "0", 0.0, None, None) for p in pre_pages]
        logger.info(
            "Batch-corrected orientation for %d pages in %.2fs",
            len(pages), time.time() - t_orient,
        )

        prepped = [
            self._prepare_page_orientation(pn, pre_pages[pn], orient_results[pn], debug_dir)
            for pn in range(len(pages))
        ]

        # Phase 2: layout detection for ALL pages in ONE batched PaddleX
        # call instead of one call per page -- see PPDocLayoutBackend.detect_batch.
        t_layout = time.time()
        raw_blocks_per_page = self.layout.detect_batch([p.img for p in prepped], self.config.layout_threshold)
        logger.info("Layout-detected %d pages in one batch in %.2fs", len(pages), time.time() - t_layout)

        # Saved HERE, right after the batch call, rather than later per-page
        # in Phase 3 -- so the layout debug images exist (one per page,
        # covering the whole batch) even if something in Phase 3 (OCR/table
        # processing) crashes on some later page, and so you can eyeball
        # them immediately to confirm the batch call itself produced sane,
        # correctly-ordered per-page results.
        if debug_dir:
            for pn, raw_blocks in enumerate(raw_blocks_per_page):
                save_layout_debug(prepped[pn].img, raw_blocks, debug_dir, pn)

        # Phase 2.5: build every page's block skeleton (crop + deskew each
        # table, classify every other block's type) -- table CONTENT is
        # left as None, collected into one flat list spanning ALL pages, so
        # Phase 2.6 can batch the table-structure model across the WHOLE
        # document instead of per-page (a document with 1 table/page across
        # many pages would otherwise never batch anything).
        t_skel = time.time()
        page_blocks: list[list[PageBlock]] = []
        all_table_items: list[tuple[int, int, Image.Image]] = []  # (pn, tno, crop)
        for pn in range(len(pages)):
            blocks, table_crops = self._build_table_skeleton(pn, prepped[pn].img, raw_blocks_per_page[pn])
            page_blocks.append(blocks)
            all_table_items.extend((pn, tno, crop) for tno, crop in table_crops)
        logger.info(
            "Built page skeletons (%d table(s) total across %d pages) in %.2fs",
            len(all_table_items), len(pages), time.time() - t_skel,
        )

        # Phase 2.6: fill in table content -- batched across EVERY table in
        # the WHOLE DOCUMENT in one call when the backend supports it
        # (process_batch, mineru backend), otherwise one crop at a time
        # (tsr backend, or any backend that doesn't opt into batching).
        if all_table_items:
            t_tables = time.time()
            if hasattr(self.table_processor, "process_batch"):
                contents = self.table_processor.process_batch(all_table_items, debug_dir=debug_dir)
                kinds = getattr(self.table_processor, "last_table_kinds", None) or [None] * len(all_table_items)
            else:
                contents, kinds = [], []
                for pn, tno, crop in all_table_items:
                    contents.append(self.table_processor(crop, debug_dir=debug_dir, pn=pn, tno=tno))
                    # last_table_kind ("wire"/"wireless") is only set by the
                    # mineru backend (see MinerUTableProcessor) after it has
                    # classified this crop -- the tsr backend has no such
                    # concept, so getattr falls back to no suffix for it.
                    kinds.append(getattr(self.table_processor, "last_table_kind", None))
            logger.info(
                "Processed %d table(s) (batched across whole document) in %.2fs",
                len(all_table_items), time.time() - t_tables,
            )

            # tno IS the table's index within its own page's block list (see
            # _build_table_skeleton -- tables are appended to `blocks` in
            # tno order), so it doubles as the lookup key back into
            # page_blocks[pn]'s table-block positions.
            table_block_idx_by_page: dict[int, list[int]] = {
                pn: [i for i, b in enumerate(blocks) if b["content_type"] == "table"]
                for pn, blocks in enumerate(page_blocks)
            }
            for (pn, tno, crop), content, kind in zip(all_table_items, contents, kinds):
                block_idx = table_block_idx_by_page[pn][tno]
                page_blocks[pn][block_idx]["content"] = content
                if debug_dir:
                    save_table_crop(crop, debug_dir, pn, tno, suffix=f"_{kind}" if kind else "")

        # Phase 3a: per-page OCR detection + cropping (cheap, no FastOCR
        # call yet) -- collects every page's not-yet-recognized crops into
        # ONE flat list spanning the WHOLE document, mirroring table
        # content's Phase 2.5/2.6 pattern, so FastOCR recognizes them all
        # in as few batched forward passes as possible instead of once per
        # page.
        t_ocr_prep = time.time()
        ocr_preps: list[PageOcrPrep] = []
        all_ocr_crops: list[np.ndarray] = []
        # Which page each all_ocr_crops[i] belongs to -- built in the SAME
        # per-page order crops are appended below, so it can be zipped
        # back against the recognition output 1:1 with no risk of a page's
        # text ending up assigned to a different page.
        crop_owner_pn: list[int] = []
        for pn in range(len(pages)):
            page_crop_debug_dir = os.path.join(debug_dir, f"page_{pn + 1}_fastocr_crops") if debug_dir else None
            prep, crops = prepare_ocr_page(
                prepped[pn].img, self.ocr, crop_debug_dir=page_crop_debug_dir,
                dt_boxes=prepped[pn].dt_boxes_reuse, prerecognized=prepped[pn].prerecognized_reuse,
            )
            ocr_preps.append(prep)
            all_ocr_crops.extend(crops)
            crop_owner_pn.extend([pn] * len(crops))
        logger.info(
            "Prepared OCR boxes for %d pages (%d crop(s) need fresh recognition) in %.2fs",
            len(pages), len(all_ocr_crops), time.time() - t_ocr_prep,
        )

        # Phase 3b: recognize EVERY crop across the WHOLE document in ONE
        # FastOCR call -- TextRecognizer.__call__ still chunks internally
        # by ocr.fastocr_max_batch_size (see module/ocr/engine.py), so this
        # stays memory-bounded even for a huge multi-page document.
        t_ocr_rec = time.time()
        fresh_rec_res_all, _ = self.ocr.text_recognizer[0](all_ocr_crops) if all_ocr_crops else ([], 0.0)
        logger.info(
            "Recognized %d text crop(s) (batched across whole document) in %.2fs",
            len(all_ocr_crops), time.time() - t_ocr_rec,
        )

        # Slot each page's OWN slice of the whole-document recognition
        # results back to it -- crop_owner_pn[i]/fresh_rec_res_all[i] are
        # the SAME index into the SAME flat list built above (order is
        # preserved end-to-end through TextRecognizer.__call__), so this
        # grouping can't mix up which recognized text belongs to which page.
        fresh_rec_res_by_page: list[list] = [[] for _ in pages]
        for pn, res in zip(crop_owner_pn, fresh_rec_res_all):
            fresh_rec_res_by_page[pn].append(res)

        pages_ocr_boxes = [
            finish_ocr_page(ocr_preps[pn], fresh_rec_res_by_page[pn], self.ocr)
            for pn in range(len(pages))
        ]

        # Phase 3c: per-page reading-order/content assembly, now that every
        # page's blocks (table content) and ocr_boxes (recognized text) are
        # already built.
        pages_blocks = [
            self._process_page_after_layout(pn, prepped[pn], page_blocks[pn], pages_ocr_boxes[pn], debug_dir)
            for pn in range(len(pages))
        ]
        return pages_blocks

    def _prepare_page_pre_orientation(self, pn: int, img: Image.Image) -> _PrePage:
        """Whitespace crop + deskew -- kept separate from orientation
        correction so process_pdf() can batch-detect the 0-degree
        candidate for every page in one call in between (see
        process_pdf()'s Phase 1b) before running orientation correction
        per page."""
        cfg = self.config
        timings: dict[str, float] = {}
        t_stage = time.time()

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

        if cfg.page_deskew_enabled:
            img, tilt_angle = deskew_page(
                img, self.ocr,
                min_boxes=cfg.page_deskew_min_boxes,
                angle_threshold=cfg.page_deskew_angle_threshold,
                max_angle=cfg.page_deskew_max_angle,
            )
            if tilt_angle:
                logger.info("Page %d: deskewed by %.2f°", pn + 1, tilt_angle)
        _mark("page_deskew")

        return _PrePage(img=img, timings=timings)

    def _prepare_page_orientation(
        self,
        pn: int,
        pre: _PrePage,
        orient_result: tuple,
        debug_dir: str | None = None,
    ) -> _PreparedPage:
        """Per-page bookkeeping (timing mark, debug image) for the
        orientation result process_pdf() already computed for EVERY page,
        batched across the whole document -- see
        page_orientation.py's batch_correct_page_orientation(). No model
        call happens in this method anymore."""
        cfg = self.config
        timings = dict(pre.timings)
        t_stage = time.time()

        def _mark(name: str) -> None:
            nonlocal t_stage
            now = time.time()
            timings[name] = now - t_stage
            t_stage = now

        img, orient_label, orient_score, dt_boxes_reuse, prerecognized_reuse = orient_result
        if cfg.page_orientation_enabled:
            logger.info("Page %d: orientation %s° (score=%.2f)", pn + 1, orient_label, orient_score)
        _mark("page_orientation")

        if debug_dir:
            save_input_image(img, debug_dir, pn)

        return _PreparedPage(
            img=img,
            dt_boxes_reuse=dt_boxes_reuse,
            prerecognized_reuse=prerecognized_reuse,
            timings=timings,
        )

    def _build_table_skeleton(
        self,
        pn: int,
        img: Image.Image,
        raw_blocks: list[LayoutBlock],
    ) -> tuple[list[PageBlock], list[tuple[int, Image.Image]]]:
        """Crops + deskews every table block (cheap, no model calls) and
        classifies every other block's type. Table CONTENT is left as
        None -- process_pdf() collects every page's table crops into one
        flat list first (Phase 2.5), batches the table-structure model
        across the WHOLE document (Phase 2.6, see
        MinerUTableProcessor.process_batch), then fills the None slots in
        afterward -- so a single page's block list alone is never enough
        to know a table's final content by the time this method returns.

        Returns (blocks, table_crops) -- table_crops is [(tno, crop), ...],
        tno matching each table block's position within `blocks` (i.e.
        table_crops[k][0] is also that table's index among ONLY the table
        blocks in `blocks`, in order).
        """
        label_schema = self.layout.label_schema
        w, h = img.size
        blocks: list[PageBlock] = []
        table_crops: list[tuple[int, Image.Image]] = []
        tno = 0
        for b in raw_blocks:
            btype = b["type"].lower()
            if btype in label_schema.table_types:
                x0, y0, x1, y1 = map(int, b["bbox"])
                crop = img.crop((max(0, x0 - 2), max(0, y0 - 2), min(w, x1 + 2), min(h, y1 + 2)))
                crop, skew_angle = deskew_crop(crop)
                if abs(skew_angle) > 0.1:
                    logger.debug("Page %d: table deskewed by %.2f°", pn + 1, skew_angle)
                table_crops.append((tno, crop))
                blocks.append({**b, "content_type": "table", "content": None})
                tno += 1
            elif btype in label_schema.skip_types:
                blocks.append({**b, "content_type": "skip", "content": None})
            else:
                blocks.append({**b, "content_type": "text", "content": None})
        return blocks, table_crops

    def _process_page_after_layout(
        self,
        pn: int,
        prepped: _PreparedPage,
        blocks: list[PageBlock],
        ocr_boxes: list,
        debug_dir: str | None = None,
    ) -> list[PageBlock]:
        """Reading-order/content assembly for one page. `blocks` (from
        process_pdf()'s Phase 2.5/2.6) already has table content filled
        in, and `ocr_boxes` (from process_pdf()'s Phase 3a/3b) already has
        every text box detected+recognized -- both batched across the
        WHOLE document beforehand, so nothing in this method calls a
        model anymore."""
        cfg = self.config
        label_schema = self.layout.label_schema
        img = prepped.img
        w, h = img.size
        timings = prepped.timings
        n_raw_blocks = len(blocks)  # `blocks` gets reassigned below (dedup/unmatched/etc.)
        t_stage = time.time()

        def _mark(name: str) -> None:
            nonlocal t_stage
            now = time.time()
            timings[name] = now - t_stage
            t_stage = now

        if debug_dir:
            # save_layout_debug() already ran for every page right after
            # the Phase 2 batch call, in process_pdf() -- not repeated here.
            save_ocr_debug(img, ocr_boxes, debug_dir, pn)
        _mark("ocr_debug_save")

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

        # Sum of this page's OWN marked stages (prep phases + post-layout
        # phase) -- NOT wall-clock since Phase 1a started, since that would
        # also count time spent prepping OTHER pages and running the
        # batched detect/layout calls, none of which is this page's own cost.
        elapsed = sum(timings.values())
        n_tbl = sum(1 for b in blocks if b["content_type"] == "table")
        n_text = sum(1 for b in blocks if b["content_type"] == "text")
        timings_str = " ".join(f"{k}={v:.2f}s" for k, v in timings.items())
        logger.info(
            "Page %d done in %.1fs: layout=%d ocr=%d table=%d text=%d | %s",
            pn + 1, elapsed, n_raw_blocks, len(ocr_boxes), n_tbl, n_text, timings_str,
        )

        return blocks


def build_pipeline(conf: dict | None = None) -> DocumentPipeline:
    """Single source of truth for constructing a DocumentPipeline -- used by
    both the CLI (full_pipeline.py) and the FastAPI server so there is no
    duplicated business logic between the two entry points."""
    conf = conf or load_pipeline_conf()
    config = PipelineConfig.from_conf(conf)
    ocr = get_ocr_engine(conf)
    return DocumentPipeline(
        get_layout_backend(conf),
        ocr,
        get_table_processor(conf, ocr),
        config,
    )
