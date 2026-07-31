import html as html_module
import logging

import numpy as np
from PIL import Image

from ..ocr import OCREngine
from ..pipeline.debug import save_table_ocr_debug
from .surya_wireless import SuryaWirelessTableProcessor

logger = logging.getLogger(__name__)

# MinerU's UNet wired-table cell recovery joins the OCR text fragments
# matched to the same cell with no separator, which drops the space at
# fragment boundaries within a cell (e.g. "Ngành" + "nông," -> "Ngànhnông,").
# Join with a space instead.
import mineru.model.table.rec.unet_table.utils_table_recover as _unet  # noqa: E402
_unet._cell_text = lambda m, i: " ".join(m.get(i, []))

# MinerU's UnetTableModel.predict() (mineru/model/table/rec/unet_table/main.py)
# decides wired-vs-wireless via 4 OR'd conditions. Per real testing on this
# project's documents, the FIRST one -- a cell-count-based `switch_flag`
# (estimates wired's table "scale" as sqrt(non-blank cell count), assuming a
# roughly-square table, then switches to wireless whenever wireless reports
# meaningfully more non-blank cells than that estimate) -- was misclassifying
# genuinely wired tables as wireless too often (wireless's own structure
# model can report more "non-blank cells" than wired even on a table that
# IS wired, e.g. by segmenting a merged cell into several). The other 3
# conditions (total-cell-count gap + wired-cell-shortfall, tiny-equal-count
# tables, and wired-OCR-text-coverage shortfall -- see mineru_processor's
# _wired.predict() docs) are kept EXACTLY as mineru implements them; only
# this one heuristic is disabled. Reimplemented here (not just deleting a
# few lines via monkeypatch, since switch_flag is computed inline, not in
# its own patchable function) by copying predict()'s body from mineru
# 2025-xx (see MinerU/mineru/model/table/rec/unet_table/main.py) with the
# switch_flag block and the now-unused blank-cell BeautifulSoup counting
# removed -- re-verify this patch still matches upstream's OTHER 3
# conditions if mineru is ever upgraded.
import mineru.model.table.rec.unet_table.main as _unet_main  # noqa: E402


def _wired_predict_without_cell_count_switch(self, input_img, ocr_result, wireless_html_code, return_metadata=False):
    if isinstance(input_img, Image.Image):
        np_img = np.asarray(input_img)
    elif isinstance(input_img, np.ndarray):
        np_img = input_img
    else:
        raise ValueError("Input must be a pillow object or a numpy array.")

    if ocr_result is None:
        import cv2
        bgr_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
        ocr_result = self.ocr_engine.ocr(bgr_img)[0]
        ocr_result = [
            [item[0], _unet_main.escape_html(item[1][0]), item[1][1]]
            for item in ocr_result
            if len(item) == 2 and isinstance(item[1], tuple)
        ]

    try:
        wired_table_results = self.wired_table_model(np_img, ocr_result)
        wired_structure_results = (
            self.wired_table_model(np_img, need_ocr=False)
            if return_metadata
            else None
        )

        wired_html_code = wired_table_results.pred_html
        wired_len = _unet_main.count_table_cells_physical(wired_html_code)
        wireless_len = _unet_main.count_table_cells_physical(wireless_html_code)
        gap_of_len = wireless_len - wired_len

        wireless_text_count = 0
        wired_text_count = 0
        for ocr_res in ocr_result:
            if ocr_res[1] in wireless_html_code:
                wireless_text_count += 1
            if ocr_res[1] in wired_html_code:
                wired_text_count += 1

        selected_model = "wired"
        if (
            (0 <= gap_of_len <= 5 and wired_len <= round(wireless_len * 0.75))
            or (gap_of_len == 0 and wired_len <= 4)
            or (wired_text_count <= wireless_text_count * 0.6 and wireless_text_count >= 10)
        ):
            html_code = wireless_html_code
            selected_model = "wireless"
        else:
            html_code = wired_html_code

        if return_metadata:
            return {
                "html": html_code,
                "selected_model": selected_model,
                "wired_cell_bboxes": None if wired_structure_results is None else wired_structure_results.cell_bboxes,
                "wired_logic_points": None if wired_structure_results is None else wired_structure_results.logic_points,
                "wired_html": wired_html_code,
            }
        return html_code
    except Exception as e:
        logger.warning("Wired-vs-wireless compare failed, falling back to wireless: %s", e)
        if return_metadata:
            return {
                "html": wireless_html_code,
                "selected_model": "wireless",
                "wired_cell_bboxes": None,
                "wired_logic_points": None,
                "wired_html": "",
            }
        return wireless_html_code


_unet_main.UnetTableModel.predict = _wired_predict_without_cell_count_switch


class MinerUTableProcessor:
    """Table backend: orientation fix -> wired/wireless classify -> OCR ->
    table-structure model -> HTML.

    Wireless (`SuryaWirelessTableProcessor`, see surya_wireless.py) ALWAYS
    runs -- it's the baseline/fallback candidate for every table regardless
    of classification. When classified wired, the wired (UNet) model ALSO
    runs, via MinerU's own `UnetTableModel.predict()` wrapper -- which
    internally re-runs `wired_table_model()` itself, then applies MinerU's
    hard-coded cell-count/text-match/blank-cell heuristic
    (`mineru/model/table/rec/unet_table/main.py:283-370`) comparing the
    wired result against the wireless HTML we hand it, falling back to
    wireless when wired looks clearly worse (too few cells, too little
    matched OCR text, etc). This restores MinerU's original auto-compare
    design (which this file's strict-either/or dispatch had replaced) --
    with Surya swapped in for MinerU's own SLANet-Plus as the wireless
    candidate everywhere that heuristic reads "wireless".

    Wired-vs-wireless classification uses RapidAI's `table_cls` (QAnything
    classifier, `model_type="q"`, `pip install table_cls`) instead of
    MinerU's own `TableCls` atom model -- swapped per user testing. Unlike
    MinerU's classifier, this one returns only a label ("wired"/"wireless"),
    no confidence score, so there's no low-confidence-falls-back-to-wired
    heuristic anymore.

    For the WIRED path, detection AND recognition reuse the shared
    `OCREngine` entirely -- `ocr.text_detector[0]` (whatever
    `ocr.det_backend` resolves to: the onnx detector by default, or
    MinerU's own PP-OCRv6 if configured) and `ocr.text_recognizer[0].detector`
    (the VietOCR predictor module/ocr already loaded, same weight file) --
    instead of loading a second VietOCR copy.

    The WIRELESS path (`SuryaWirelessTableProcessor`) is the one deliberate
    exception: per explicit user request it uses Surya's OWN
    `DetectionPredictor` for bbox detection (mirroring
    surya_v1_table_to_html_vietocr.ipynb exactly), NOT the shared detector --
    only recognition there still reuses the shared VietOCR predictor.

    `process_batch()` is an OPTIONAL extra entry point (not part of the
    `TableProcessor` protocol -- callers duck-type-check for it via
    `hasattr`) that batches the Surya call across MULTIPLE table crops at
    once instead of calling `__call__()` per crop; document_pipeline.py
    uses it when present (this backend) and falls back to per-crop
    `__call__()` otherwise (the "tsr" backend, which has no such method).

    All `mineru` imports needed JUST for this table backend are lazy (inside
    methods, not at module import time) so `module.table` stays importable
    -- and the "tsr" table backend still works -- on installs that don't
    have `mineru` and never select `table.backend: mineru` (independent of
    whatever `ocr.det_backend` is set to).
    """

    def __init__(self, ocr: OCREngine, max_batch_size: int = 8, vietocr_max_batch_size: int = 64):
        from mineru.backend.pipeline.model_init import AtomModelSingleton

        self._ocr = ocr
        self._vietocr = ocr.text_recognizer[0].detector
        self._atom = AtomModelSingleton()
        # Caps how many crops process_batch() hands to TableOrientationCls
        # at once -- a document with many tables would otherwise batch them
        # all in one unbounded call and risk a GPU OOM.
        self._max_batch_size = max_batch_size
        # Caps how many text-line crops _recognize_from_boxes() (wired-table
        # VietOCR recognition) and SuryaWirelessTableProcessor's own VietOCR
        # calls stack into one predict_batch() forward pass at a time --
        # shared with module/ocr/engine.py's page-level OCR cap (VietOCR's
        # own per-crop cost is much smaller than a table-structure-model
        # forward pass, so it gets its own, usually higher, cap instead of
        # reusing max_batch_size above).
        self._vietocr_max_batch_size = vietocr_max_batch_size
        # Set by __call__ once TableCls has run -- "wire" or "wireless".
        # Read by document_pipeline.py (via getattr, since only this backend
        # has the concept) to suffix the deskewed-crop debug filename.
        self.last_table_kind: str | None = None
        # Set by process_batch() -- one kind ("wire"/"wireless") per item in
        # the last batch call, same order (last_table_kind above is only
        # meaningful for the single-crop __call__ path).
        self.last_table_kinds: list[str] = []
        self._load_table_models()

    def _load_table_models(self) -> None:
        import torch
        from mineru.backend.pipeline.model_list import AtomicModel
        from mineru.utils.config_reader import get_device
        from table_cls import TableCls

        self._AtomicModel = AtomicModel
        self._ori_cls = self._atom.get_atom_model(atom_model_name=AtomicModel.TableOrientationCls)
        # TableOrientationCls/WiredTable both resolve their device through
        # mineru's own global get_device() (see build_table_onnx_providers) --
        # log that resolved value here rather than "cpu"/"cuda" guessed from
        # torch.cuda.is_available() directly, since get_device() also
        # respects the MINERU_DEVICE_MODE env override.
        logger.info("[device] TableOrientationCls (mineru): device=%s", get_device())
        self._table_cls = TableCls(model_type="q")  # RapidAI QAnything wired/wireless classifier
        # table_cls's own OrtInferSession hardcodes CPUExecutionProvider only
        # -- no GPU branch exists in this library, confirmed from its source.
        logger.info("[device] table_cls (RapidAI wired/wireless classifier): device=cpu (no GPU support in this library)")
        self._wired = self._atom.get_atom_model(atom_model_name=AtomicModel.WiredTable, lang=None)
        logger.info("[device] WiredTable / UNet (mineru): device=%s", get_device())
        self._surya = SuryaWirelessTableProcessor(
            self._vietocr,
            max_batch_size=self._max_batch_size,
            vietocr_max_batch_size=self._vietocr_max_batch_size,
        )
        logger.info(
            "[device] Surya (wireless table structure + detector): device=%s",
            "cuda" if torch.cuda.is_available() else "cpu",
        )

    @staticmethod
    def _to_rgb(image) -> np.ndarray:
        import cv2

        if isinstance(image, str):
            return np.asarray(Image.open(image).convert("RGB"))
        if isinstance(image, Image.Image):
            return np.asarray(image.convert("RGB"))
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            if image.shape[2] == 4:
                return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            return image
        raise TypeError(f"Unsupported image type: {type(image)}")

    def _fix_orientation(self, img: np.ndarray) -> np.ndarray:
        import cv2

        label = str(self._ori_cls.predict(img) or "0")
        if label == "270":
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        if label == "90":
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return img

    def _fix_orientation_batch(self, imgs: list[np.ndarray]) -> list[np.ndarray]:
        """Batches TableOrientationCls across MULTIPLE crops, chunked
        `self._max_batch_size` at a time -- `.predict()` above is itself
        just a thin single-item wrapper over
        `MineruTableOrientationClsModel.batch_predict()`
        (mineru/model/table/cls/mineru_table_ori_cls.py:29-33), so this is
        the SAME underlying code path, just given more than one image at
        once. Chunking (both the outer crop list AND det_batch_size, its
        own internal OCR-detection sub-batch size) keeps a document with
        many tables from handing them all to one unbounded call. Applies
        to every crop regardless of wired/wireless classification (this
        runs BEFORE classification)."""
        import cv2

        if not imgs:
            return []
        fixed = []
        for start in range(0, len(imgs), self._max_batch_size):
            chunk = imgs[start:start + self._max_batch_size]
            labels = self._ori_cls.batch_predict(
                [{"table_img": img} for img in chunk], det_batch_size=self._max_batch_size
            )
            for img, label in zip(chunk, labels):
                label = str(label or "0")
                if label == "270":
                    fixed.append(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))
                elif label == "90":
                    fixed.append(cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE))
                else:
                    fixed.append(img)
        return fixed

    def _recognize_from_boxes(self, bgr: np.ndarray, dt_boxes) -> list:
        """VietOCR recognition for ALREADY-detected boxes -- shared by
        _run_ocr() (detects then recognizes, single crop) and
        process_batch() (detects ALL wired crops in one batched call, then
        calls this per-crop for the recognition step).

        Recognizes every box's crop in ONE VietOCR encoder+decoder forward
        pass (chunked by self._vietocr_max_batch_size), instead of one call
        per box -- predict_batch() handles the variable-width padding/masking
        correctly (see vietocr/tool/predictor.py).

        Returns list of [quad_box (4x2 list), html_escaped_text (str), score (float)].
        """
        import cv2

        if dt_boxes is None:
            return []

        boxes = [np.array(box, dtype=np.float32) for box in dt_boxes]
        # Use the SAME crop function (with its tuned diacritic padding) that
        # the shared VietOCR predictor was validated against -- MinerU's own
        # get_rotate_crop_image_for_text_rec() crops tighter and was
        # clipping Vietnamese diacritics, degrading recognition.
        crops = [
            Image.fromarray(cv2.cvtColor(self._ocr.get_rotate_crop_image(bgr, box), cv2.COLOR_BGR2RGB))
            for box in boxes
        ]

        result = []
        for start in range(0, len(crops), self._vietocr_max_batch_size):
            chunk_boxes = boxes[start : start + self._vietocr_max_batch_size]
            chunk_crops = crops[start : start + self._vietocr_max_batch_size]
            for box, (text, score) in zip(chunk_boxes, self._vietocr.predict_batch(chunk_crops)):
                result.append([box.tolist(), html_module.escape(text), score])

        return result

    def _run_ocr(self, img: np.ndarray) -> list:
        """Shared detector (via ocr.detect_sorted()) finds bbox -> shared
        VietOCR predictor reads text (single crop; process_batch() uses
        detect_sorted_batch() + _recognize_from_boxes() instead to batch
        the detection step across multiple wired crops).
        """
        import cv2

        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # MinerU's wireless-table matcher (matcher.py::match_result) appends
        # OCR fragments to a cell in WHATEVER order dt_boxes arrives in --
        # no positional re-sort of its own (unlike the wired/UNet path,
        # which does sort_and_gather_ocr_res internally). Raw detector
        # output order isn't reading order, so without this the OCR result
        # order silently becomes the final cell-text order once a cell
        # holds more than one fragment. detect_sorted() already applies
        # ocr.sorted_boxes() for us.
        dt_boxes = self._ocr.detect_sorted(bgr)
        return self._recognize_from_boxes(bgr, dt_boxes)

    def _is_wired(self, img: np.ndarray) -> bool:
        """Return True when the table has visible grid lines (wired table)."""
        import cv2

        # table_cls's LoadImage leaves a raw ndarray untouched (assumes it's
        # already BGR, cv2's native order) -- our img is RGB.
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        label, _elapse = self._table_cls(bgr)
        return label == "wired"

    @staticmethod
    def _trim_to_table_tag(raw: str | None) -> str:
        """Strip <html><body> wrappers, keep only <table>...</table>."""
        if not raw:
            return "<table></table>"
        s = raw.find("<table>")
        e = raw.rfind("</table>")
        if s != -1 and e != -1:
            return raw[s : e + len("</table>")]
        return raw

    def __call__(
        self,
        crop: Image.Image,
        debug_dir: str | None = None,
        pn: int = 0,
        tno: int = 0,
    ) -> str:
        img = self._to_rgb(crop)
        img = self._fix_orientation(img)

        is_wired = self._is_wired(img)
        self.last_table_kind = "wire" if is_wired else "wireless"
        logger.info("Page %d table %d: classified as %s", pn + 1, tno, self.last_table_kind)

        # Wireless (Surya) always runs -- baseline for every table, and the
        # fallback candidate MinerU's own compare heuristic below can pick
        # instead of wired.
        wireless_html = self._surya(img)

        if is_wired:
            # Shared detector -- feeds the wired model's own recognition AND
            # is what UnetTableModel.predict()'s heuristic below counts
            # matched-text against for both candidates.
            ocr_result = self._run_ocr(img)
            if ocr_result:
                result = self._wired.predict(img, ocr_result, wireless_html, return_metadata=True)
                final_html = result["html"]
                logger.info(
                    "Page %d table %d: wired-vs-wireless compare -> selected %s",
                    pn + 1, tno, result["selected_model"],
                )
            else:
                final_html = wireless_html
        else:
            ocr_result = self._surya.last_ocr_result
            final_html = wireless_html

        logger.debug("OCR found %d text boxes", len(ocr_result))
        if debug_dir:
            save_table_ocr_debug(
                Image.fromarray(img),
                [(box, (text, score)) for box, text, score in ocr_result],
                debug_dir, pn, tno,
                suffix=f"_{self.last_table_kind}",
            )

        return self._trim_to_table_tag(final_html or wireless_html)

    def process_batch(
        self,
        items: list[tuple[int, int, Image.Image]],
        debug_dir: str | None = None,
    ) -> list[str]:
        """Batches across ALL given table crops in ONE call each, instead of
        __call__()'s one-crop-at-a-time path, for every model that exposes
        a batch API:
          - TableOrientationCls (_fix_orientation_batch) -- every crop,
            wired or wireless, since this runs before classification.
          - Surya's wireless-structure model (SuryaWirelessTableProcessor.call_batch)
            -- every crop too, since it's always the baseline/fallback
            candidate for the compare heuristic (see class docstring), not
            just wireless-classified ones.
          - The shared detector (ocr.detect_sorted_batch), across every
            WIRED-classified crop -- feeds both the wired model's own
            recognition and the compare heuristic's text-match count.
            Chunked by THIS class's own self._max_batch_size (table crops),
            not the OCR instance's page-level default -- a sane batch size
            for full pages and for small table crops isn't the same number.
        Classification (table_cls) and the wired (UNet) model itself stay
        per-crop -- neither library exposes a batch API (see class
        docstring for why UNet specifically can't be batched: its
        postprocessing is classical per-image OpenCV line/contour
        detection, not a vectorizable batch operation).

        `items` is a list of (pn, tno, crop) -- pn/tno are only used for
        logging/debug filenames, not for anything semantic. Returns one
        HTML content string per item, in the SAME ORDER as `items`, and
        refreshes `self.last_table_kinds` (parallel list) for the caller's
        debug-filename suffix.
        """
        import cv2

        if not items:
            self.last_table_kinds = []
            return []

        rgb_imgs = [self._to_rgb(crop) for _pn, _tno, crop in items]
        fixed_imgs = self._fix_orientation_batch(rgb_imgs)

        prepared = []  # (pn, tno, img_rgb, is_wired)
        for (pn, tno, _crop), img in zip(items, fixed_imgs):
            is_wired = self._is_wired(img)  # no batch API -- stays per-crop
            prepared.append((pn, tno, img, is_wired))

        wireless_htmls = self._surya.call_batch([p[2] for p in prepared])
        surya_ocr_results = list(self._surya.last_ocr_results)

        # Shared detector, batched across ALL wired-classified crops.
        wired_positions = [i for i, p in enumerate(prepared) if p[3]]
        wired_bgrs = [cv2.cvtColor(prepared[i][2], cv2.COLOR_RGB2BGR) for i in wired_positions]
        wired_dt_boxes_batch = (
            self._ocr.detect_sorted_batch(wired_bgrs, max_batch_size=self._max_batch_size)
            if wired_bgrs else []
        )
        wired_ocr_results = {
            i: self._recognize_from_boxes(bgr, dt_boxes)
            for i, bgr, dt_boxes in zip(wired_positions, wired_bgrs, wired_dt_boxes_batch)
        }

        self.last_table_kinds = []
        results = []
        for i, ((pn, tno, img, is_wired), wireless_html, surya_ocr_result) in enumerate(
            zip(prepared, wireless_htmls, surya_ocr_results)
        ):
            kind = "wire" if is_wired else "wireless"
            self.last_table_kinds.append(kind)
            logger.info("Page %d table %d: classified as %s", pn + 1, tno, kind)

            if is_wired:
                ocr_result = wired_ocr_results[i]
                if ocr_result:
                    result = self._wired.predict(img, ocr_result, wireless_html, return_metadata=True)
                    final_html = result["html"]
                    logger.info(
                        "Page %d table %d: wired-vs-wireless compare -> selected %s",
                        pn + 1, tno, result["selected_model"],
                    )
                else:
                    final_html = wireless_html
            else:
                ocr_result = surya_ocr_result
                final_html = wireless_html

            if debug_dir:
                save_table_ocr_debug(
                    Image.fromarray(img),
                    [(box, (text, score)) for box, text, score in ocr_result],
                    debug_dir, pn, tno,
                    suffix=f"_{kind}",
                )

            results.append(self._trim_to_table_tag(final_html or wireless_html))

        return results
