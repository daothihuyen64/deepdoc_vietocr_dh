import html as html_module
import logging
import time
import traceback

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

# ---------------------------------------------------------------------------
# The 3 monkeypatches below add REAL batch support to MinerU's wired (UNet)
# table pipeline, which the version of `mineru[pipeline]` this project
# installs from PyPI does not have at all (only single-image `predict()`/
# `__call__()`/`infer()` exist there -- confirmed by reading that package's
# actual installed source). Ported from a WIP fork
# (C:\Proton_X\MinerU, not a project dependency -- copied here, not
# installed/imported from there) that adds true ONNX batch inference
# (TSRUnet.batch_call: N images concatenated into ONE forward pass, instead
# of always forcing batch=1) plus batched wired-vs-wireless orchestration on
# top. Kept as monkeypatches (same pattern as `_cell_text` above) rather
# than switching this project's mineru dependency to that fork, so every
# deploy target (local/Docker/vast.ai) keeps installing plain
# `mineru[pipeline]` from PyPI with zero changes -- this code IS the patch,
# not a reference to an external package.
import mineru.model.table.rec.unet_table.main as _unet_main  # noqa: E402
import mineru.model.table.rec.unet_table.table_structure_unet as _unet_tsr  # noqa: E402


def _tsr_postprocess_to_polygons(self, img, pred, **kwargs):
    polygons, rotated_polygons = self.postprocess(img, pred, **kwargs)
    if polygons.size == 0:
        return None, None
    polygons = polygons.reshape(polygons.shape[0], 4, 2)
    polygons[:, 3, :], polygons[:, 1, :] = (
        polygons[:, 1, :].copy(),
        polygons[:, 3, :].copy(),
    )
    rotated_polygons = rotated_polygons.reshape(rotated_polygons.shape[0], 4, 2)
    rotated_polygons[:, 3, :], rotated_polygons[:, 1, :] = (
        rotated_polygons[:, 1, :].copy(),
        rotated_polygons[:, 3, :].copy(),
    )
    _, idx = _unet.sorted_ocr_boxes(
        [_unet.box_4_2_poly_to_box_4_1(poly_box) for poly_box in rotated_polygons],
        threhold=0.4,
    )
    return polygons[idx], rotated_polygons[idx]


def _tsr_batch_call(self, imgs, **kwargs):
    """Batches N images into as few ONNX calls as possible -- TSRUnet.infer()
    always forces batch=1 (`session(input["img"][None, ...])`); this groups
    images by their (already deterministic) preprocessed shape and runs ONE
    concatenated ONNX call PER GROUP instead.

    preprocess() keeps each image's own aspect ratio (resize_img's
    imrescale/rescale_size fits the image within (inp_height, inp_width)
    WITHOUT forcing both dims to exactly that size -- only the LONGER side
    hits the target; the shorter side ends up smaller unless the crop
    happens to be square: `scale_factor = min(long_edge/max(h,w),
    short_edge/min(h,w))`, applied to BOTH dims). Different table crops
    generally have different aspect ratios, so their preprocessed shapes
    differ too -- np.concatenate needs matching shapes.

    An earlier version of this method zero-padded every image up to the
    fixed (inp_height, inp_width) canvas instead of grouping, to allow ONE
    single call regardless of shape. That was tried and REJECTED: UNet's
    large receptive field (many conv layers + skip connections) reads the
    artificial zero-padding border when convolving near the real image's
    edge -- the model was never trained on inputs with an artificial zero
    region like that, so its line-detection near that edge gets subtly
    corrupted. Confirmed empirically (mineru side): padding produced 173
    detected cells vs 174 for the padding-free single-image path on the
    same table -- a real, silent accuracy regression, not just a
    theoretical risk. Grouping by shape instead means every ONNX call is
    byte-identical to the single-image path (no artificial content ever
    added), at the cost of needing more than one call when a batch mixes
    aspect ratios -- same shape-grouping technique already used for the
    layout backend, see PPDocLayoutBackend.detect_batch(). Real PDFs
    mostly have tables sharing 1-2 orientations, so this still batches
    the large majority of same-shaped tables together in practice.
    """
    if not imgs:
        return []
    imgs_info = [self.preprocess(img) for img in imgs]

    groups: dict[tuple[int, int], list[int]] = {}
    for idx, info in enumerate(imgs_info):
        _, _c, h, w = info["img"].shape
        groups.setdefault((h, w), []).append(idx)

    results: list = [None] * len(imgs)
    for indices in groups.values():
        batch = np.concatenate([imgs_info[i]["img"] for i in indices], axis=0)
        raw_output = self.session([batch])[0]  # (1, len(indices), h, w)
        for pos, idx in enumerate(indices):
            pred = raw_output[0, pos].astype(np.uint8)
            results[idx] = self._postprocess_to_polygons(imgs[idx], pred, **kwargs)
    return results  # every index was filled by exactly one group above


_unet_tsr.TSRUnet._postprocess_to_polygons = _tsr_postprocess_to_polygons
_unet_tsr.TSRUnet.batch_call = _tsr_batch_call


def _wired_call_with_polygons(self, img, ocr_result, polygons, rotated_polygons, **kwargs):
    """The single-image part of WiredTableRecognition.__call__() (OCR-cell
    matching + HTML generation) factored out so batch_call() below can
    reuse it against ALREADY-computed TSRUnet polygons instead of
    re-running structure detection per crop."""
    need_ocr = kwargs.get("need_ocr", True) if kwargs else True
    col_threshold = kwargs.get("col_threshold", 15) if kwargs else 15
    row_threshold = kwargs.get("row_threshold", 10) if kwargs else 10

    if polygons is None:
        return _unet_main.WiredTableOutput("", None, None, 0.0)

    s = time.perf_counter()
    try:
        table_res, logi_points = self.table_recover(rotated_polygons, row_threshold, col_threshold)
        polygons[:, 1, :], polygons[:, 3, :] = (polygons[:, 3, :].copy(), polygons[:, 1, :].copy())
        if not need_ocr:
            sorted_polygons, idx_list = _unet.sorted_ocr_boxes(
                [_unet.box_4_2_poly_to_box_4_1(box) for box in polygons]
            )
            return _unet_main.WiredTableOutput("", sorted_polygons, logi_points[idx_list], time.perf_counter() - s)

        cell_box_det_map, _not_match_ocr_boxes = _unet_main.match_ocr_cell(ocr_result, polygons)
        cell_box_det_map = self.fill_blank_rec(img, polygons, cell_box_det_map)
        t_rec_ocr_list = self.transform_res(cell_box_det_map, polygons, logi_points)
        t_rec_ocr_list = self.sort_and_gather_ocr_res(t_rec_ocr_list)
        cell_box_det_map = {
            t_box_ocr["cell_idx"]: [ocr_box_and_text[1] for ocr_box_and_text in t_box_ocr["t_ocr_res"]]
            for t_box_ocr in t_rec_ocr_list
        }
        pred_html = _unet_main.plot_html_table(logi_points, cell_box_det_map, polygons)
        polygons = np.array(polygons).reshape(-1, 8)
        logi_points = np.array(logi_points)
        elapse = time.perf_counter() - s
    except Exception:
        logger.warning("Wired table cell/HTML generation failed:\n%s", traceback.format_exc())
        return _unet_main.WiredTableOutput("", None, None, 0.0)
    return _unet_main.WiredTableOutput(pred_html, polygons, logi_points, elapse)


def _wired_batch_call(self, imgs_with_ocr, **kwargs):
    """imgs_with_ocr: list of (rgb_img, ocr_result) pairs -- batches TSRUnet
    structure detection (one ONNX call for every image) then runs the
    per-image OCR-matching/HTML-generation tail for each."""
    if not imgs_with_ocr:
        return []
    imgs = [self.load_img(pair[0]) for pair in imgs_with_ocr]
    ocr_results = [pair[1] for pair in imgs_with_ocr]
    structure_results = self.table_structure.batch_call(imgs, **kwargs)
    return [
        self._call_with_polygons(img, ocr, polygons, rotated_polygons, **kwargs)
        for img, ocr, (polygons, rotated_polygons) in zip(imgs, ocr_results, structure_results)
    ]


_unet_main.WiredTableRecognition._call_with_polygons = _wired_call_with_polygons
_unet_main.WiredTableRecognition.batch_call = _wired_batch_call


def _select_html_without_cell_count_switch(self, wired_html_code, wireless_html_code, ocr_result):
    """Wired-vs-wireless decision -- factored out of predict() so both it
    and batch_predict() below share ONE implementation.

    MinerU's own version ORs in a 4th condition first: a cell-count-based
    `switch_flag` (estimates wired's table "scale" as sqrt(non-blank cell
    count), assuming a roughly-square table, then switches to wireless
    whenever wireless reports meaningfully more non-blank cells than that
    estimate). Per real testing on this project's documents, that
    condition was misclassifying genuinely wired tables as wireless too
    often (wireless's own structure model can report more "non-blank
    cells" than wired even on a table that IS wired, e.g. by segmenting a
    merged cell into several) -- deliberately DROPPED here. The other 3
    conditions (total-cell-count gap + wired-cell-shortfall, tiny-equal-
    count tables, wired-OCR-text-coverage shortfall) are kept exactly as
    mineru implements them.
    """
    wired_len = _unet_main.count_table_cells_physical(wired_html_code)
    wireless_len = _unet_main.count_table_cells_physical(wireless_html_code)
    gap_of_len = wireless_len - wired_len

    wireless_text_count = 0
    wired_text_count = 0
    for ocr_res in ocr_result:
        if ocr_res[1] in (wireless_html_code or ""):
            wireless_text_count += 1
        if ocr_res[1] in (wired_html_code or ""):
            wired_text_count += 1

    if (
        (0 <= gap_of_len <= 5 and wired_len <= round(wireless_len * 0.75))
        or (gap_of_len == 0 and wired_len <= 4)
        or (wired_text_count <= wireless_text_count * 0.6 and wireless_text_count >= 10)
    ):
        return "wireless", wireless_html_code
    return "wired", wired_html_code


def _wired_predict(self, input_img, ocr_result, wireless_html_code, return_metadata=False):
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
        selected_model, html_code = self._select_html(wired_html_code, wireless_html_code, ocr_result)

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


def _wired_batch_predict(self, table_res_list):
    """Batches the wired (UNet) structure model across EVERY entry in
    `table_res_list` that has a non-empty "ocr_result" in ONE ONNX call
    (see WiredTableRecognition.batch_call/TSRUnet.batch_call above), then
    runs the (cheap, CPU-only) wired-vs-wireless compare per entry.

    Each entry is a dict: {"wired_table_img": ..., "ocr_result": ...,
    "table_res": {"html": <wireless html, pre-filled>}}. Updates
    table_res["html"] IN PLACE (to the winning HTML) for every entry that
    had a usable ocr_result -- entries without one are left untouched
    (keeping their pre-filled wireless HTML, same fallback as predict()'s
    `if ocr_result:` guard). Also stashes table_res["selected_model"] (not
    part of mineru's own version) so callers can still log which model won,
    same as predict()'s return_metadata=True path does for the single-item case.
    """
    valid = [d for d in table_res_list if d.get("ocr_result")]
    if not valid:
        return

    imgs_with_ocr = []
    for d in valid:
        img = d["wired_table_img"]
        np_img = np.asarray(img) if isinstance(img, Image.Image) else img
        imgs_with_ocr.append((np_img, d["ocr_result"]))

    wired_outputs = self.wired_table_model.batch_call(imgs_with_ocr)

    for d, wired_out in zip(valid, wired_outputs):
        wireless_html = d["table_res"].get("html")
        try:
            selected_model, html_code = self._select_html(wired_out.pred_html, wireless_html, d["ocr_result"])
        except Exception as e:
            logger.warning("Wired-vs-wireless batch compare failed, falling back to wireless: %s", e)
            selected_model, html_code = "wireless", wireless_html
        d["table_res"]["html"] = html_code
        d["table_res"]["selected_model"] = selected_model


_unet_main.UnetTableModel._select_html = _select_html_without_cell_count_switch
_unet_main.UnetTableModel.predict = _wired_predict
_unet_main.UnetTableModel.batch_predict = _wired_batch_predict


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
    (the FastOCR predictor module/ocr already loaded, same weight file) --
    instead of loading a second FastOCR copy.

    The WIRELESS path (`SuryaWirelessTableProcessor`) is the one deliberate
    exception: per explicit user request it uses Surya's OWN
    `DetectionPredictor` for bbox detection (mirroring
    surya_v1_table_to_html_fastocr.ipynb exactly), NOT the shared detector --
    only recognition there still reuses the shared FastOCR predictor.

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

    def __init__(
        self,
        ocr: OCREngine,
        max_batch_size: int = 8,
        fastocr_max_batch_size: int = 64,
        wired_max_batch_size: int = 4,
    ):
        from mineru.backend.pipeline.model_init import AtomModelSingleton

        self._ocr = ocr
        self._fastocr = ocr.text_recognizer[0].detector
        self._atom = AtomModelSingleton()
        # Caps how many crops process_batch() hands to TableOrientationCls
        # at once -- a document with many tables would otherwise batch them
        # all in one unbounded call and risk a GPU OOM.
        self._max_batch_size = max_batch_size
        # Caps how many text-line crops _recognize_from_boxes() (wired-table
        # FastOCR recognition) and SuryaWirelessTableProcessor's own FastOCR
        # calls stack into one predict_batch() forward pass at a time --
        # shared with module/ocr/engine.py's page-level OCR cap (FastOCR's
        # own per-crop cost is much smaller than a table-structure-model
        # forward pass, so it gets its own, usually higher, cap instead of
        # reusing max_batch_size above).
        self._fastocr_max_batch_size = fastocr_max_batch_size
        # Caps how many wired-classified table crops UnetTableModel.batch_predict()
        # (monkeypatched in above) hands to TSRUnet.batch_call() at a time.
        # Its own cap, separate from max_batch_size above -- batch_call()
        # further splits this chunk into same-preprocessed-shape groups
        # before actually batching (see _tsr_batch_call's docstring for why
        # it groups instead of padding), so a document with many
        # same-shaped wired tables doesn't still hand an unbounded number
        # of them to one ONNX call within a single shape group.
        self._wired_max_batch_size = wired_max_batch_size
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
            self._fastocr,
            max_batch_size=self._max_batch_size,
            fastocr_max_batch_size=self._fastocr_max_batch_size,
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
        """FastOCR recognition for ALREADY-detected boxes -- shared by
        _run_ocr() (detects then recognizes, single crop) and
        process_batch() (detects ALL wired crops in one batched call, then
        calls this per-crop for the recognition step).

        Recognizes every box's crop in ONE FastOCR encoder+decoder forward
        pass (chunked by self._fastocr_max_batch_size), instead of one call
        per box -- predict_batch() handles the variable-width padding/masking
        correctly (see fastocr/tool/predictor.py).

        Returns list of [quad_box (4x2 list), html_escaped_text (str), score (float)].
        """
        import cv2

        if dt_boxes is None:
            return []

        boxes = [np.array(box, dtype=np.float32) for box in dt_boxes]
        # Use the SAME crop function (with its tuned diacritic padding) that
        # the shared FastOCR predictor was validated against -- MinerU's own
        # get_rotate_crop_image_for_text_rec() crops tighter and was
        # clipping Vietnamese diacritics, degrading recognition.
        crops = [
            Image.fromarray(cv2.cvtColor(self._ocr.get_rotate_crop_image(bgr, box), cv2.COLOR_BGR2RGB))
            for box in boxes
        ]

        result = []
        for start in range(0, len(crops), self._fastocr_max_batch_size):
            chunk_boxes = boxes[start : start + self._fastocr_max_batch_size]
            chunk_crops = crops[start : start + self._fastocr_max_batch_size]
            for box, (text, score) in zip(chunk_boxes, self._fastocr.predict_batch(chunk_crops)):
                result.append([box.tolist(), html_module.escape(text), score])

        return result

    def _run_ocr(self, img: np.ndarray) -> list:
        """Shared detector (via ocr.detect_sorted()) finds bbox -> shared
        FastOCR predictor reads text (single crop; process_batch() uses
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
          - The wired (UNet) structure model itself, across every
            WIRED-classified crop with a usable ocr_result, in ONE ONNX
            call -- see UnetTableModel.batch_predict() monkeypatched in
            above the top of this file. Vanilla `mineru[pipeline]` (from
            PyPI) has no batch API for this model at all; this project adds
            one via monkeypatch rather than depending on a custom fork.
        Only classification (table_cls) stays genuinely per-crop -- that
        library exposes no batch API at all.

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

        self.last_table_kinds = [
            "wire" if is_wired else "wireless" for _pn, _tno, _img, is_wired in prepared
        ]
        for (pn, tno, _img, _is_wired), kind in zip(prepared, self.last_table_kinds):
            logger.info("Page %d table %d: classified as %s", pn + 1, tno, kind)

        # Wired (UNet) structure model, batched across ALL wired-classified
        # crops with a usable ocr_result -- chunked by self._wired_max_batch_size
        # (NOT an unbounded single call: every crop here gets padded to a
        # fixed (1024,1024) canvas before the ONNX forward pass, a much
        # bigger per-item tensor than TableOrientationCls/Surya's own input,
        # so a document with many wired tables could otherwise OOM a GPU in
        # one shot). table_res["html"] is pre-filled with the wireless HTML
        # so batch_predict() only needs to OVERWRITE it for entries it
        # actually processed; entries it skips (empty ocr_result) keep
        # their wireless fallback automatically -- same semantics as the
        # old per-crop `if ocr_result: ... else: final_html = wireless_html`.
        table_res_list = [
            {
                "wired_table_img": prepared[i][2],
                "ocr_result": wired_ocr_results[i],
                "table_res": {"html": wireless_htmls[i]},
            }
            for i in wired_positions
        ]
        for start in range(0, len(table_res_list), self._wired_max_batch_size):
            chunk = table_res_list[start : start + self._wired_max_batch_size]
            self._wired.batch_predict(chunk)
        wired_html_by_pos = dict(zip(wired_positions, table_res_list))

        results = []
        for i, (pn, tno, img, is_wired) in enumerate(prepared):
            kind = self.last_table_kinds[i]
            wireless_html = wireless_htmls[i]

            if is_wired:
                ocr_result = wired_ocr_results[i]
                entry = wired_html_by_pos[i]
                final_html = entry["table_res"]["html"]
                selected_model = entry["table_res"].get("selected_model")
                if selected_model:
                    logger.info(
                        "Page %d table %d: wired-vs-wireless compare -> selected %s",
                        pn + 1, tno, selected_model,
                    )
            else:
                ocr_result = surya_ocr_results[i]
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
