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

    All `mineru` imports needed JUST for this table backend are lazy (inside
    methods, not at module import time) so `module.table` stays importable
    -- and the "tsr" table backend still works -- on installs that don't
    have `mineru` and never select `table.backend: mineru` (independent of
    whatever `ocr.det_backend` is set to).
    """

    def __init__(self, ocr: OCREngine):
        from mineru.backend.pipeline.model_init import AtomModelSingleton

        self._ocr = ocr
        self._text_det = ocr.text_detector[0]
        self._vietocr = ocr.text_recognizer[0].detector
        self._atom = AtomModelSingleton()
        # Set by __call__ once TableCls has run -- "wire" or "wireless".
        # Read by document_pipeline.py (via getattr, since only this backend
        # has the concept) to suffix the deskewed-crop debug filename.
        self.last_table_kind: str | None = None
        self._load_table_models()

    def _load_table_models(self) -> None:
        from mineru.backend.pipeline.model_list import AtomicModel
        from table_cls import TableCls

        self._AtomicModel = AtomicModel
        self._ori_cls = self._atom.get_atom_model(atom_model_name=AtomicModel.TableOrientationCls)
        self._table_cls = TableCls(model_type="q")  # RapidAI QAnything wired/wireless classifier
        self._wired = self._atom.get_atom_model(atom_model_name=AtomicModel.WiredTable, lang=None)
        self._surya = SuryaWirelessTableProcessor(self._vietocr)

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

    def _run_ocr(self, img: np.ndarray) -> list:
        """Shared det.onnx detects bbox -> shared VietOCR predictor reads text.

        Returns list of [quad_box (4x2 list), html_escaped_text (str), score (float)].
        """
        import cv2

        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        dt_boxes, _ = self._text_det(bgr)
        if dt_boxes is None or len(dt_boxes) == 0:
            return []
        # MinerU's wireless-table matcher (matcher.py::match_result) appends
        # OCR fragments to a cell in WHATEVER order dt_boxes arrives in --
        # no positional re-sort of its own (unlike the wired/UNet path,
        # which does sort_and_gather_ocr_res internally). Raw detector
        # output order isn't reading order, so without this the OCR result
        # order silently becomes the final cell-text order once a cell
        # holds more than one fragment.
        dt_boxes = self._ocr.sorted_boxes(dt_boxes)

        result = []
        for box in dt_boxes:
            box = np.array(box, dtype=np.float32)
            # Use the SAME crop function (with its tuned diacritic padding)
            # that the shared VietOCR predictor was validated against --
            # MinerU's own get_rotate_crop_image_for_text_rec() crops tighter
            # and was clipping Vietnamese diacritics, degrading recognition.
            crop_bgr = self._ocr.get_rotate_crop_image(bgr, box)
            crop_pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
            text, score = self._vietocr.predict(crop_pil, return_prob=True)
            result.append([box.tolist(), html_module.escape(text), score])

        return result

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
