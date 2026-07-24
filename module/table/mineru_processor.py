import html as html_module
import logging

import numpy as np
from PIL import Image

from ..ocr import OCREngine
from ..pipeline.debug import save_table_backend_compare, save_table_ocr_debug
from ..pipeline.table import table_to_markdown
from ..tsr import TSRBackend

logger = logging.getLogger(__name__)

# MinerU's UNet wired-table cell recovery joins the OCR text fragments
# matched to the same cell with no separator, which drops the space at
# fragment boundaries within a cell (e.g. "Ngành" + "nông," -> "Ngànhnông,").
# Join with a space instead.
import mineru.model.table.rec.unet_table.utils_table_recover as _unet  # noqa: E402
_unet._cell_text = lambda m, i: " ".join(m.get(i, []))


class MinerUTableProcessor:
    """Table backend built on MinerU's model stack: orientation fix ->
    wired/wireless classify -> OCR -> table-structure model (SLANet-Plus
    wireless, always; UNet wired, for wired tables -- internally compares
    both and keeps whichever HTML has better cell/text coverage) -> HTML.

    Detection AND recognition reuse the shared `OCREngine` entirely --
    `ocr.text_detector[0]` (whatever `ocr.det_backend` resolves to: the onnx
    detector by default, or MinerU's own PP-OCRv6 if configured) and
    `ocr.text_recognizer[0].detector` (the VietOCR predictor module/ocr
    already loaded, same weight file) -- instead of loading a second VietOCR
    copy, so table detection always stays consistent with whatever detector
    page-level OCR is using. Only the table-structure/orientation/
    wired-vs-wireless models are unconditionally MinerU's own.

    All `mineru` imports needed JUST for this table backend are lazy (inside
    methods, not at module import time) so `module.table` stays importable
    -- and the "tsr" table backend still works -- on installs that don't
    have `mineru` and never select `table.backend: mineru` (independent of
    whatever `ocr.det_backend` is set to).

    `tsr`, when given, is used ONLY as a side-by-side comparison for manual
    quality review -- run and dumped to debug_dir alongside the real
    (mineru) result whenever a table comes back wireless, never fed into
    the returned content. `get_table_processor()` only constructs one when
    debug is enabled, so production runs don't pay for a TSR model they
    never look at.
    """

    def __init__(self, ocr: OCREngine, tsr: TSRBackend | None = None, tsr_threshold: float = 0.2):
        from mineru.backend.pipeline.model_init import AtomModelSingleton

        self._ocr = ocr
        self._text_det = ocr.text_detector[0]
        self._vietocr = ocr.text_recognizer[0].detector
        self._tsr = tsr
        self._tsr_threshold = tsr_threshold
        self._atom = AtomModelSingleton()
        self._load_table_models()

    def _load_table_models(self) -> None:
        from mineru.backend.pipeline.model_list import AtomicModel

        self._AtomicModel = AtomicModel
        self._ori_cls = self._atom.get_atom_model(atom_model_name=AtomicModel.TableOrientationCls)
        self._table_cls = self._atom.get_atom_model(atom_model_name=AtomicModel.TableCls)
        self._wireless = self._atom.get_atom_model(atom_model_name=AtomicModel.WirelessTable, lang=None)
        self._wired = self._atom.get_atom_model(atom_model_name=AtomicModel.WiredTable, lang=None)

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
        label, score = self._table_cls.predict(img)
        return label == self._AtomicModel.WiredTable or score < 0.9

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
        logger.info("Page %d table %d: classified as %s", pn + 1, tno, "wired" if is_wired else "wireless")

        ocr_result = self._run_ocr(img)
        logger.debug("OCR found %d text boxes", len(ocr_result))
        if debug_dir:
            save_table_ocr_debug(
                Image.fromarray(img),
                [(box, (text, score)) for box, text, score in ocr_result],
                debug_dir, pn, tno,
            )

        # Wireless model always runs (baseline). Wired model runs only for
        # wired tables and internally picks whichever HTML is better.
        wireless_html, *_ = self._wireless.predict(img, ocr_result or None)
        if is_wired and ocr_result:
            final_html = self._wired.predict(img, ocr_result, wireless_html)
        else:
            final_html = wireless_html
            if debug_dir and self._tsr is not None:
                logger.info("Page %d table %d: wireless -> writing tsr comparison", pn + 1, tno)
                self._save_tsr_compare(img, ocr_result, wireless_html, debug_dir, pn, tno)
            elif debug_dir:
                logger.info("Page %d table %d: wireless but no tsr backend loaded (debug was off at startup?) -- skipping comparison", pn + 1, tno)

        return self._trim_to_table_tag(final_html or wireless_html)

    def _save_tsr_compare(
        self,
        img: np.ndarray,
        ocr_result: list,
        mineru_html: str,
        debug_dir: str,
        pn: int,
        tno: int,
    ) -> None:
        """Runs the "tsr" backend on the SAME crop + OCR result that just
        went through mineru's wireless model, purely so the two outputs can
        be eyeballed side by side -- never used for the real block content."""
        crop_img = Image.fromarray(img)
        tsr_result = self._tsr([crop_img], thr=self._tsr_threshold)
        cpns = tsr_result[0] if tsr_result else []
        raw = [(box, (text, score)) for box, text, score in ocr_result]
        tsr_content = table_to_markdown(crop_img, cpns, self._ocr, raw=raw)
        save_table_backend_compare(mineru_html or "", tsr_content, debug_dir, pn, tno)
