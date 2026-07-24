import os

from PIL import Image

from ..ocr import OCREngine
from ..pipeline.debug import save_table_ocr_debug
from ..pipeline.table import raw_ocr, table_to_markdown
from ..tsr import TSRBackend


class TSRTableProcessor:
    """Table backend built on DeepDoc's own Table Structure Recognition model:
    TSR components + a raw OCR pass over the crop are glued into a Markdown
    table via table_to_markdown()/construct_table()."""

    def __init__(self, tsr: TSRBackend, ocr: OCREngine, tsr_threshold: float = 0.2):
        self.tsr = tsr
        self.ocr = ocr
        self.tsr_threshold = tsr_threshold

    def __call__(
        self,
        crop: Image.Image,
        debug_dir: str | None = None,
        pn: int = 0,
        tno: int = 0,
    ) -> str:
        tsr_result = self.tsr([crop], thr=self.tsr_threshold)
        cpns = tsr_result[0] if tsr_result else []

        crop_debug_dir = (
            os.path.join(debug_dir, f"page_{pn + 1}_table_{tno}_vietocr_crops")
            if debug_dir else None
        )
        raw_ocr_crop = raw_ocr(crop, self.ocr, crop_debug_dir=crop_debug_dir)
        if debug_dir:
            save_table_ocr_debug(crop, raw_ocr_crop, debug_dir, pn, tno)

        return table_to_markdown(crop, cpns, self.ocr, raw=raw_ocr_crop)
