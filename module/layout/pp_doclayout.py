import os
import tempfile

import cv2
import numpy as np
from PIL import Image

from .base import LayoutBlock, LayoutLabelSchema


class PPDocLayoutBackend:
    """Layout detector backed by PaddleOCR's PP-DocLayout_plus-L model.

    Replaces DeepDoc's own YOLOv10 layout model, which is no longer used.
    """

    label_schema = LayoutLabelSchema(
        table_types=frozenset({"table", "table_caption", "table_footnote"}),
        skip_types=frozenset({
            "seal", "figure", "image", "chart", "abandon",
            "figure_caption", "figure_title", "header_image", "footer_image",
            "formula", "isolate_formula", "formula_caption", "formula_number",
            "algorithm", "header", "footer", "page_number",
        }),
        title_types=frozenset({"document_title", "title"}),
        h2_types=frozenset({"paragraph_title"}),
    )

    def __init__(self, model_name: str = "PP-DocLayout_plus-L", device: str = "cpu"):
        from paddleocr import LayoutDetection
        self._model = LayoutDetection(model_name=model_name, device=device)

    def detect(self, image: Image.Image, threshold: float) -> list[LayoutBlock]:
        img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp_path = tmp.name
            cv2.imwrite(tmp_path, img_bgr)

            # paddleocr's LayoutDetection.predict() needs a filesystem path,
            # not an in-memory array.
            output = self._model.predict(tmp_path, batch_size=1, layout_nms=True)
            paddle_result = next(iter(output))
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

        if hasattr(paddle_result, "boxes"):
            boxes = paddle_result.boxes
        elif isinstance(paddle_result, dict):
            boxes = paddle_result.get("boxes", [])
        else:
            boxes = paddle_result if isinstance(paddle_result, list) else []

        blocks: list[LayoutBlock] = []
        for b in boxes:
            score = float(b.get("score", 1.0))
            if score < threshold:
                continue
            coord = b.get("coordinate") or b.get("bbox") or b.get("coord")
            if coord is None:
                continue
            x0, y0, x1, y1 = float(coord[0]), float(coord[1]), float(coord[2]), float(coord[3])
            blocks.append({
                "type": b.get("label", "text"),
                "bbox": [x0, y0, x1, y1],
                "score": score,
            })
        return blocks
