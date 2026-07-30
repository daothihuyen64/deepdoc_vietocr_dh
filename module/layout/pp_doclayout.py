import logging
import os
import tempfile

import cv2
import numpy as np
from PIL import Image

from .base import LayoutBlock, LayoutLabelSchema

logger = logging.getLogger(__name__)


class PPDocLayoutBackend:
    """Layout detector backed by PaddleOCR's PP-DocLayout_plus-L model.

    Replaces DeepDoc's own YOLOv10 layout model, which is no longer used.
    """

    # The 20 real labels PP-DocLayout_plus-L outputs (from its own
    # inference.yml label_list, ground truth -- not the model's own docs
    # summary, which is looser): paragraph_title, image, text, number,
    # abstract, content, figure_title, formula, table, reference, doc_title,
    # footnote, header, algorithm, footer, seal, chart, formula_number,
    # aside_text, reference_content. Anything below not in this list is a
    # typo/dead entry that can never match real model output.
    label_schema = LayoutLabelSchema(
        table_types=frozenset({"table"}),
        skip_types=frozenset({
            "seal", "image", "chart", "formula", "formula_number",
            "algorithm", "number", "aside_text", "footnote",
        }),
        title_types=frozenset({"doc_title"}),
        h2_types=frozenset({"paragraph_title"}),
    )

    def __init__(self, model_name: str = "PP-DocLayout_plus-L", device: str = "cpu", max_batch_size: int = 8):
        from paddleocr import LayoutDetection
        self._model = LayoutDetection(model_name=model_name, device=device)
        logger.info("[device] Layout (%s): device=%s", model_name, device)
        # Caps how many images PaddleX actually stacks into one forward
        # pass at a time -- `predict()`'s own `batch_size` arg handles this
        # internally (still returns one result per input image, in order,
        # just chunked under the hood), so a document with far more pages
        # than fit in GPU memory at once doesn't OOM just because we handed
        # it every page in one detect_batch() call.
        self._max_batch_size = max_batch_size

    def detect(self, image: Image.Image, threshold: float) -> list[LayoutBlock]:
        return self.detect_batch([image], threshold)[0]

    def detect_batch(self, images: list[Image.Image], threshold: float) -> list[list[LayoutBlock]]:
        """Runs layout detection for MULTIPLE page images, grouping them by
        (width, height) first and batching each same-shaped group in its
        own PaddleX predict() call.

        Stacking DIFFERENT-shaped images into one batch tensor forces
        PaddleX to resize/pad every image in that batch to a shared shape
        -- a document with mixed portrait/landscape pages (e.g. some pages
        scanned sideways) would otherwise waste real compute padding
        smaller pages up to match the largest one in the batch, for no
        benefit. Grouping first means each predict() call only ever
        contains genuinely same-shaped images, so no page pays for another
        page's shape.

        Returns one list[LayoutBlock] per input image, in the SAME ORDER
        as `images` (PaddleX's predict() is documented to preserve input
        order WITHIN a call -- a mismatched return length for any group is
        treated as a hard error below rather than silently mis-aligning
        pages to the wrong layout result).
        """
        if not images:
            return []

        groups: dict[tuple[int, int], list[int]] = {}
        for idx, image in enumerate(images):
            groups.setdefault(image.size, []).append(idx)

        results: list[list[LayoutBlock] | None] = [None] * len(images)
        for indices in groups.values():
            group_images = [images[i] for i in indices]
            group_blocks = self._detect_batch_same_shape(group_images, threshold)
            for idx, blocks in zip(indices, group_blocks):
                results[idx] = blocks

        return results  # every index was filled by exactly one group above

    def _detect_batch_same_shape(self, images: list[Image.Image], threshold: float) -> list[list[LayoutBlock]]:
        """The actual batched predict() call -- ONLY for images the caller
        (detect_batch) has already confirmed share the same (width, height),
        so no cross-image padding waste happens inside this call."""
        tmp_paths: list[str] = []
        try:
            for image in images:
                img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                # paddleocr's LayoutDetection.predict() needs filesystem
                # paths, not in-memory arrays.
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    tmp_path = tmp.name
                cv2.imwrite(tmp_path, img_bgr)
                tmp_paths.append(tmp_path)

            batch_size = min(len(tmp_paths), self._max_batch_size)
            output = self._model.predict(tmp_paths, batch_size=batch_size, layout_nms=True)
            paddle_results = list(output)
        finally:
            for tmp_path in tmp_paths:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        if len(paddle_results) != len(images):
            raise RuntimeError(
                f"PP-DocLayout batch predict returned {len(paddle_results)} results "
                f"for {len(images)} input images -- refusing to guess how they line up."
            )

        return [self._parse_boxes(paddle_result, threshold) for paddle_result in paddle_results]

    @staticmethod
    def _parse_boxes(paddle_result, threshold: float) -> list[LayoutBlock]:
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
