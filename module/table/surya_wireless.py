import html as html_module
import logging
import threading

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class SuryaWirelessTableProcessor:
    """Wireless-table pipeline: Surya's `TableRecPredictor` (table structure)
    + Surya's own `DetectionPredictor` (text-line bbox detection) + the
    shared FastOCR predictor (recognition) -- mirrors
    `surya_v1_table_to_html_fastocr.ipynb` exactly, per explicit user
    request to use Surya's own detector for this path specifically (unlike
    the rest of this file, which reuses the project's shared detector
    everywhere else -- this is the one deliberate exception).

    Recognition still reuses the shared FastOCR predictor (passed in, not
    loaded again) instead of Surya's own recognizer, since FastOCR reads
    Vietnamese diacritics more accurately.
    """

    def __init__(self, fastocr_predictor, max_batch_size: int = 8, fastocr_max_batch_size: int = 64) -> None:
        from surya.detection import DetectionPredictor
        from surya.table_rec import TableRecPredictor

        self._det = DetectionPredictor()
        self._table_rec = TableRecPredictor()
        self._fastocr = fastocr_predictor
        # Caps how many crops call_batch() stacks into one TableRecPredictor/
        # DetectionPredictor forward pass at a time -- a document with many
        # wireless tables would otherwise hand every single one to one
        # unbounded batch call and risk a GPU OOM.
        self._max_batch_size = max_batch_size
        # Caps how many text-line crops call_batch() stacks into one FastOCR
        # predict_batch() forward pass at a time -- FastOCR's own per-crop
        # cost is much smaller than TableRecPredictor/DetectionPredictor's,
        # so it gets its own (usually higher) cap instead of reusing
        # max_batch_size above.
        self._fastocr_max_batch_size = fastocr_max_batch_size
        # This instance (and _det/_table_rec) is shared across every request
        # the server handles concurrently. Confirmed via a real crash
        # (RuntimeError: mismatched attention key/value shapes inside
        # TableRecPredictor's decoder) that these models are NOT safe to call
        # from multiple threads at once -- likely internal autoregressive/
        # KV-cache state on the model getting corrupted by interleaved calls.
        # Separate locks (not one shared lock) so a request's _det call and
        # another request's _table_rec call can still overlap.
        self._det_lock = threading.Lock()
        self._table_rec_lock = threading.Lock()

    @staticmethod
    def _crop_with_margin(image: Image.Image, bbox, margin: int = 3) -> Image.Image:
        x1, y1, x2, y2 = bbox
        w, h = image.size
        x1 = max(0, int(x1) - margin)
        y1 = max(0, int(y1) - margin)
        x2 = min(w, int(x2) + margin)
        y2 = min(h, int(y2) + margin)
        return image.crop((x1, y1, x2, y2))

    @staticmethod
    def _bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @staticmethod
    def _point_in_bbox(point: tuple[float, float], bbox: tuple[float, float, float, float], margin: float = 2) -> bool:
        px, py = point
        x1, y1, x2, y2 = bbox
        return (x1 - margin) <= px <= (x2 + margin) and (y1 - margin) <= py <= (y2 + margin)

    @classmethod
    def _quad_to_bbox(cls, quad) -> tuple[float, float, float, float]:
        """Our detector returns a 4-point quad; Surya's cells use axis-aligned
        (x1, y1, x2, y2) -- reduce the quad to its bounding rect."""
        xs = [p[0] for p in quad]
        ys = [p[1] for p in quad]
        return (min(xs), min(ys), max(xs), max(ys))

    def _fill_cell_texts(self, cells, ocr_result: list) -> dict[int, str]:
        lines = []
        for box, text, _score in ocr_result:
            lines.append((self._quad_to_bbox(box), text))

        cell_texts: dict[int, list[tuple[tuple[float, float, float, float], str]]] = {i: [] for i in range(len(cells))}
        for bbox, text in lines:
            center = self._bbox_center(bbox)
            best_idx, best_area = None, None
            for i, cell in enumerate(cells):
                if self._point_in_bbox(center, cell.bbox):
                    x1, y1, x2, y2 = cell.bbox
                    area = (x2 - x1) * (y2 - y1)
                    if best_area is None or area < best_area:
                        best_idx, best_area = i, area
            if best_idx is not None:
                cell_texts[best_idx].append((bbox, text))

        result = {}
        for i, entries in cell_texts.items():
            entries_sorted = sorted(entries, key=lambda e: (e[0][1], e[0][0]))
            result[i] = " ".join(text.strip() for _bbox, text in entries_sorted if text.strip())
        return result

    @staticmethod
    def _build_html_table(cells, cell_text_map: dict[int, str]) -> str:
        if not cells:
            return "<table></table>"

        num_rows = max(c.row_id + c.rowspan for c in cells)
        num_cols = max(c.col_id + c.colspan for c in cells)

        occupied = [[False] * num_cols for _ in range(num_rows)]
        origin_map = {}
        for i, cell in enumerate(cells):
            origin_map[(cell.row_id, cell.col_id)] = i
            for r in range(cell.row_id, cell.row_id + cell.rowspan):
                for c in range(cell.col_id, cell.col_id + cell.colspan):
                    if r < num_rows and c < num_cols:
                        occupied[r][c] = "origin" if (r, c) == (cell.row_id, cell.col_id) else True

        html_rows = []
        for r in range(num_rows):
            row_cells_html = []
            c = 0
            while c < num_cols:
                cell_state = occupied[r][c]
                if cell_state == "origin":
                    i = origin_map[(r, c)]
                    cell = cells[i]
                    # ocr_result's text fragments are already html-escaped
                    # (see __call__ above) -- do not escape again here.
                    text = cell_text_map.get(i, "")
                    tag = "th" if cell.is_header else "td"
                    attrs = ""
                    if cell.rowspan > 1:
                        attrs += f' rowspan="{cell.rowspan}"'
                    if cell.colspan > 1:
                        attrs += f' colspan="{cell.colspan}"'
                    row_cells_html.append(f"<{tag}{attrs}>{text}</{tag}>")
                    c += cell.colspan
                elif cell_state is True:
                    c += 1
                else:
                    row_cells_html.append("<td></td>")
                    c += 1
            html_rows.append("  <tr>\n    " + "\n    ".join(row_cells_html) + "\n  </tr>")

        return '<table border="1" cellspacing="0" cellpadding="4">\n' + "\n".join(html_rows) + "\n</table>"

    def __call__(self, img: np.ndarray) -> tuple[str, list]:
        htmls, ocr_results = self.call_batch([img])
        return htmls[0], ocr_results[0]

    def call_batch(self, imgs: list[np.ndarray]) -> tuple[list[str], list[list]]:
        """Batches `TableRecPredictor` + `DetectionPredictor` across
        MULTIPLE table crops in ONE call each, instead of __call__()'s
        one-crop-at-a-time path.

        Both Surya predictors already do their OWN internal chunking
        (`batch_size=` kwarg on `__call__`, defaulting to
        `settings.TABLE_REC_BATCH_SIZE`/`DETECTOR_BATCH_SIZE` if omitted --
        verified from source, `surya/table_rec/__init__.py`'s
        `batch_table_recognition()` and `surya/detection/__init__.py`'s
        `batch_detection()` both slice their own `images` list in a loop),
        AND resize every image to a fixed canonical size independent of
        whatever else is in the same call (`DetectionPredictor` even
        splits tall images into fixed-height pieces and packs the batch by
        a "how many pieces fit" budget, not a raw image count) -- so unlike
        the layout backend (PaddleX's LayoutDetection, which does NOT do
        any of this itself, see PPDocLayoutBackend.detect_batch), there's
        nothing for US to chunk or size-group here: just pass
        `self._max_batch_size` straight through as Surya's own `batch_size`.

        FastOCR recognition batches every detected text line across ALL
        images in this call into one flat list first (chunked by
        self._fastocr_max_batch_size), instead of one predict() call per
        line -- predict_batch() handles the variable-width padding/masking
        correctly (see fastocr/tool/predictor.py).

        Returns `(html_list, ocr_results_per_image)` -- one HTML string and
        one ocr_result list per input image, both in the SAME ORDER as
        `imgs`. Returned directly (not stashed on `self`) because this
        instance is shared across every request the server handles
        concurrently -- storing "the last call's result" as instance state
        would let one request's data leak into another's if two calls
        overlap.
        """
        if not imgs:
            return [], []

        pil_imgs = [Image.fromarray(img).convert("RGB") for img in imgs]
        with self._table_rec_lock:
            table_preds = self._table_rec(pil_imgs, batch_size=self._max_batch_size)
        with self._det_lock:
            det_preds = self._det(pil_imgs, batch_size=self._max_batch_size)

        # Flatten every detected text-line crop across ALL images into one
        # list first, so FastOCR recognizes the whole document's table
        # crops in as few batched forward passes as possible, instead of
        # per-image.
        flat_crops: list[Image.Image] = []
        flat_quads: list[list] = []
        owner_idx: list[int] = []
        for i, (pil_img, det_pred) in enumerate(zip(pil_imgs, det_preds)):
            for box in det_pred.bboxes:
                crop = self._crop_with_margin(pil_img, box.bbox)
                if crop.size[0] < 2 or crop.size[1] < 2:
                    continue
                x1, y1, x2, y2 = box.bbox
                flat_crops.append(crop)
                # Keep the same [4-point quad, text, score] triple shape
                # used everywhere else in this file, even though Surya's
                # own bboxes are already axis-aligned rects -- _quad_to_bbox
                # reduces it right back, but this way ocr_result's shape
                # never depends on which detector produced it.
                flat_quads.append([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
                owner_idx.append(i)

        flat_texts_scores = []
        for start in range(0, len(flat_crops), self._fastocr_max_batch_size):
            chunk = flat_crops[start : start + self._fastocr_max_batch_size]
            flat_texts_scores.extend(self._fastocr.predict_batch(chunk))

        ocr_results_per_image: list[list] = [[] for _ in imgs]
        for owner, quad, (text, score) in zip(owner_idx, flat_quads, flat_texts_scores):
            ocr_results_per_image[owner].append([quad, html_module.escape(text), score])

        results = []
        for table_pred, ocr_result in zip(table_preds, ocr_results_per_image):
            cell_text_map = self._fill_cell_texts(table_pred.cells, ocr_result)
            results.append(self._build_html_table(table_pred.cells, cell_text_map))
        return results, ocr_results_per_image
