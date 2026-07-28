import html as html_module
import logging

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class SuryaWirelessTableProcessor:
    """Wireless-table pipeline: Surya's `TableRecPredictor` (table structure)
    + Surya's own `DetectionPredictor` (text-line bbox detection) + the
    shared VietOCR predictor (recognition) -- mirrors
    `surya_v1_table_to_html_vietocr.ipynb` exactly, per explicit user
    request to use Surya's own detector for this path specifically (unlike
    the rest of this file, which reuses the project's shared detector
    everywhere else -- this is the one deliberate exception).

    Recognition still reuses the shared VietOCR predictor (passed in, not
    loaded again) instead of Surya's own recognizer, since VietOCR reads
    Vietnamese diacritics more accurately.
    """

    def __init__(self, vietocr_predictor) -> None:
        from surya.detection import DetectionPredictor
        from surya.table_rec import TableRecPredictor

        self._det = DetectionPredictor()
        self._table_rec = TableRecPredictor()
        self._vietocr = vietocr_predictor
        # Set by __call__ -- the [quad_box, text, score] triples this
        # instance's OWN detector+recognizer just produced, for
        # MinerUTableProcessor to reuse for the debug OCR overlay instead of
        # recomputing (and instead of showing the WRONG, shared-detector
        # boxes, which this path no longer uses).
        self.last_ocr_result: list = []

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

    def __call__(self, img: np.ndarray) -> str:
        pil_img = Image.fromarray(img).convert("RGB")

        table_pred = self._table_rec([pil_img])[0]
        cells = table_pred.cells

        det_pred = self._det([pil_img])[0]
        ocr_result = []
        for box in det_pred.bboxes:
            crop = self._crop_with_margin(pil_img, box.bbox)
            if crop.size[0] < 2 or crop.size[1] < 2:
                continue
            text, score = self._vietocr.predict(crop, return_prob=True)
            x1, y1, x2, y2 = box.bbox
            # Keep the same [4-point quad, text, score] triple shape used
            # everywhere else in this file, even though Surya's own bboxes
            # are already axis-aligned rects -- _quad_to_bbox below reduces
            # it right back, but this way ocr_result's shape never depends
            # on which detector produced it.
            quad = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            ocr_result.append([quad, html_module.escape(text), score])

        self.last_ocr_result = ocr_result

        cell_text_map = self._fill_cell_texts(cells, ocr_result)
        return self._build_html_table(cells, cell_text_map)
