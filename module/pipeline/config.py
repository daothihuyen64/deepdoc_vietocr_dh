from dataclasses import dataclass

from utils import read_config


def load_pipeline_conf() -> dict:
    """Loads conf/pipeline_conf.yaml, merged with conf/local.pipeline_conf.yaml if present."""
    return read_config("pipeline_conf.yaml")


@dataclass(frozen=True)
class PipelineConfig:
    pdf_dpi: int
    layout_threshold: float
    tsr_threshold: float
    map_overlap_threshold: float
    figure_drop_types: frozenset[str]
    figure_drop_ratio: float
    line_overlap_min: float
    single_page_ratio_max: float
    spanning_min_ratio: float
    anchor_min_width: float
    anchor_band_mult: float
    output_dir: str
    debug_enabled: bool
    debug_dir: str
    crop_whitespace_enabled: bool
    crop_whitespace_threshold: int
    crop_whitespace_dilate_kernel: tuple[int, int]
    crop_whitespace_dilate_iter: int
    crop_whitespace_margin: int
    crop_whitespace_open_kernel: tuple[int, int]
    crop_whitespace_min_contour_area_ratio: float
    crop_whitespace_max_content_area_ratio: float
    page_deskew_enabled: bool
    page_deskew_min_boxes: int
    page_deskew_angle_threshold: float
    page_deskew_max_angle: float
    page_orientation_enabled: bool
    page_orientation_score_threshold: float
    page_orientation_sample_max: int
    page_orientation_min_scores: int
    page_orientation_sideways_min_count: int
    page_orientation_sideways_min_ratio: float

    @classmethod
    def from_conf(cls, conf: dict) -> "PipelineConfig":
        mapping = conf.get("mapping", {})
        reading_order = conf.get("reading_order", {})
        debug = conf.get("debug", {})
        crop_whitespace = conf.get("crop_whitespace", {})
        page_deskew = conf.get("page_deskew", {})
        page_orientation = conf.get("page_orientation", {})
        return cls(
            pdf_dpi=conf.get("pdf", {}).get("dpi", 200),
            layout_threshold=conf.get("layout", {}).get("threshold", 0.2),
            tsr_threshold=conf.get("tsr", {}).get("threshold", 0.2),
            map_overlap_threshold=mapping.get("overlap_threshold", 0.5),
            figure_drop_types=frozenset(mapping.get("figure_drop_types", ["image"])),
            figure_drop_ratio=mapping.get("figure_drop_ratio", 0.4),
            line_overlap_min=mapping.get("line_overlap_min", 0.5),
            single_page_ratio_max=reading_order.get("single_page_ratio_max", 0.85),
            spanning_min_ratio=reading_order.get("spanning_min_ratio", 0.15),
            anchor_min_width=reading_order.get("anchor_min_width", 100),
            anchor_band_mult=reading_order.get("anchor_band_mult", 1.0),
            output_dir=conf.get("output", {}).get("dir", "./pipeline_outputs"),
            debug_enabled=debug.get("enabled", False),
            debug_dir=debug.get("dir", "./debug_outputs"),
            crop_whitespace_enabled=crop_whitespace.get("enabled", True),
            crop_whitespace_threshold=crop_whitespace.get("threshold", 200),
            crop_whitespace_dilate_kernel=tuple(crop_whitespace.get("dilate_kernel", [15, 15])),
            crop_whitespace_dilate_iter=crop_whitespace.get("dilate_iter", 3),
            crop_whitespace_margin=crop_whitespace.get("margin", 15),
            crop_whitespace_open_kernel=tuple(crop_whitespace.get("open_kernel", [3, 3])),
            crop_whitespace_min_contour_area_ratio=crop_whitespace.get("min_contour_area_ratio", 0.0008),
            crop_whitespace_max_content_area_ratio=crop_whitespace.get("max_content_area_ratio", 0.5),
            page_deskew_enabled=page_deskew.get("enabled", False),
            page_deskew_min_boxes=page_deskew.get("min_boxes", 5),
            page_deskew_angle_threshold=page_deskew.get("angle_threshold", 0.5),
            page_deskew_max_angle=page_deskew.get("max_angle", 10.0),
            page_orientation_enabled=page_orientation.get("enabled", False),
            page_orientation_score_threshold=page_orientation.get("score_threshold", 0.8),
            page_orientation_sample_max=page_orientation.get("sample_max", 20),
            page_orientation_min_scores=page_orientation.get("min_scores", 5),
            page_orientation_sideways_min_count=page_orientation.get("sideways_min_count", 4),
            page_orientation_sideways_min_ratio=page_orientation.get("sideways_min_ratio", 0.3),
        )
