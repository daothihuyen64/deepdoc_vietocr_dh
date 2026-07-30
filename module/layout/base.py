from dataclasses import dataclass
from typing import Protocol, TypedDict, runtime_checkable

from PIL import Image


@dataclass(frozen=True)
class LayoutLabelSchema:
    """Label taxonomy owned by a specific layout backend.

    Different layout models use entirely different label vocabularies (e.g.
    DeepDoc's old YOLOv10 model used 10 labels, PP-DocLayout uses 23), so the
    table/skip/title/h2 classification must travel with the backend that
    produced the labels, not live as pipeline-level global constants.
    """

    table_types: frozenset[str]
    skip_types: frozenset[str]
    title_types: frozenset[str]
    h2_types: frozenset[str]


class LayoutBlock(TypedDict):
    type: str
    bbox: list[float]
    score: float


@runtime_checkable
class LayoutBackend(Protocol):
    """Contract a swappable layout-detection backend must satisfy."""

    label_schema: LayoutLabelSchema

    def detect(self, image: Image.Image, threshold: float) -> list[LayoutBlock]: ...
    def detect_batch(self, images: list[Image.Image], threshold: float) -> list[list[LayoutBlock]]: ...
