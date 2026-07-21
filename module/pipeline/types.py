from typing import TypedDict


class OCRBox(TypedDict, total=False):
    bbox: list[float]
    bbox_row: list[float]
    quad: list[list[float]]
    text: str


class PageBlock(TypedDict, total=False):
    type: str
    bbox: list[float]
    score: float
    content_type: str
    content: str | None
    text_items: list[OCRBox]
