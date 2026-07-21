from typing import Protocol, runtime_checkable

from PIL import Image


@runtime_checkable
class TSRBackend(Protocol):
    """Contract a swappable Table Structure Recognition backend must satisfy."""

    def __call__(self, images: list[Image.Image], thr: float = 0.2) -> list[list[dict]]: ...
