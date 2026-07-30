from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class OCREngine(Protocol):
    """Contract a swappable OCR backend must satisfy.

    Mirrors what `module/pipeline` actually touches today (per-device
    detector/recognizer lists + the couple of helper methods used to build
    rotated crops and sort boxes) rather than a single high-level call, since
    the pipeline layer needs det/rec access separately (e.g. to run
    detection once per page and recognition on custom crop batches).
    """

    text_detector: list[Any]
    text_recognizer: list[Any]
    drop_score: float

    def sorted_boxes(self, dt_boxes: np.ndarray) -> list: ...

    def get_rotate_crop_image(self, img: np.ndarray, points: np.ndarray) -> np.ndarray: ...

    def detect_raw(self, img_bgr: np.ndarray, device_id: int | None = None) -> np.ndarray | None: ...

    def detect_sorted(self, img_bgr: np.ndarray, device_id: int | None = None) -> np.ndarray | None: ...

    def detect_raw_batch(self, img_list: list[np.ndarray], device_id: int | None = None) -> list[np.ndarray | None]: ...

    def detect_sorted_batch(self, img_list: list[np.ndarray], device_id: int | None = None) -> list[np.ndarray | None]: ...

    def __call__(self, img: np.ndarray, device_id: int = 0, cls: bool = True, crop_debug_dir: str | None = None) -> list: ...
