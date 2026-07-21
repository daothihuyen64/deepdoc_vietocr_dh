from .base import LayoutBackend, LayoutBlock, LayoutLabelSchema
from .pp_doclayout import PPDocLayoutBackend


def _auto_layout_device() -> str:
    """Same GPU/CPU auto-detection the OCR/TSR onnxruntime sessions and
    VietOCR already use -- lets one codebase run on either a CPU-only dev
    machine or a rented GPU box without touching config, unless the config
    explicitly pins a device."""
    try:
        import torch
        return "gpu:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def get_layout_backend(conf: dict) -> LayoutBackend:
    layout_conf = conf.get("layout", {})
    backend = layout_conf.get("backend", "pp_doclayout")
    if backend == "pp_doclayout":
        return PPDocLayoutBackend(
            model_name=layout_conf.get("model_name", "PP-DocLayout_plus-L"),
            device=layout_conf.get("device") or _auto_layout_device(),
        )
    raise ValueError(f"Unknown layout backend: {backend!r}")


__all__ = [
    "LayoutBackend",
    "LayoutBlock",
    "LayoutLabelSchema",
    "PPDocLayoutBackend",
    "get_layout_backend",
]
