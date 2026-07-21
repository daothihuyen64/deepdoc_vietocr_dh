from .base import TSRBackend
from .engine import TableStructureRecognizer


def get_tsr_backend(conf: dict) -> TableStructureRecognizer:
    return TableStructureRecognizer()


__all__ = [
    "TableStructureRecognizer",
    "TSRBackend",
    "get_tsr_backend",
]
