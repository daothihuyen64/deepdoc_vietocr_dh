from ..ocr import OCREngine
from .base import TableProcessor


def get_table_processor(conf: dict, ocr: OCREngine) -> TableProcessor:
    # Concrete processors are imported lazily, INSIDE this function, not at
    # module top level: tsr_processor.py reaches into ..pipeline.table /
    # ..pipeline.debug, and ..pipeline imports THIS package (module.table)
    # from document_pipeline.py -- an eager top-level import here would be a
    # circular import that only "worked" depending on which module happened
    # to be imported first.
    table_conf = conf.get("table", {})
    backend = table_conf.get("backend", "tsr")
    if backend == "tsr":
        from ..tsr import get_tsr_backend
        from .tsr_processor import TSRTableProcessor
        return TSRTableProcessor(
            get_tsr_backend(conf), ocr,
            tsr_threshold=conf.get("tsr", {}).get("threshold", 0.2),
        )
    if backend == "mineru":
        from .mineru_processor import MinerUTableProcessor
        tsr_threshold = conf.get("tsr", {}).get("threshold", 0.2)
        # Only load the TSR model when debug output is on -- it's used
        # exclusively as a manual side-by-side comparison for wireless
        # tables (see MinerUTableProcessor), never for the real result, so
        # a production run with debug off shouldn't pay to load it at all.
        if conf.get("debug", {}).get("enabled", False):
            from ..tsr import get_tsr_backend
            return MinerUTableProcessor(ocr, tsr=get_tsr_backend(conf), tsr_threshold=tsr_threshold)
        return MinerUTableProcessor(ocr)
    raise ValueError(f"Unknown table backend: {backend!r}")


__all__ = [
    "TableProcessor",
    "get_table_processor",
]
