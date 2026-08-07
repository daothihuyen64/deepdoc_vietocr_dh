import asyncio
import gc
import logging
import os
import tempfile
import time
from functools import partial

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pdf2image import pdfinfo_from_path
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError,
)

from module.pipeline import DocumentPipeline

from .deps import get_pipeline
from .schemas import BatchItemResult, BatchOCRResponse, OCRResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/ocr", tags=["ocr"])


def _release_memory() -> None:
    """Best-effort end-of-request cleanup -- run in `finally` by every route
    below, regardless of success/failure, so a request's own working set
    (rendered pages, layout/OCR/table intermediates) doesn't linger in RAM
    or GPU memory once the response has been built. Complements the
    per-model-call empty_cache() already in fastocr/tool/predictor.py and
    module/layout/pp_doclayout.py -- those only return GPU memory right
    after each individual model call; this catches whatever's left (Python
    reference cycles gc.collect() cleans up, and any GPU memory a request's
    LAST model call didn't already release) once the whole request is done.
    """
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        logger.exception("torch.cuda.empty_cache() failed during end-of-request cleanup")
    try:
        import paddle
        if paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
            paddle.device.cuda.empty_cache()
    except Exception:
        logger.exception("paddle.device.cuda.empty_cache() failed during end-of-request cleanup")
    try:
        # gc.collect()/empty_cache() above only free Python objects and GPU
        # memory -- glibc's malloc does NOT return freed heap memory (numpy/
        # PIL/OpenCV buffers) back to the OS on its own, it keeps it mapped
        # in per-thread arenas for reuse. malloc_trim(0) forces it to give
        # back whatever's currently free, which is what actually moves RSS
        # (as opposed to gc-tracked object counts) back down. Linux-only.
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        logger.exception("malloc_trim(0) failed during end-of-request cleanup")


def _log_rss(label: str) -> None:
    """Diagnostic only -- logs this process's CURRENT resident memory (RSS,
    i.e. real RAM actually held right now, not GPU) by reading
    /proc/self/status directly (Linux-only, no extra dependency like psutil
    needed). Used to pin down exactly how much RSS _release_memory() above
    actually reclaims within a single request, instead of guessing from
    `free -h` snapshots taken between separate requests."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    logger.info("[rss] %s: %s", label, line.split(":", 1)[1].strip())
                    return
    except Exception:
        pass


def _log_torch_tensor_stats(label: str) -> None:
    """Diagnostic only -- counts every torch.Tensor object CURRENTLY alive
    (via gc.get_objects(), no extra dependency) and, of those, how many
    still carry a non-None .grad_fn -- i.e. are still attached to a live
    autograd computation graph. If this count stays high (doesn't drop to
    ~0) even after _release_memory()'s gc.collect(), that's direct evidence
    some model call is building autograd graphs that never get released
    (e.g. a third-party model call not wrapped in torch.no_grad()/
    inference_mode()), rather than a guess."""
    try:
        import gc as _gc
        import torch
        tensors = [obj for obj in _gc.get_objects() if torch.is_tensor(obj)]
        with_grad_fn = sum(1 for t in tensors if t.grad_fn is not None)
        requires_grad = sum(1 for t in tensors if t.requires_grad)
        total_elements = sum(t.numel() for t in tensors)
        logger.info(
            "[torch-mem] %s: %d tensor(s) alive, %d with grad_fn, %d requires_grad, %d total elements",
            label, len(tensors), with_grad_fn, requires_grad, total_elements,
        )
    except Exception:
        logger.exception("torch tensor stats failed during diagnostic")


def _log_memory_hogs(label: str, top_n: int = 15) -> None:
    """Diagnostic only -- two views of what's CURRENTLY alive in memory
    (via gc.get_objects(), no extra dependency), broader than
    _log_torch_tensor_stats which only looks at torch.Tensor:
      1. Object counts grouped by type name (top N) -- catches ANY Python
         object type accumulating (list, dict, PIL.Image, ndarray, ...).
      2. Total byte size specifically for numpy.ndarray (via .nbytes, exact)
         and PIL.Image (width*height*bands, estimated) -- the most likely
         non-tensor culprits in a PDF/image pipeline, since rendered pages
         and OpenCV/PIL crops never show up in a torch-tensor-only scan.
    """
    try:
        import gc as _gc
        from collections import Counter

        objs = _gc.get_objects()
        counts = Counter(type(obj).__name__ for obj in objs)
        logger.info(
            "[obj-count] %s: total=%d objects, top types: %s",
            label, len(objs), counts.most_common(top_n),
        )
    except Exception:
        logger.exception("object type count failed during diagnostic")

    try:
        import gc as _gc
        import numpy as np
        from PIL import Image

        ndarray_count = 0
        ndarray_bytes = 0
        image_count = 0
        image_bytes_est = 0
        for obj in _gc.get_objects():
            if isinstance(obj, np.ndarray):
                ndarray_count += 1
                ndarray_bytes += obj.nbytes
            elif isinstance(obj, Image.Image):
                image_count += 1
                try:
                    image_bytes_est += obj.width * obj.height * len(obj.getbands())
                except Exception:
                    pass
        logger.info(
            "[mem-hogs] %s: ndarray count=%d (%.1f MB), PIL.Image count=%d (~%.1f MB est)",
            label, ndarray_count, ndarray_bytes / 1024**2, image_count, image_bytes_est / 1024**2,
        )
    except Exception:
        logger.exception("memory hogs diagnostic failed")


def _log_duplicate_functions(label: str, top_n: int = 20) -> None:
    """Diagnostic only -- groups every 'function' object currently alive by
    (__module__, __qualname__) and shows which ones appear the MOST times.
    Normally a function is defined ONCE (at module/class load time) and
    simply referenced/called many times -- every (module, qualname) pair
    should show up as exactly 1 distinct function object no matter how many
    times it's called. If the SAME (module, qualname) instead shows up as
    thousands of DISTINCT function objects, that's direct proof something
    is re-creating that function/closure on every call (e.g. a decorator
    re-applied per request, or a nested function/lambda defined inside a
    hot loop) instead of reusing it -- the most direct way to localize
    exactly which piece of code is responsible, instead of guessing from
    aggregate counts alone."""
    try:
        import gc as _gc
        import types
        from collections import Counter

        funcs = [o for o in _gc.get_objects() if isinstance(o, types.FunctionType)]
        keys = [f"{getattr(f, '__module__', None) or '?'}:{getattr(f, '__qualname__', None) or '?'}" for f in funcs]
        counts = Counter(keys)
        logger.info(
            "[dup-func] %s: %d function object(s) total, top duplicated (module:qualname -> count): %s",
            label, len(funcs), counts.most_common(top_n),
        )
    except Exception:
        logger.exception("duplicate function diagnostic failed")

    try:
        import gc as _gc
        from collections import Counter

        cells = [o for o in _gc.get_objects() if type(o).__name__ == "cell"]
        content_types = Counter()
        for c in cells:
            try:
                content_types[type(c.cell_contents).__name__] += 1
            except ValueError:
                content_types["<empty cell>"] += 1
        logger.info(
            "[dup-func] %s: %d cell (closure) object(s) total, contents by type: %s",
            label, len(cells), content_types.most_common(top_n),
        )
    except Exception:
        logger.exception("cell content diagnostic failed")

    try:
        import gc as _gc
        from collections import Counter

        params = [o for o in _gc.get_objects() if type(o).__name__ == "Parameter"]
        param_names = Counter(getattr(p, "name", "?") for p in params)
        logger.info(
            "[dup-func] %s: %d inspect.Parameter object(s) total, top names: %s",
            label, len(params), param_names.most_common(top_n),
        )
    except Exception:
        logger.exception("parameter diagnostic failed")


@router.post("/pdf", response_model=OCRResponse, response_model_by_alias=True)
async def ocr_pdf(
    file: UploadFile = File(...),
    pipeline: DocumentPipeline = Depends(get_pipeline),
) -> OCRResponse:
    t_request = time.time()
    _log_rss("pdf: request start")
    _log_torch_tensor_stats("pdf: request start")
    _log_memory_hogs("pdf: request start")
    _log_duplicate_functions("pdf: request start")
    try:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only .pdf files are supported")

        t_read = time.time()
        pdf_bytes = await file.read()
        if not pdf_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        logger.info("Read upload body: %.2fs (%d bytes)", time.time() - t_read, len(pdf_bytes))

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(pdf_bytes)
                tmp_path = tmp.name

            # process_pdf is synchronous CPU/GPU-bound work -- run it in a
            # worker thread so it doesn't block the asyncio event loop for its
            # whole duration (a direct call here would stall every other
            # request, including just reading a new request's body, until this
            # one finishes).
            result = await asyncio.to_thread(
                partial(pipeline.process_pdf, tmp_path, source_filename=file.filename)
            )
        except (PDFPageCountError, PDFSyntaxError, PDFInfoNotInstalledError) as e:
            raise HTTPException(status_code=422, detail=f"Unable to read PDF: {e}") from e
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("OCR pipeline failed for upload %s", file.filename)
            raise HTTPException(status_code=500, detail="Internal processing error") from e
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

        t_response = time.time()
        response = OCRResponse(file=result["json"]["file"], json=result["json"], markdown=result["markdown"])
        logger.info(
            "Route %s done: total=%.2fs (upload_read included, build_response=%.2fs)",
            file.filename, time.time() - t_request, time.time() - t_response,
        )
        return response
    finally:
        _release_memory()
        _log_rss("pdf: after release_memory")
        _log_torch_tensor_stats("pdf: after release_memory")
        _log_memory_hogs("pdf: after release_memory")
        _log_duplicate_functions("pdf: after release_memory")


@router.post("/pdfs", response_model=BatchOCRResponse, response_model_by_alias=True)
async def ocr_pdfs(
    files: list[UploadFile] = File(...),
    pipeline: DocumentPipeline = Depends(get_pipeline),
) -> BatchOCRResponse:
    """Accepts multiple PDFs, groups them so no group exceeds
    batch_api.max_pages_per_group total pages (a single oversized PDF just
    becomes its own group), and processes groups ONE AT A TIME -- each
    group's pages are concatenated into one combined batch call
    (DocumentPipeline.process_pdf_group), then split back per-source-PDF for
    separate output.json/output.md writes under pipeline_outputs/<file>/,
    written to disk as soon as that group finishes (not all held until the
    end) -- the same json/markdown are also included per-file in the response.
    A PDF that fails to upload/read is marked "error" and skipped without
    aborting the rest of the batch."""
    t_request = time.time()
    _log_rss("pdfs: request start")
    _log_torch_tensor_stats("pdfs: request start")
    _log_memory_hogs("pdfs: request start")
    _log_duplicate_functions("pdfs: request start")
    try:
        max_pages_per_group = pipeline.config.batch_api_max_pages_per_group

        results: list[BatchItemResult] = []
        saved: list[tuple[str, str, int]] = []  # (tmp_path, label, page_count)
        tmp_paths: list[str] = []

        for file in files:
            if not file.filename or not file.filename.lower().endswith(".pdf"):
                results.append(BatchItemResult(file=file.filename or "<unknown>", status="error", error="Only .pdf files are supported"))
                continue
            data = await file.read()
            if not data:
                results.append(BatchItemResult(file=file.filename, status="error", error="Uploaded file is empty"))
                continue
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(data)
                tmp_path = tmp.name
            tmp_paths.append(tmp_path)
            try:
                # Cheap metadata-only page count (poppler's pdfinfo) -- doesn't
                # render any pages, so grouping can be decided before paying the
                # rendering cost, and pages only get rendered once (inside
                # process_pdf_group) at actual processing time.
                info = await asyncio.to_thread(pdfinfo_from_path, tmp_path)
            except (PDFInfoNotInstalledError, PDFPageCountError, PDFSyntaxError) as e:
                results.append(BatchItemResult(file=file.filename, status="error", error=f"Unable to read PDF: {e}"))
                continue
            saved.append((tmp_path, file.filename, info["Pages"]))

        groups: list[list[tuple[str, str]]] = []
        current: list[tuple[str, str]] = []
        current_pages = 0
        for tmp_path, label, page_count in saved:
            if current and current_pages + page_count > max_pages_per_group:
                groups.append(current)
                current = []
                current_pages = 0
            current.append((tmp_path, label))
            current_pages += page_count
        if current:
            groups.append(current)

        try:
            for group in groups:
                try:
                    group_results = await asyncio.to_thread(pipeline.process_pdf_group, group)
                except Exception:
                    logger.exception("process_pdf_group crashed for a %d-file group", len(group))
                    group_results = [
                        {"file": label, "status": "error", "error": "Internal processing error"}
                        for _, label in group
                    ]
                results.extend(BatchItemResult(**r) for r in group_results)
                _release_memory()
                _log_rss("pdfs: after group release_memory")
                _log_torch_tensor_stats("pdfs: after group release_memory")
                _log_memory_hogs("pdfs: after group release_memory")
                _log_duplicate_functions("pdfs: after group release_memory")
        finally:
            for p in tmp_paths:
                if os.path.exists(p):
                    os.remove(p)

        logger.info(
            "Route /pdfs done: %d file(s) in %d group(s), total=%.2fs",
            len(files), len(groups), time.time() - t_request,
        )
        return BatchOCRResponse(results=results)
    finally:
        _release_memory()
        _log_rss("pdfs: after final release_memory")
        _log_torch_tensor_stats("pdfs: after final release_memory")
        _log_memory_hogs("pdfs: after final release_memory")
        _log_duplicate_functions("pdfs: after final release_memory")


@router.post("/images", response_model=BatchOCRResponse, response_model_by_alias=True)
async def ocr_images(
    files: list[UploadFile] = File(...),
    pipeline: DocumentPipeline = Depends(get_pipeline),
) -> BatchOCRResponse:
    """Accepts one or more images, groups them so no group exceeds
    batch_api.max_images_per_group images, and processes groups ONE AT A
    TIME -- each group's images are batched together in one combined call
    (DocumentPipeline.process_image_group, 1 image == 1 "page"), then split
    back per-image for separate output.json/output.md writes under
    pipeline_outputs/<file>/, written to disk as soon as that group
    finishes -- the same json/markdown are also included per-file in the
    response. An image that fails to upload/decode is marked "error" and
    skipped without aborting the rest of the batch."""
    t_request = time.time()
    _log_rss("images: request start")
    _log_torch_tensor_stats("images: request start")
    _log_memory_hogs("images: request start")
    _log_duplicate_functions("images: request start")
    try:
        max_images_per_group = pipeline.config.batch_api_max_images_per_group

        results: list[BatchItemResult] = []
        saved: list[tuple[str, str]] = []  # (tmp_path, label)
        tmp_paths: list[str] = []

        for file in files:
            if not file.filename:
                results.append(BatchItemResult(file="<unknown>", status="error", error="Missing filename"))
                continue
            data = await file.read()
            if not data:
                results.append(BatchItemResult(file=file.filename, status="error", error="Uploaded file is empty"))
                continue
            suffix = os.path.splitext(file.filename)[1] or ".img"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(data)
                tmp_path = tmp.name
            tmp_paths.append(tmp_path)
            saved.append((tmp_path, file.filename))

        groups: list[list[tuple[str, str]]] = [
            saved[i:i + max_images_per_group] for i in range(0, len(saved), max_images_per_group)
        ]

        try:
            for group in groups:
                try:
                    group_results = await asyncio.to_thread(pipeline.process_image_group, group)
                except Exception:
                    logger.exception("process_image_group crashed for a %d-file group", len(group))
                    group_results = [
                        {"file": label, "status": "error", "error": "Internal processing error"}
                        for _, label in group
                    ]
                results.extend(BatchItemResult(**r) for r in group_results)
                _release_memory()
                _log_rss("images: after group release_memory")
                _log_torch_tensor_stats("images: after group release_memory")
                _log_memory_hogs("images: after group release_memory")
                _log_duplicate_functions("images: after group release_memory")
        finally:
            for p in tmp_paths:
                if os.path.exists(p):
                    os.remove(p)

        logger.info(
            "Route /images done: %d file(s) in %d group(s), total=%.2fs",
            len(files), len(groups), time.time() - t_request,
        )
        return BatchOCRResponse(results=results)
    finally:
        _release_memory()
        _log_rss("images: after final release_memory")
        _log_torch_tensor_stats("images: after final release_memory")
        _log_memory_hogs("images: after final release_memory")
        _log_duplicate_functions("images: after final release_memory")
