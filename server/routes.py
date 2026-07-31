import asyncio
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


@router.post("/pdf", response_model=OCRResponse, response_model_by_alias=True)
async def ocr_pdf(
    file: UploadFile = File(...),
    pipeline: DocumentPipeline = Depends(get_pipeline),
) -> OCRResponse:
    t_request = time.time()
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


@router.post("/pdfs", response_model=BatchOCRResponse)
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
    end). A PDF that fails to upload/read is marked "error" and skipped
    without aborting the rest of the batch."""
    t_request = time.time()
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
    finally:
        for p in tmp_paths:
            if os.path.exists(p):
                os.remove(p)

    logger.info(
        "Route /pdfs done: %d file(s) in %d group(s), total=%.2fs",
        len(files), len(groups), time.time() - t_request,
    )
    return BatchOCRResponse(results=results)


@router.post("/images", response_model=BatchOCRResponse)
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
    finishes. An image that fails to upload/decode is marked "error" and
    skipped without aborting the rest of the batch."""
    t_request = time.time()
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
    finally:
        for p in tmp_paths:
            if os.path.exists(p):
                os.remove(p)

    logger.info(
        "Route /images done: %d file(s) in %d group(s), total=%.2fs",
        len(files), len(groups), time.time() - t_request,
    )
    return BatchOCRResponse(results=results)
