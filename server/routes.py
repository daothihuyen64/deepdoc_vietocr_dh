import asyncio
import logging
import os
import tempfile
import time
from functools import partial

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError,
)

from module.pipeline import DocumentPipeline

from .deps import get_pipeline
from .schemas import OCRResponse

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
