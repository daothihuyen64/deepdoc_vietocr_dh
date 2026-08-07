import logging

# Must run before any other import (paddle/paddlex configure the root
# logger as a side effect of import, which makes basicConfig() a silent
# no-op if it runs after them -- our own INFO logs would then vanish while
# uvicorn's independently-configured access logs still show up fine).
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

from contextlib import asynccontextmanager  # noqa: E402
from pathlib import Path  # noqa: E402

from fastapi import Depends, FastAPI  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from module.pipeline import DocumentPipeline, build_pipeline  # noqa: E402

from . import routes  # noqa: E402
from .deps import get_pipeline  # noqa: E402

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading OCR pipeline (layout/OCR/TSR models)...")
    app.state.pipeline = build_pipeline()
    logger.info("Pipeline ready")
    yield
    app.state.pipeline = None


app = FastAPI(title="DeepDoc FastOCR API", lifespan=lifespan)


@app.get("/health")
async def health(pipeline: DocumentPipeline = Depends(get_pipeline)) -> dict:
    """Liveness/readiness check -- 200 once the pipeline has finished
    loading (via the same get_pipeline dependency every OCR route uses),
    503 while still starting up or if it failed to load."""
    return {"status": "ok"}


app.include_router(routes.router)
# Mounted last so it doesn't shadow the /api/v1/... routes above -- serves
# server/static/index.html at "/" as a minimal manual-test UI for the API.
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
