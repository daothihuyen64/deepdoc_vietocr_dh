import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from module.pipeline import build_pipeline

from . import routes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading OCR pipeline (layout/OCR/TSR models)...")
    app.state.pipeline = build_pipeline()
    logger.info("Pipeline ready")
    yield
    app.state.pipeline = None


app = FastAPI(title="DeepDoc VietOCR API", lifespan=lifespan)
app.include_router(routes.router)
# Mounted last so it doesn't shadow the /api/v1/... routes above -- serves
# server/static/index.html at "/" as a minimal manual-test UI for the API.
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
