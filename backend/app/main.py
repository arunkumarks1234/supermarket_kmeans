from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from app.api.routes import router as api_router
from app.core.config import OUTPUT_DIR, DATA_FILE
from app.ml.preprocess import load_data, clean_data
from app.api.routes import get_data

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Ensure output dir exists, attempt to load data to cache
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if DATA_FILE.exists():
        try:
            get_data()
            logger.info("Data loaded successfully on startup.")
        except Exception as e:
            logger.error(f"Failed to load data on startup: {e}")
    else:
        logger.warning("Data file not found on startup.")
    yield
    # Shutdown
    logger.info("Shutting down.")

app = FastAPI(title="Supermarket Segmentation API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")
