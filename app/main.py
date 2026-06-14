"""
app/main.py

Async FastAPI application entry-point for the Redrob AI Engine.
Handles application lifecycle, global exception middleware, and
route registration.
"""

from __future__ import annotations

import logging
import traceback
from contextlib import asynccontextmanager
from typing import AsyncGenerator

# pyrefly: ignore [missing-import]
from fastapi import FastAPI, HTTPException, Request, status
# pyrefly: ignore [missing-import]
from fastapi.middleware.cors import CORSMiddleware
# pyrefly: ignore [missing-import]
from fastapi.responses import JSONResponse

from app.services.embedder import index_manager

from app.schemas import (
    JobDescriptionRequest,
    RankedResultsResponse,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("redrob.main")


# ---------------------------------------------------------------------------
# Lifespan — warm up / tear down shared resources
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:  # noqa: D401
    """
    FastAPI lifespan context manager.
    Phase 3 will load the FAISS index and sentence-transformer model here.
    """
    logger.info("🚀 Redrob AI Engine starting up …")
    # Phase 3: load/create FAISS index and wire into app state
    index_manager.load_or_create()
    app.state.index_manager = index_manager
    logger.info("FAISS index ready — %d candidates indexed.", index_manager.total_candidates)
    yield
    # Persist index to disk on clean shutdown
    await index_manager.save()
    logger.info("🛑 FAISS index saved. Redrob AI Engine shutting down …")



# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Redrob AI Engine",
    description=(
        "Intelligent Candidate Discovery using dense vector embeddings "
        "and parallel Cosine Similarity matching — Track 1, Redrob Hackathon."
    ),
    version="0.1.0",
    lifespan=lifespan,
    # Hide internal schema paths from public OpenAPI docs
    openapi_url="/openapi.json",
    docs_url="/docs",
    redoc_url=None,
)

# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Tighten in production
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


# ---------------------------------------------------------------------------
# Global exception handlers
# ---------------------------------------------------------------------------

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Return clean JSON error envelope — never expose raw tracebacks."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "status_code": exc.status_code},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Catch-all for unhandled runtime exceptions.
    Logs the full traceback server-side; returns a sanitized 500 to the client.
    """
    logger.error("Unhandled exception on %s:\n%s", request.url.path, traceback.format_exc())
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred.", "status_code": 500},
    )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
async def health_check() -> dict[str, str]:
    """Liveness probe — returns 200 OK when the engine is running."""
    return {"status": "ok", "engine": "Redrob AI Engine v0.1.0"}


# ---------------------------------------------------------------------------
# Placeholder match endpoint (wired up fully in Phase 3 / 4)
# ---------------------------------------------------------------------------

@app.post(
    "/match",
    response_model=RankedResultsResponse,
    tags=["Candidate Discovery"],
    summary="Rank candidates against a job description using vector similarity.",
)
async def match_candidates(payload: JobDescriptionRequest) -> RankedResultsResponse:
    """
    Accepts a job description and returns the top-K ranked candidates
    by cosine similarity against their resume embeddings.

    Full embedding + FAISS logic is injected in Phase 3.
    """
    # Guard: FAISS index must be loaded (Phase 3 populates app.state)
    if not hasattr(app.state, "faiss_index"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector index not yet initialised. Upload resumes first via /ingest.",
        )

    # Phase 3/4 will replace this stub
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Matching engine not yet implemented — coming in Phase 3.",
    )
