"""
app/routers/candidates.py

Phase 4: Candidate Scoring Synthesis

Public API router exposing two endpoints:

  POST /api/v1/ingest
    - Accepts a multipart resume file (.pdf / .docx)
    - Delegates to the secure parser layer (size guard + sanitisation)
    - Chunks → embeds → mean-pools → commits to FAISS index
    - Returns ingestion confirmation with stable candidate_id

  POST /api/v1/match
    - Accepts a job description string + top_k modifier
    - Embeds the query via the same sentence-transformer pipeline
    - Runs cosine-similarity search across the active FAISS index
    - Returns a descending ranked list of matched candidates
"""

from __future__ import annotations

import logging
import uuid
from pathlib import PurePosixPath
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status

from app.schemas import (
    JobDescriptionRequest,
    RankedCandidate,
    RankedResultsResponse,
)
from app.services.embedder import index_manager
from app.services.parser import secure_extract_pdf_text

logger = logging.getLogger("redrob.router.candidates")

# ---------------------------------------------------------------------------
# Router definition
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/api/v1", tags=["Candidate Discovery"])

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_UPLOAD_BYTES: int = 5 * 1024 * 1024   # 5 MB — mirrors parser layer guard
_ALLOWED_CONTENT_TYPES: frozenset[str] = frozenset(
    {
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword",
    }
)


# ---------------------------------------------------------------------------
# POST /api/v1/ingest
# ---------------------------------------------------------------------------

@router.post(
    "/ingest",
    status_code=status.HTTP_201_CREATED,
    summary="Ingest a candidate resume into the vector index.",
    response_description="Ingestion confirmation with the assigned candidate ID.",
)
async def ingest_resume(
    request: Request,
    file: Annotated[UploadFile, File(description="Resume file — .pdf or .docx only.")],
    candidate_name: Annotated[
        str | None,
        Form(description="Optional display name for the candidate."),
    ] = None,
) -> dict:
    """
    Pipeline:
      1. MIME-type pre-check (fast reject before reading body).
      2. Read raw bytes with a hard ceiling to prevent memory exhaustion.
      3. Delegate to `secure_extract_pdf_text()` which enforces:
           - Path traversal sanitisation on the filename.
           - 5 MB decompression-bomb guard.
           - Text normalisation / injection stripping.
      4. Chunk → embed → mean-pool → add to FAISS index.
      5. Persist index to disk after each ingestion.
      6. Return stable candidate_id for downstream reference.
    """
    # ── Guard: MIME type ─────────────────────────────────────────────────────
    content_type = (file.content_type or "").lower()
    filename = file.filename or "unknown"
    suffix = PurePosixPath(filename).suffix.lower()

    if content_type not in _ALLOWED_CONTENT_TYPES and suffix not in {".pdf", ".docx", ".doc"}:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported file type '{content_type}'. "
                "Only PDF and DOCX resumes are accepted."
            ),
        )

    # ── Read raw bytes (secondary bomb guard before parser) ──────────────────
    try:
        file_bytes: bytes = await file.read()
    except Exception as exc:
        logger.error("Failed to read uploaded file '%s': %s", filename, exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not read the uploaded file stream.",
        ) from exc

    if len(file_bytes) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"Upload exceeds the 5 MB limit "
                f"({len(file_bytes) / 1_048_576:.2f} MB). "
                "Please compress or trim the resume."
            ),
        )

    if len(file_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is empty.",
        )

    # ── Secure extraction (parser layer handles sanitisation + normalisation) ─
    normalised_text: str = await secure_extract_pdf_text(
        file_bytes=file_bytes,
        filename=filename,
    )

    # ── Generate a stable candidate ID ───────────────────────────────────────
    # Use a deterministic prefix from the filename stem for traceability
    safe_stem = PurePosixPath(filename).stem[:40]          # truncate long names
    candidate_id = f"{safe_stem}_{uuid.uuid4().hex[:8]}"

    # ── Embed + index ─────────────────────────────────────────────────────────
    try:
        await index_manager.add_candidate(
            candidate_id=candidate_id,
            resume_text=normalised_text,
        )
    except Exception as exc:
        logger.error("FAISS indexing failed for candidate '%s': %s", candidate_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Vector indexing failed. Please retry.",
        ) from exc

    # ── Persist index to disk ─────────────────────────────────────────────────
    try:
        await index_manager.save()
    except Exception as exc:
        # Non-fatal — log but don't fail the request (index is in memory)
        logger.warning("FAISS disk persist failed (index still in memory): %s", exc)

    logger.info(
        "Ingested candidate '%s' from file '%s' — %d chars extracted. "
        "Total indexed: %d",
        candidate_id,
        filename,
        len(normalised_text),
        index_manager.total_candidates,
    )

    return {
        "status": "success",
        "candidate_id": candidate_id,
        "display_name": candidate_name or PurePosixPath(filename).stem,
        "filename": filename,
        "characters_extracted": len(normalised_text),
        "total_candidates_indexed": index_manager.total_candidates,
    }


# ---------------------------------------------------------------------------
# POST /api/v1/match
# ---------------------------------------------------------------------------

@router.post(
    "/match",
    response_model=RankedResultsResponse,
    summary="Rank indexed candidates against a job description.",
    response_description="Descending ranked list of candidates by cosine similarity.",
)
async def match_candidates(
    request: Request,
    payload: JobDescriptionRequest,
) -> RankedResultsResponse:
    """
    Pipeline:
      1. Validate JD payload via Pydantic (handled by FastAPI automatically).
         The JobDescriptionRequest validator also strips prompt-injection patterns.
      2. Guard: require at least one candidate in the index.
      3. Embed the job description using the same sentence-transformer pipeline.
      4. Run cosine similarity search against the FAISS index.
      5. Build and return a structured RankedResultsResponse.
    """
    # ── Guard: index must have candidates ────────────────────────────────────
    if not index_manager.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "No candidates are indexed yet. "
                "Please upload resumes via POST /api/v1/ingest first."
            ),
        )

    # ── Vector similarity search ──────────────────────────────────────────────
    try:
        raw_results = await index_manager.search(
            query_text=payload.description,
            top_k=payload.top_k,
        )
    except Exception as exc:
        logger.error("FAISS search failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Vector search failed. Please retry.",
        ) from exc

    if not raw_results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No matching candidates found for the provided job description.",
        )

    # ── Build ranked response ─────────────────────────────────────────────────
    ranked: list[RankedCandidate] = []
    for rank_idx, (candidate_id, score) in enumerate(raw_results, start=1):
        # Reconstruct display name from candidate_id stem (before the UUID suffix)
        display_name = candidate_id.rsplit("_", 1)[0].replace("_", " ").title()

        ranked.append(
            RankedCandidate(
                rank=rank_idx,
                candidate_id=candidate_id,
                full_name=display_name,
                source_filename=f"{candidate_id}",
                similarity_score=round(max(0.0, min(1.0, score)), 6),
            )
        )

    logger.info(
        "Match query for '%s' returned %d results (top score: %.4f)",
        payload.title,
        len(ranked),
        ranked[0].similarity_score if ranked else 0.0,
    )

    return RankedResultsResponse(
        job_title=payload.title,
        total_candidates_evaluated=index_manager.total_candidates,
        results=ranked,
    )
