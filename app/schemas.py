"""
app/schemas.py

Pydantic v2 data schemas — the single source of truth for all
request/response validation contracts in the IR-Data-Matching-Engine.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional


# ---------------------------------------------------------------------------
# Inbound Models
# ---------------------------------------------------------------------------

class JobDescriptionRequest(BaseModel):
    """Payload submitted by the recruiter to drive candidate discovery."""

    title: str = Field(
        ...,
        min_length=2,
        max_length=200,
        description="Job title being recruited for.",
    )
    description: str = Field(
        ...,
        min_length=20,
        max_length=10_000,
        description="Full plain-text job description.",
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of top-ranked candidates to return.",
    )

    @field_validator("description")
    @classmethod
    def strip_suspicious_commands(cls, v: str) -> str:
        """
        Indirect Prompt Injection Mitigation (Section 3, Point 3):
        Strip known command-injection patterns that bad actors embed in
        job descriptions to poison the embedding pipeline.
        """
        forbidden_prefixes = ("ignore previous", "system:", "<<SYS>>", "[INST]")
        lowered = v.lower()
        for prefix in forbidden_prefixes:
            if prefix in lowered:
                raise ValueError(
                    "Job description contains disallowed control sequences."
                )
        return v


# ---------------------------------------------------------------------------
# Candidate / Profile Models
# ---------------------------------------------------------------------------

class CandidateProfile(BaseModel):
    """
    Structured representation of a parsed candidate resume.
    Populated by the document ingestion layer (Phase 2).
    """

    candidate_id: str = Field(..., description="Unique stable identifier.")
    full_name: Optional[str] = Field(default=None, max_length=200)
    email: Optional[str] = Field(default=None, max_length=320)
    raw_text: str = Field(
        ...,
        min_length=10,
        description="Sanitized plain-text extracted from the resume file.",
    )
    source_filename: str = Field(
        ...,
        max_length=255,
        description="Original filename — already path-traversal sanitized by the ingestion layer.",
    )


# ---------------------------------------------------------------------------
# Outbound / Scoring Models
# ---------------------------------------------------------------------------

class RankedCandidate(BaseModel):
    """Single entry in the ranked output returned to the recruiter."""

    rank: int = Field(..., ge=1)
    candidate_id: str
    full_name: Optional[str] = None
    source_filename: str
    similarity_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Cosine similarity score between candidate embedding and JD embedding.",
    )


class RankedResultsResponse(BaseModel):
    """Top-level API response envelope for the /match endpoint."""

    job_title: str
    total_candidates_evaluated: int
    results: list[RankedCandidate]
