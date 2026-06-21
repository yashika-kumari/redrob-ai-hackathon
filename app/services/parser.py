"""
app/services/parser.py

Secure, asynchronous multi-format resume document parser.

Security posture (Section 3 of conversation_state.md):
  1. Decompression/Parsing Bomb mitigation — hard 5 MB ceiling on raw bytes
     before any parsing library touches the buffer.
  2. Path Traversal mitigation — filename is sanitised via pathlib before
     being used anywhere; relative components (../) are completely stripped.
  3. Indirect Prompt Injection mitigation — extracted text is passed through
     a normalisation pipeline that lowercases, removes control codes, and
     strips hidden carriage-return escapes before returning to caller.
"""

from __future__ import annotations

import asyncio
import io
import logging
import re
import unicodedata
from pathlib import PurePosixPath, PureWindowsPath

import PyPDF2
from docx import Document as DocxDocument
from fastapi import HTTPException, status

logger = logging.getLogger("ir_data_matching_engine.parser")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_BYTES: int = 5 * 1024 * 1024          # 5 MB hard ceiling
_ALLOWED_SUFFIXES: frozenset[str] = frozenset({".pdf", ".docx", ".doc"})

# Control-code pattern: strips C0 / C1 control chars except normal whitespace
# (\t, \n, \r are excluded from stripping so we can replace them cleanly later)
_CONTROL_CODE_RE = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]"
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sanitise_filename(filename: str) -> str:
    """
    Path Traversal Mitigation (Section 3, Point 1):

    Strip every path component except the final stem+suffix so that strings
    like '../../etc/passwd.pdf' or '/abs/path/resume.pdf' are reduced to
    just 'resume.pdf'.  Uses both POSIX and Windows path parsers to handle
    mixed-separator payloads.
    """
    # Normalise separators — replace backslashes so PurePosixPath catches them
    cleaned = filename.replace("\\", "/")

    # Extract only the final name element (discards any leading path segments)
    name_only = PurePosixPath(cleaned).name

    # Secondary Windows parse in case of drive-letter prefixes
    name_only = PureWindowsPath(name_only).name

    # If nothing remains (e.g. attacker sent only '../'), raise immediately
    if not name_only:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid filename: no safe base name could be extracted.",
        )

    suffix = PurePosixPath(name_only).suffix.lower()
    if suffix not in _ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"File type '{suffix}' is not supported. "
                f"Accepted formats: {', '.join(sorted(_ALLOWED_SUFFIXES))}."
            ),
        )

    return name_only


def _enforce_size_limit(file_bytes: bytes) -> None:
    """
    Decompression / Parsing Bomb Mitigation (Section 3, Point 2):

    Reject the payload *before* handing it to any parsing library.
    A 5 MB ceiling ensures DoS via deeply nested or malformed archives
    cannot consume CPU or memory.
    """
    if len(file_bytes) > _MAX_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"File exceeds the 5 MB upload limit "
                f"({len(file_bytes) / 1_048_576:.2f} MB received). "
                "Please compress or trim the document."
            ),
        )


def _normalise_text(raw: str) -> str:
    """
    Indirect Prompt Injection Mitigation (Section 3, Point 3):

    1. Lowercase — equalises surface form for embedding consistency.
    2. Strip Unicode control characters — removes C0/C1 codes that could
       act as hidden command delimiters.
    3. Collapse carriage returns / form-feeds to a single newline.
    4. NFKC normalisation — decomposes ligatures and compatibility forms
       that could be used to smuggle look-alike command tokens.
    5. Collapse redundant whitespace runs into single spaces.
    """
    # NFKC normalisation first (decomposes ﬁ → fi, ™ → TM, etc.)
    text = unicodedata.normalize("NFKC", raw)

    # Lowercase
    text = text.lower()

    # Replace carriage-return-linefeed / carriage-return / form-feed with \n
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n")

    # Strip C0/C1 control codes
    text = _CONTROL_CODE_RE.sub("", text)

    # Collapse runs of whitespace (except newlines) to a single space
    text = re.sub(r"[^\S\n]+", " ", text)

    # Collapse 3+ consecutive newlines to two (paragraph break)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ---------------------------------------------------------------------------
# PDF extractor
# ---------------------------------------------------------------------------

def _extract_pdf_sync(file_bytes: bytes) -> str:
    """
    Synchronous PyPDF2 extraction — runs in a thread-pool executor
    so the async event-loop is never blocked.
    """
    buffer = io.BytesIO(file_bytes)
    try:
        reader = PyPDF2.PdfReader(buffer, strict=False)
    except Exception as exc:
        logger.error("PyPDF2 failed to open PDF buffer: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="The PDF file could not be parsed. It may be corrupt or encrypted.",
        ) from exc

    pages: list[str] = []
    for page_num, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
            pages.append(text)
        except Exception as exc:  # noqa: BLE001
            # Log but continue — a bad page should not abort the whole doc
            logger.warning("Skipping page %d due to extraction error: %s", page_num, exc)

    if not pages:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No readable text could be extracted from the PDF.",
        )

    return "\n".join(pages)


# ---------------------------------------------------------------------------
# DOCX extractor
# ---------------------------------------------------------------------------

def _extract_docx_sync(file_bytes: bytes) -> str:
    """
    Synchronous python-docx extraction — also runs in thread-pool executor.
    """
    buffer = io.BytesIO(file_bytes)
    try:
        doc = DocxDocument(buffer)
    except Exception as exc:
        logger.error("python-docx failed to open DOCX buffer: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="The DOCX file could not be parsed. It may be corrupt or malformed.",
        ) from exc

    paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
    if not paragraphs:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No readable text could be extracted from the DOCX.",
        )

    return "\n".join(paragraphs)


def _extract_name_from_raw_text(raw_text: str) -> str | None:
    """
    Attempt to extract the candidate's name from the first non-empty lines
    of the raw (un-normalised) text.
    """
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    for line in lines[:5]:
        words = line.split()
        if 1 <= len(words) <= 4 and not any(c in line for c in "@:/.\\,+=#_*()[]{}0123456789"):
            lower_line = line.lower()
            if not any(w in lower_line for w in ("resume", "cv", "curriculum", "page", "profile", "summary", "contact", "education", "experience")):
                return line
    return None


# ---------------------------------------------------------------------------
# Public async interface
# ---------------------------------------------------------------------------

async def secure_extract_pdf_text(file_bytes: bytes, filename: str) -> tuple[str, str | None]:
    """
    Thread-safe asynchronous entry-point for secure resume text extraction.

    Parameters
    ----------
    file_bytes : bytes
        Raw file content uploaded by the client.
    filename : str
        Original filename as received from the HTTP request — will be
        sanitised internally; do NOT pre-sanitise before calling.

    Returns
    -------
    tuple[str, str | None]
        Normalised text content and auto-extracted candidate name (if found).

    Raises
    ------
    fastapi.HTTPException
        400 — invalid/unsafe filename
        413 — file exceeds 5 MB ceiling
        415 — unsupported file type
        422 — unparseable or empty document
    """
    # ── 1. Path traversal sanitisation ──────────────────────────────────────
    safe_name = _sanitise_filename(filename)
    suffix = PurePosixPath(safe_name).suffix.lower()

    logger.info("Extracting text from '%s' (%d bytes)", safe_name, len(file_bytes))

    # ── 2. Size enforcement BEFORE parsing ──────────────────────────────────
    _enforce_size_limit(file_bytes)

    # ── 3. Dispatch to correct parser in thread-pool (non-blocking) ─────────
    loop = asyncio.get_running_loop()

    if suffix == ".pdf":
        raw_text: str = await loop.run_in_executor(
            None, _extract_pdf_sync, file_bytes
        )
    elif suffix in (".docx", ".doc"):
        raw_text = await loop.run_in_executor(
            None, _extract_docx_sync, file_bytes
        )
    else:
        # Should be unreachable due to _sanitise_filename check, but be safe
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {suffix}",
        )

    # Extract name from raw text before normalising
    candidate_name = _extract_name_from_raw_text(raw_text)

    # ── 4. Normalise / strip injection vectors ───────────────────────────────
    normalised = _normalise_text(raw_text)

    if len(normalised) < 20:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Extracted text is too short to be a valid resume.",
        )

    logger.info(
        "Successfully extracted %d characters from '%s' (name: %s)",
        len(normalised), safe_name, candidate_name
    )
    return normalised, candidate_name

