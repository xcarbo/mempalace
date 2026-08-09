"""
embedding_input.py — build the text fed to the EMBEDDER for chunked drawers.

The stored document is ALWAYS the caller's verbatim content; nothing in this
module ever touches what gets written to the palace. What it changes is the
vector: a chunk of a curated drawer used to be embedded as an anonymous
800-char window ("part 3 of a table nobody named"), which is why the
canonical roadmap drawer's chunks ranked 91→108,768 on exact search. Each
chunk's embedding input gets a template breadcrumb:

    "<wing>/<room> — <drawer title>, part <i>/<N>: " + chunk_body

This is the no-LLM flavor of Anthropic's contextual retrieval (their
LLM-generated version reports top-20 retrieval failures falling 5.7% → 3.7%;
the template version is free and needs no model call). The seam is
deliberately narrow so an LLM-generated header can replace
:func:`contextual_header` later without touching any call site.

Gated by ``MempalaceConfig.embed_context_headers`` (default ON,
``MEMPALACE_EMBED_CONTEXT_HEADERS=false`` to disable). Fail-open: any error
returns ``None`` and the backend embeds the stored documents as before.
"""

import logging
import re
from typing import Optional

logger = logging.getLogger("mempalace_mcp")

_TITLE_MAX_CHARS = 80


def drawer_title(content: str, max_chars: int = _TITLE_MAX_CHARS) -> str:
    """Derive a short human-readable title from drawer content.

    First non-empty line, markdown heading markers and surrounding
    whitespace stripped, internal whitespace collapsed, capped at
    ``max_chars``. Curated drawers almost always open with a heading
    ("# mempalace — Roadmap ..."), which is exactly the name search
    queries use.
    """
    for line in (content or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        stripped = re.sub(r"^#{1,6}\s+", "", stripped)
        stripped = re.sub(r"\s+", " ", stripped).strip()
        if not stripped:
            continue
        if len(stripped) > max_chars:
            stripped = stripped[: max_chars - 1].rstrip() + "…"
        return stripped
    return ""


def contextual_header(wing: str, room: str, title: str, part: int, total: int) -> str:
    """The template breadcrumb prepended to one chunk's embedding input.

    Replace THIS function with an LLM-generated variant to upgrade the
    whole pipeline — every call site goes through here.
    """
    title_part = f" — {title}" if title else ""
    return f"{wing}/{room}{title_part}, part {part}/{total}: "


def contextual_embedding_texts(wing: str, room: str, content: str, chunk_docs: list) -> list:
    """Embedding-input text for each chunk of one logical drawer."""
    title = drawer_title(content)
    total = len(chunk_docs)
    return [
        contextual_header(wing, room, title, i + 1, total) + (doc or "")
        for i, doc in enumerate(chunk_docs)
    ]


def maybe_contextual_embeddings(
    config, wing: str, room: str, content: str, chunk_docs: list
) -> Optional[list]:
    """Vectors for header-prefixed chunk texts, or ``None`` to fall back.

    ``None`` means "let the backend embed the stored documents" — the exact
    pre-header behaviour. Returned on: flag off, empty input, or any
    embedding failure (a retrievability upgrade must never block a write).

    Uses :func:`mempalace.embedding.get_embedding_function` — the same
    cached EF the Chroma collection itself resolves (``backends.chroma
    ._resolve_embedding_function``), so explicit vectors stay consistent
    with query-time vectors.
    """
    try:
        if config is None or not config.embed_context_headers:
            return None
        if not chunk_docs:
            return None
        from .embedding import get_embedding_function

        texts = contextual_embedding_texts(wing, room, content, chunk_docs)
        vectors = get_embedding_function()(input=texts)
        if vectors is None or len(vectors) != len(chunk_docs):
            return None
        # Plain Python floats: chromadb's upsert validation rejects
        # np.float32 items ("Expected embeddings to be a list of floats").
        return [[float(x) for x in v] for v in vectors]
    except Exception:
        logger.debug("contextual embedding failed; falling back to stored-doc embedding")
        return None
