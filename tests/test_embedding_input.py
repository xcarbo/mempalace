"""
test_embedding_input.py — contextual headers for the embedder (template flavor).

The contract under test, in order of importance:

1. VERBATIM: the stored document is byte-identical to what the caller
   passed — headers exist only in the embedding input.
2. The embedding actually changes: chunk vectors match the header-prefixed
   text, not the bare chunk.
3. Fail-open: flag off, empty inputs, or an embedding error all fall back
   to embedding the stored documents (returns None), never blocking a write.
"""

import pytest

from mempalace.embedding_input import (
    contextual_embedding_texts,
    contextual_header,
    drawer_title,
    maybe_contextual_embeddings,
)


class TestDrawerTitle:
    def test_strips_markdown_heading_markers(self):
        assert drawer_title("# mempalace — Roadmap (canonical)\nbody") == (
            "mempalace — Roadmap (canonical)"
        )

    def test_first_nonempty_line_wins(self):
        assert drawer_title("\n\n  ## Setup notes  \nmore") == "Setup notes"

    def test_collapses_internal_whitespace(self):
        assert drawer_title("a   b\t c") == "a b c"

    def test_caps_length(self):
        long_line = "word " * 50
        title = drawer_title(long_line)
        assert len(title) <= 80
        assert title.endswith("…")

    def test_empty_content_gives_empty_title(self):
        assert drawer_title("") == ""
        assert drawer_title("\n\n \n") == ""


class TestContextualTexts:
    def test_header_shape(self):
        assert contextual_header("mempalace", "roadmap", "The Roadmap", 3, 23) == (
            "mempalace/roadmap — The Roadmap, part 3/23: "
        )

    def test_headerless_title_omits_dash_segment(self):
        assert contextual_header("w", "r", "", 1, 2) == "w/r, part 1/2: "

    def test_texts_prefix_each_chunk(self):
        content = "# My Doc\nalpha\nbeta"
        texts = contextual_embedding_texts("w", "r", content, ["alpha", "beta"])
        assert texts == [
            "w/r — My Doc, part 1/2: alpha",
            "w/r — My Doc, part 2/2: beta",
        ]


class _Cfg:
    def __init__(self, on=True):
        self.embed_context_headers = on


class TestMaybeContextualEmbeddings:
    def test_flag_off_returns_none(self):
        assert maybe_contextual_embeddings(_Cfg(on=False), "w", "r", "c", ["c"]) is None

    def test_empty_chunks_returns_none(self):
        assert maybe_contextual_embeddings(_Cfg(), "w", "r", "c", []) is None

    def test_none_config_returns_none(self):
        assert maybe_contextual_embeddings(None, "w", "r", "c", ["c"]) is None

    def test_embedding_error_fails_open(self, monkeypatch):
        import mempalace.embedding as embedding

        def boom(*a, **kw):
            raise RuntimeError("model exploded")

        monkeypatch.setattr(embedding, "get_embedding_function", boom)
        assert maybe_contextual_embeddings(_Cfg(), "w", "r", "c", ["c"]) is None

    def test_vectors_match_header_prefixed_text(self):
        """The returned vectors are the embeddings of header+chunk, byte-for-
        byte the same EF the collection resolves — and differ from the bare
        chunk's embedding."""
        pytest.importorskip("onnxruntime")
        from mempalace.embedding import get_embedding_function

        content = "# Test Doc\nSome body text about memory palaces."
        chunks = ["Some body text about memory palaces."]
        vectors = maybe_contextual_embeddings(_Cfg(), "wing", "room", content, chunks)
        assert vectors is not None and len(vectors) == 1

        ef = get_embedding_function()
        expected = list(ef(input=["wing/room — Test Doc, part 1/1: " + chunks[0]])[0])
        bare = list(ef(input=chunks)[0])
        assert vectors[0] == pytest.approx(expected, abs=1e-6)
        assert vectors[0] != pytest.approx(bare, abs=1e-3)
