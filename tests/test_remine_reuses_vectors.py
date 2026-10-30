"""A re-mine embeds only the text that changed, and never reuses a wrong vector.

From upstream #2813, without its NORMALIZE_VERSION bump (ours is already 3,
with another meaning). Upstream reuses a vector when the text and the
dimension match. A same-dimension model swap passes that check, so here a
vector is reused only when the row also says which model made it, and that
model is the one in use. A row with no such stamp is embedded once more.
"""

import pytest

import mempalace.backends.embedding_wrapper as embedding_wrapper
import mempalace.embedding as embedding_mod
from mempalace.palace import (
    EMBEDDER_STAMP_KEY,
    _reusable_vector,
    file_already_mined,
    get_collection,
    set_palace_embedder_identity,
)


@pytest.fixture(autouse=True)
def sqlite_exact(monkeypatch):
    monkeypatch.setenv("MEMPALACE_BACKEND", "sqlite_exact")


def _write_chat(path, turns):
    body = "\n\n".join(f"> question {i} " + "asked " * 30 + "\n" + "answer " * 60 for i in turns)
    path.write_text(body + "\n", encoding="utf-8")


@pytest.fixture
def embedded(monkeypatch):
    texts = []
    real = embedding_wrapper._embed_texts

    def counting(batch, is_query=False):
        if not is_query:
            texts.extend(batch)
        return real(batch, is_query=is_query)

    monkeypatch.setattr(embedding_wrapper, "_embed_texts", counting)
    return texts


def _mine(src, palace):
    from mempalace import convo_miner

    convo_miner.mine_convos(str(src), palace, wing="chats", extract_mode="exchange")


def _bump_mtimes(src):
    import os

    for path in src.iterdir():
        st = path.stat()
        os.utime(path, (st.st_atime, st.st_mtime + 10))


def _palace_rows(palace):
    got = get_collection(palace).get(include=["documents", "metadatas", "embeddings"])
    return {
        drawer_id: (doc, meta, list(vector))
        for drawer_id, doc, meta, vector in zip(
            got["ids"], got["documents"], got["metadatas"], got["embeddings"]
        )
    }


def _strip_stamps(palace):
    col = get_collection(palace)
    got = col.get(include=["documents", "metadatas", "embeddings"])
    col.upsert(
        ids=got["ids"],
        documents=got["documents"],
        metadatas=[
            {k: v for k, v in meta.items() if k != EMBEDDER_STAMP_KEY} for meta in got["metadatas"]
        ],
        embeddings=got["embeddings"],
    )


def _chats(tmp_path, turns=range(3), names=("a.txt",)):
    src = (tmp_path / "chats").resolve()
    src.mkdir()
    for name in names:
        _write_chat(src / name, turns)
    return src, str(tmp_path / "palace")


def test_a_touched_transcript_embeds_nothing_and_keeps_its_vectors(tmp_path, embedded):
    src, palace = _chats(tmp_path)
    _write_chat(src / "b.txt", range(10, 13))
    _mine(src, palace)
    first = _palace_rows(palace)
    assert first and len(embedded) == len(first)
    assert {meta[EMBEDDER_STAMP_KEY] for _, meta, _ in first.values()} == {
        embedding_mod.current_model_name()
    }

    _bump_mtimes(src)
    embedded.clear()
    _mine(src, palace)

    assert embedded == []
    second = _palace_rows(palace)
    assert set(second) == set(first)
    for drawer_id, (doc, _, vector) in second.items():
        assert doc == first[drawer_id][0]
        assert vector == pytest.approx(first[drawer_id][2])


def test_a_grown_transcript_embeds_only_the_new_chunks(tmp_path, embedded):
    src, palace = _chats(tmp_path)
    chat = src / "a.txt"
    _mine(src, palace)
    before = _palace_rows(palace)

    _write_chat(chat, range(5))
    embedded.clear()
    _mine(src, palace)

    after = _palace_rows(palace)
    changed = [
        doc for drawer_id, (doc, _, _) in after.items() if before.get(drawer_id, (None,))[0] != doc
    ]
    assert changed
    assert sorted(embedded) == sorted(changed)
    assert len(embedded) < len(after)
    assert file_already_mined(
        get_collection(palace), str(chat), check_mtime=True, extract_mode="exchange"
    )


def test_reordered_text_gets_the_vector_of_its_own_text(tmp_path, embedded):
    """Same ids, different text at each one: nothing may keep the old vector."""
    src, palace = _chats(tmp_path, turns=range(4))
    _mine(src, palace)
    by_text = {doc: vector for doc, _, vector in _palace_rows(palace).values()}

    _write_chat(src / "a.txt", reversed(range(4)))
    _mine(src, palace)

    for doc, _, vector in _palace_rows(palace).values():
        assert vector == pytest.approx(by_text[doc])


def test_a_row_with_no_stamp_is_embedded_again_then_stamped(tmp_path, embedded):
    src, palace = _chats(tmp_path)
    _mine(src, palace)
    count = len(embedded)
    _strip_stamps(palace)

    _bump_mtimes(src)
    embedded.clear()
    _mine(src, palace)

    assert len(embedded) == count
    stamps = {meta.get(EMBEDDER_STAMP_KEY) for _, meta, _ in _palace_rows(palace).values()}
    assert stamps == {embedding_mod.current_model_name()}

    _bump_mtimes(src)
    embedded.clear()
    _mine(src, palace)
    assert embedded == []


def test_a_same_dimension_model_swap_embeds_again(tmp_path, embedded, monkeypatch):
    """``set-embedder --force`` moves the palace to a new model and leaves the
    old vectors in place. minilm and bge-small are both 384-d, so a dimension
    check alone would keep the old model's vectors forever."""
    src, palace = _chats(tmp_path)
    monkeypatch.setenv("MEMPALACE_EMBEDDING_MODEL", "minilm")
    _mine(src, palace)
    count = len(embedded)

    monkeypatch.setenv("MEMPALACE_EMBEDDING_MODEL", "bge-small")
    set_palace_embedder_identity(palace, model="bge-small", force=True)
    _bump_mtimes(src)
    embedded.clear()
    _mine(src, palace)

    assert len(embedded) == count, "a stored vector from the old model was reused"
    stamps = {meta.get(EMBEDDER_STAMP_KEY) for _, meta, _ in _palace_rows(palace).values()}
    assert stamps == {"bge-small"}


def test_a_stored_vector_of_another_dimension_is_embedded_again(tmp_path, embedded, monkeypatch):
    src, palace = _chats(tmp_path)
    _mine(src, palace)
    count = len(embedded)

    monkeypatch.setattr(embedding_mod, "probe_dimension", lambda *a, **k: 7)
    _bump_mtimes(src)
    embedded.clear()
    _mine(src, palace)

    assert len(embedded) == count


def test_a_contextual_vector_is_never_reused_for_a_mined_row(tmp_path, embedded):
    """MCP drawers embed a breadcrumb plus the text, and carry no stamp. A row
    with that vector must not pass as the plain embedding of its text."""
    src, palace = _chats(tmp_path)
    _mine(src, palace)
    col = get_collection(palace)
    got = col.get(include=["documents", "metadatas"])
    wrong = [[1.0] + [0.0] * 383 for _ in got["ids"]]
    col.upsert(
        ids=got["ids"],
        documents=got["documents"],
        metadatas=[
            {k: v for k, v in meta.items() if k != EMBEDDER_STAMP_KEY} for meta in got["metadatas"]
        ],
        embeddings=wrong,
    )

    _bump_mtimes(src)
    _mine(src, palace)

    for _, _, vector in _palace_rows(palace).values():
        assert vector != pytest.approx(wrong[0])


def test_a_failed_vector_read_embeds_instead_of_failing_the_mine(tmp_path, embedded, monkeypatch):
    from mempalace import palace as palace_mod

    src, palace = _chats(tmp_path)
    _mine(src, palace)
    count = len(embedded)

    monkeypatch.setattr(palace_mod, "_stored_vectors", lambda collection, ids: {})
    _bump_mtimes(src)
    embedded.clear()
    _mine(src, palace)

    assert len(embedded) == count


def test_a_reused_vector_still_drops_a_key_the_new_metadata_lacks(tmp_path, embedded):
    src, palace = _chats(tmp_path)
    _mine(src, palace)
    col = get_collection(palace)
    got = col.get(include=["metadatas"])
    col.update(
        ids=got["ids"], metadatas=[{**meta, "stale_key": "left over"} for meta in got["metadatas"]]
    )

    _bump_mtimes(src)
    embedded.clear()
    _mine(src, palace)

    assert embedded == []
    for _, meta, _ in _palace_rows(palace).values():
        assert "stale_key" not in meta


def test_chroma_reuses_a_stored_vector_too(tmp_path, monkeypatch):
    """Chroma hands back numpy arrays and merges metadata on upsert."""
    from mempalace.palace import write_file_drawers

    monkeypatch.setenv("MEMPALACE_BACKEND", "chroma")
    col = get_collection(str(tmp_path / "palace"))
    drawer = ("drawer_0", "verbatim", {"source_file": "/s.md", "chunk_index": 0})
    write_file_drawers(col, "/s.md", [drawer])
    before = col.get(ids=["drawer_0"], include=["embeddings"])["embeddings"][0]

    filed = write_file_drawers(col, "/s.md", [drawer])

    after = col.get(ids=["drawer_0"], include=["embeddings", "metadatas"])
    assert list(after["embeddings"][0]) == pytest.approx(list(before))
    assert after["metadatas"][0]["chunk_total"] == 1
    assert filed.unchanged is True


def test_reusable_vector_needs_the_current_dimension():
    assert _reusable_vector([0.1, 0.2], 2)
    assert not _reusable_vector([], 2)
    assert not _reusable_vector(None, 2)
    assert not _reusable_vector([0.1, 0.2, 0.3], 2)
    assert not _reusable_vector([0.1, 0.2], 0)
