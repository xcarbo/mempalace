"""Re-mines on a real palace: what each miner leaves behind for one file.

Pins the contract every miner shares through ``write_file_drawers`` (upstream
#2798): a re-mine replaces the file's drawers in its own scope, leaves other
scopes alone, and a failure or a kill at any point never leaves the file with
no drawers at all. The next mine always finishes the job.
"""

import json
import os
from pathlib import Path

import pytest

from mempalace.palace import (
    file_already_mined,
    get_closets_collection,
    get_collection,
    prefetch_mined_set,
)


class _Killed(BaseException):
    """Stands in for ``kill -9``: no ``except Exception`` handler runs."""


def _paragraphs(prefix: str, count: int) -> str:
    return "\n\n".join(f"{prefix} paragraph {i}. " + "verbatim words " * 40 for i in range(count))


def _touch(path):
    st = path.stat()
    os.utime(path, (st.st_atime, st.st_mtime + 10))


def _rows(palace, source_file):
    got = get_collection(palace).get(
        where={"source_file": str(source_file)}, include=["documents", "metadatas"]
    )
    return list(zip(got["documents"], got["metadatas"]))


@pytest.fixture(params=["sqlite_exact", "chroma"], autouse=True)
def backend(request, monkeypatch):
    monkeypatch.setenv("MEMPALACE_BACKEND", request.param)
    return request.param


class _Project:
    def __init__(self, tmp_path):
        from mempalace import miner

        self.miner = miner
        self.src = (tmp_path / "proj").resolve()
        self.src.mkdir()
        self.doc = self.src / "notes.md"
        self.palace = str(tmp_path / "palace")

    def write(self, text):
        self.doc.write_text(text, encoding="utf-8")

    def mine(self):
        self.miner.mine(str(self.src), self.palace, wing_override="proj", agent="test")

    def rows(self):
        return _rows(self.palace, self.doc)

    def mined(self):
        return file_already_mined(
            get_collection(self.palace), str(self.doc), check_mtime=True, extract_mode=None
        )


class _Convo:
    def __init__(self, tmp_path):
        from mempalace import convo_miner

        self.convo_miner = convo_miner
        self.src = (tmp_path / "chats").resolve()
        self.src.mkdir()
        self.doc = self.src / "chat.txt"
        self.palace = str(tmp_path / "palace")

    def write(self, turns, path=None):
        body = "\n\n".join(
            f"> question {i} " + "asked " * 30 + "\n" + "answer " * 60 for i in turns
        )
        (path or self.doc).write_text(body + "\n", encoding="utf-8")

    def mine(self, extract_mode="exchange"):
        self.convo_miner.mine_convos(
            str(self.src), self.palace, wing="chats", extract_mode=extract_mode
        )

    def rows(self, extract_mode="exchange"):
        return [
            (doc, meta)
            for doc, meta in _rows(self.palace, self.doc)
            if meta.get("extract_mode") == extract_mode
        ]

    def mined(self):
        return file_already_mined(
            get_collection(self.palace), str(self.doc), check_mtime=True, extract_mode="exchange"
        )


def _collection_type(palace):
    """The class under the embedding wrapper, where the writes end up."""
    col = get_collection(palace)
    return type(getattr(col, "_inner", col))


def _kill_deletes(monkeypatch, palace, error=_Killed):
    def die(self, *args, **kwargs):
        raise error("killed before the stale delete")

    monkeypatch.setattr(_collection_type(palace), "delete", die)


def _kill_upserts(monkeypatch, palace, error=_Killed):
    def die(self, *args, **kwargs):
        raise error("killed during the write")

    monkeypatch.setattr(_collection_type(palace), "upsert", die)


def _kill_updates(monkeypatch, palace):
    def die(self, *args, **kwargs):
        raise _Killed("killed before the final marker")

    monkeypatch.setattr(_collection_type(palace), "update", die)


# -- project miner: shrink, grow, reorder -------------------------------------


def test_project_remine_of_a_shorter_file_leaves_only_its_new_drawers(tmp_path):
    project = _Project(tmp_path)
    project.write(_paragraphs("long", 8))
    project.mine()
    assert len(project.rows()) > 2

    project.write(_paragraphs("short", 1))
    _touch(project.doc)
    project.mine()

    rows = project.rows()
    assert len(rows) == 1
    assert rows[0][0].startswith("short paragraph 0.")
    assert rows[0][1]["chunk_total"] == 1


def test_project_remine_of_a_longer_file_files_every_chunk(tmp_path):
    project = _Project(tmp_path)
    project.write(_paragraphs("short", 1))
    project.mine()

    project.write(_paragraphs("long", 8))
    _touch(project.doc)
    project.mine()

    rows = project.rows()
    assert len(rows) > 2
    assert all("long paragraph" in doc for doc, _ in rows)
    assert {meta["chunk_total"] for _, meta in rows} == {len(rows)}
    assert project.mined()


def test_project_remine_of_reordered_text_keeps_every_paragraph_once(tmp_path):
    project = _Project(tmp_path)
    words = [f"topic{i}" for i in range(6)]
    project.write("\n\n".join(f"{w} " + "verbatim words " * 40 for w in words))
    project.mine()
    count = len(project.rows())

    project.write("\n\n".join(f"{w} " + "verbatim words " * 40 for w in reversed(words)))
    _touch(project.doc)
    project.mine()

    rows = project.rows()
    assert len(rows) == count
    for word in words:
        assert sum(word in doc for doc, _ in rows) == 1, word
    assert {meta["chunk_total"] for _, meta in rows} == {count}
    assert project.mined()


def test_project_remine_replaces_every_drawer_for_the_path(tmp_path):
    """The project miner's purge scope is the whole path, whatever filed it."""
    project = _Project(tmp_path)
    project.write(_paragraphs("long", 2))
    project.mine()
    get_collection(project.palace).upsert(
        ids=["other_scope"],
        documents=["filed by another miner"],
        metadatas=[{"source_file": str(project.doc), "wing": "elsewhere", "chunk_index": 0}],
    )

    project.write(_paragraphs("new", 2))
    _touch(project.doc)
    project.mine()

    docs = [doc for doc, _ in project.rows()]
    assert "filed by another miner" not in docs
    assert all("new paragraph" in doc for doc in docs)


def test_project_remine_drops_metadata_the_new_text_no_longer_earns(tmp_path):
    """Chroma merges upserted metadata; a re-mine must not keep stale keys."""
    project = _Project(tmp_path)
    project.write(("We changed MemoryStack in rag/foo.py via do_thing_now(). " * 3 + "\n") * 4)
    project.mine()
    assert any(meta.get("entities") for _, meta in project.rows())

    project.write(_paragraphs("plain lowercase words only", 1))
    _touch(project.doc)
    project.mine()

    rows = project.rows()
    assert len(rows) == 1
    assert "entities" not in rows[0][1], rows[0][1]


def test_a_moved_drawer_does_not_stop_the_next_mine(tmp_path):
    """Moves keep the id, so the old row is there when the new one is written."""
    project = _Project(tmp_path)
    project.write(_paragraphs("old", 2))
    project.mine()
    col = get_collection(project.palace)
    got = col.get(where={"source_file": str(project.doc)}, include=["metadatas"])
    col.update(
        ids=got["ids"][:1],
        metadatas=[{**got["metadatas"][0], "wing": "moved", "room": "elsewhere"}],
    )

    project.write(_paragraphs("new", 2))
    _touch(project.doc)
    project.mine()

    assert all("new paragraph" in doc for doc, _ in project.rows())
    assert project.mined()


# -- project miner: failures and kills ----------------------------------------


def test_project_failed_write_keeps_the_previous_drawers(tmp_path, monkeypatch):
    project = _Project(tmp_path)
    project.write(_paragraphs("old", 3))
    project.mine()
    before = sorted(doc for doc, _ in project.rows())

    project.write(_paragraphs("new", 3))
    _touch(project.doc)
    _kill_upserts(monkeypatch, project.palace, error=RuntimeError)
    with pytest.raises(RuntimeError, match="killed during the write"):
        project.mine()

    assert sorted(doc for doc, _ in project.rows()) == before


def test_project_failed_stale_delete_keeps_text_and_the_next_mine_repairs_it(tmp_path, monkeypatch):
    project = _Project(tmp_path)
    project.write(_paragraphs("old", 3))
    project.mine()

    project.write(_paragraphs("new", 1))
    _touch(project.doc)
    _kill_deletes(monkeypatch, project.palace, error=RuntimeError)
    project.mine()
    assert project.rows(), "the file was left with no drawers"
    assert not project.mined()

    monkeypatch.undo()
    project.mine()
    rows = project.rows()
    assert len(rows) == 1
    assert rows[0][0].startswith("new paragraph 0.")
    assert project.mined()


def test_project_kill_between_write_and_prune_shows_both_sets_then_converges(tmp_path, monkeypatch):
    project = _Project(tmp_path)
    project.write(_paragraphs("old", 4))
    project.mine()
    assert len(project.rows()) == 4

    project.write(_paragraphs("new", 2))
    _touch(project.doc)
    _kill_deletes(monkeypatch, project.palace)
    with pytest.raises(_Killed):
        project.mine()
    monkeypatch.undo()

    docs = [doc for doc, _ in project.rows()]
    assert sum("new paragraph" in doc for doc in docs) == 2
    assert sum("old paragraph" in doc for doc in docs) == 2
    assert not project.mined()

    project.mine()
    rows = project.rows()
    assert len(rows) == 2
    assert all("new paragraph" in doc for doc, _ in rows)
    assert {meta["chunk_total"] for _, meta in rows} == {2}
    assert project.mined()


def test_search_after_a_kill_finds_both_sets_until_the_next_mine(tmp_path, monkeypatch):
    from mempalace.searcher import search_memories

    def texts(query):
        results = search_memories(query, project.palace, n_results=20)
        return " ".join(hit["text"] for hit in results["results"])

    project = _Project(tmp_path)
    project.write(_paragraphs("old", 4))
    project.mine()
    project.write(_paragraphs("new", 2))
    _touch(project.doc)
    _kill_deletes(monkeypatch, project.palace)
    with pytest.raises(_Killed):
        project.mine()
    monkeypatch.undo()

    assert "new paragraph 1." in texts("new paragraph 1")
    assert "old paragraph 3." in texts("old paragraph 3")

    project.mine()
    assert "old paragraph" not in texts("old paragraph 3")
    assert "new paragraph 1." in texts("new paragraph 1")


# -- conversation miner -------------------------------------------------------


def test_convo_remine_of_a_shorter_transcript_leaves_only_its_new_drawers(tmp_path):
    convo = _Convo(tmp_path)
    convo.write(range(6))
    convo.mine()
    assert len(convo.rows()) > 2

    convo.write([0])
    _touch(convo.doc)
    convo.mine()

    rows = convo.rows()
    assert len(rows) == 1
    assert "question 0" in rows[0][0]
    assert rows[0][1]["chunk_total"] == 1


def test_convo_remine_of_a_longer_transcript_files_every_exchange(tmp_path):
    convo = _Convo(tmp_path)
    convo.write([0])
    convo.mine()

    convo.write(range(6))
    _touch(convo.doc)
    convo.mine()

    rows = convo.rows()
    assert len(rows) == 6
    assert {meta["chunk_total"] for _, meta in rows} == {6}
    assert convo.mined()


def test_convo_remine_of_reordered_turns_keeps_every_exchange_once(tmp_path):
    convo = _Convo(tmp_path)
    convo.write(range(5))
    convo.mine()
    before = sorted(doc for doc, _ in convo.rows())

    convo.write(reversed(range(5)))
    _touch(convo.doc)
    convo.mine()

    rows = convo.rows()
    assert sorted(doc for doc, _ in rows) == before
    assert {meta["chunk_total"] for _, meta in rows} == {5}
    assert convo.mined()


def test_convo_remine_leaves_the_other_extract_mode_alone(tmp_path):
    convo = _Convo(tmp_path)
    convo.write(range(3))
    convo.mine(extract_mode="general")
    general = sorted(doc for doc, _ in convo.rows("general"))
    assert general

    convo.mine()
    convo.write([0])
    _touch(convo.doc)
    convo.mine()

    assert len(convo.rows()) == 1
    assert sorted(doc for doc, _ in convo.rows("general")) == general


def test_convo_failed_write_keeps_the_previous_drawers(tmp_path, monkeypatch):
    convo = _Convo(tmp_path)
    convo.write(range(3))
    convo.mine()
    before = sorted(doc for doc, _ in convo.rows())

    convo.write(range(5))
    _touch(convo.doc)
    _kill_upserts(monkeypatch, convo.palace, error=RuntimeError)
    with pytest.raises(RuntimeError, match="killed during the write"):
        convo.mine()

    assert sorted(doc for doc, _ in convo.rows()) == before


def test_convo_failed_stale_delete_keeps_text_and_the_next_mine_repairs_it(tmp_path, monkeypatch):
    convo = _Convo(tmp_path)
    convo.write(range(4))
    convo.mine()

    convo.write([0])
    _touch(convo.doc)
    _kill_deletes(monkeypatch, convo.palace, error=RuntimeError)
    convo.mine()
    assert convo.rows(), "the transcript was left with no drawers"
    assert not convo.mined()

    monkeypatch.undo()
    convo.mine()
    rows = convo.rows()
    assert len(rows) == 1
    assert "question 0" in rows[0][0]
    assert convo.mined()


def test_convo_kill_between_write_and_prune_reads_as_not_mined(tmp_path, monkeypatch):
    convo = _Convo(tmp_path)
    convo.write(range(6))
    convo.mine()

    convo.write([0, 1])
    _touch(convo.doc)
    _kill_deletes(monkeypatch, convo.palace)
    with pytest.raises(_Killed):
        convo.mine()
    monkeypatch.undo()

    assert len(convo.rows()) == 6, "old and new rows should both still be there"
    assert not convo.mined()
    col = get_collection(convo.palace)
    mined = prefetch_mined_set(col, extract_mode="exchange", source_files=[str(convo.doc)])
    assert mined.get(str(convo.doc)) != pytest.approx(convo.doc.stat().st_mtime)

    convo.mine()
    rows = convo.rows()
    assert len(rows) == 2
    assert {meta["chunk_total"] for _, meta in rows} == {2}
    assert convo.mined()


def test_convo_kill_before_the_final_marker_is_finished_by_the_next_mine(tmp_path, monkeypatch):
    convo = _Convo(tmp_path)
    convo.write(range(4))
    convo.mine()

    convo.write([0, 1])
    _touch(convo.doc)
    _kill_updates(monkeypatch, convo.palace)
    with pytest.raises(_Killed):
        convo.mine()
    monkeypatch.undo()

    assert {meta["chunk_total"] for _, meta in convo.rows()} == {3}
    assert not convo.mined()

    convo.mine()
    assert {meta["chunk_total"] for _, meta in convo.rows()} == {2}
    assert convo.mined()


def test_convo_kill_part_way_through_the_write_keeps_old_text_and_converges(tmp_path, monkeypatch):
    convo = _Convo(tmp_path)
    convo.write(range(6))
    convo.mine()
    old = {doc for doc, _ in convo.rows()}

    monkeypatch.setattr("mempalace.palace.DRAWER_UPSERT_BATCH_SIZE", 2)
    col_type = _collection_type(convo.palace)
    real_upsert = col_type.upsert
    calls = []

    def die_on_second(self, *args, **kwargs):
        calls.append(1)
        if len(calls) > 1:
            raise _Killed("killed between batches")
        return real_upsert(self, *args, **kwargs)

    convo.write(range(10, 13))
    _touch(convo.doc)
    monkeypatch.setattr(col_type, "upsert", die_on_second)
    with pytest.raises(_Killed):
        convo.mine()
    monkeypatch.setattr(col_type, "upsert", real_upsert)

    survivors = {doc for doc, _ in convo.rows()}
    assert len(survivors & old) == len(old) - 2, "only the overwritten ids may have moved"
    assert not convo.mined()

    convo.mine()
    rows = convo.rows()
    assert len(rows) == 3
    assert all(any(f"question {t}" in doc for t in (10, 11, 12)) for doc, _ in rows)
    assert convo.mined()


# -- half-written sets read as not mined --------------------------------------


def _fake_rows(col, source_file, mtime, totals):
    col.upsert(
        ids=[f"row_{i}" for i in range(len(totals))],
        documents=[f"text {i}" for i in range(len(totals))],
        metadatas=[
            {
                "source_file": source_file,
                "source_mtime": mtime,
                "chunk_total": total,
                "normalize_version": 3,
                "chunk_index": i,
            }
            for i, total in enumerate(totals)
        ],
    )


@pytest.mark.parametrize("totals", [[4, 4, 4, 5, 5], [5, 5, 4, 4, 4]])
def test_a_same_mtime_half_written_set_is_not_mined_in_any_row_order(tmp_path, totals):
    """New rows (3 drawers, marker 4) beside two old rows of a 5-row file, all
    under one mtime. Counting toward the lower marker would call it finished."""
    source = tmp_path / "chat.txt"
    source.write_text("text\n", encoding="utf-8")
    palace = str(tmp_path / "palace")
    col = get_collection(palace)
    _fake_rows(col, str(source), os.path.getmtime(source), totals)

    assert not file_already_mined(col, str(source), check_mtime=True)
    mined = prefetch_mined_set(col, source_files=[str(source)])
    assert str(source) not in mined


def test_a_marker_left_on_a_finished_set_costs_one_more_mine_and_no_text(tmp_path, monkeypatch):
    project = _Project(tmp_path)
    project.write(_paragraphs("old", 3))
    project.mine()
    col = get_collection(project.palace)
    got = col.get(where={"source_file": str(project.doc)}, include=["metadatas"])
    col.update(
        ids=got["ids"][:1],
        metadatas=[{**got["metadatas"][0], "chunk_total": 4}],
    )
    assert not project.mined()

    project.mine()
    assert len(project.rows()) == 3
    assert project.mined()


# -- the content-hash dedup gate (ours, no upstream equivalent) ---------------


def test_unchanged_chunks_are_not_their_own_duplicates_on_a_write_first_remine(tmp_path):
    """The file's own old rows are still there when the gate runs. If they
    counted as twins the chunk would be skipped, then pruned: the only copy."""
    convo = _Convo(tmp_path)
    convo.write(range(4))
    convo.mine()
    before = {doc for doc, _ in convo.rows()}

    convo.write(range(6))
    _touch(convo.doc)
    convo.mine()

    rows = convo.rows()
    assert before <= {doc for doc, _ in rows}
    assert len(rows) == 6
    assert {meta["chunk_total"] for _, meta in rows} == {6}
    assert not (Path(convo.palace) / "dedup_lineage.jsonl").exists()


def test_a_chunk_shared_with_another_file_keeps_exactly_one_copy_through_a_remine(tmp_path):
    convo = _Convo(tmp_path)
    twin = convo.src / "fork.txt"
    convo.write(range(3))
    convo.mine()
    canonical = sorted(doc for doc, _ in convo.rows())

    # The twin repeats the history and adds one turn. Its shared chunks skip.
    convo.write([*range(3), 3], path=twin)
    convo.mine()
    assert len(_rows(convo.palace, twin)) == 1

    # Grow the canonical file: its own rows must not read as duplicates, and
    # the twin's lone row must not suppress anything else.
    convo.write(range(3, 5))
    _touch(convo.doc)
    convo.mine()

    docs_everywhere = [doc for doc, _ in _rows(convo.palace, convo.doc)]
    docs_everywhere += [doc for doc, _ in _rows(convo.palace, twin)]
    for doc in {*canonical, *docs_everywhere}:
        assert docs_everywhere.count(doc) <= 1

    lineage_path = Path(convo.palace) / "dedup_lineage.jsonl"
    for line in lineage_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        assert record["source_file"] != record["duplicate_of_source"]


def test_a_chunk_dropped_by_the_gate_counts_out_of_chunk_total_on_a_remine(tmp_path):
    convo = _Convo(tmp_path)
    twin = convo.src / "fork.txt"
    convo.write(range(3), path=twin)
    convo.mine()

    convo.write(range(5))
    convo.mine()
    assert len(convo.rows()) == 2

    convo.write(range(6))
    _touch(convo.doc)
    convo.mine()

    rows = convo.rows()
    assert len(rows) == 3
    assert {meta["chunk_total"] for _, meta in rows} == {3}
    assert convo.mined()


def test_a_transcript_that_became_all_duplicates_keeps_its_twin_and_a_sentinel(tmp_path):
    convo = _Convo(tmp_path)
    twin = convo.src / "fork.txt"
    convo.write(range(3))
    convo.mine()
    convo.write(range(3, 7), path=twin)
    convo.mine()
    assert len(_rows(convo.palace, twin)) == 4

    # Every chunk now lives in the twin, so nothing is written here and the
    # old rows go. The twin's text stays, and the sentinel marks this mined.
    convo.write(range(3, 6))
    _touch(convo.doc)
    convo.mine()

    assert len(_rows(convo.palace, twin)) == 4
    assert [meta["ingest_mode"] for _, meta in convo.rows()] == ["registry"]
    assert convo.mined()


# -- closets ------------------------------------------------------------------


def _project_with_closet(tmp_path):
    project = _Project(tmp_path)
    project.write(_paragraphs("alpha", 3))
    project.mine()
    get_closets_collection(project.palace).upsert(
        ids=["closet_llm_notes"],
        documents=["deploy plan|alice|→drawer"],
        metadatas=[{"source_file": str(project.doc), "generated_by": "llm:test-model"}],
    )
    return project


def _closet_ids(project):
    """A mine closes the Chroma client, so each read opens the collection again."""
    got = get_closets_collection(project.palace).get(where={"source_file": str(project.doc)})
    return sorted(got["ids"])


def test_a_remine_that_changes_no_text_keeps_the_closets(tmp_path):
    project = _project_with_closet(tmp_path)
    before = _closet_ids(project)

    _touch(project.doc)
    project.mine()

    assert _closet_ids(project) == before
    assert project.mined()


def test_a_remine_of_rows_with_no_embedder_stamp_still_keeps_the_closets(tmp_path):
    """Those rows are embedded again, but their text did not change."""
    project = _project_with_closet(tmp_path)
    before = _closet_ids(project)
    col = get_collection(project.palace)
    got = col.get(where={"source_file": str(project.doc)}, include=["documents", "metadatas"])
    col.upsert(
        ids=got["ids"],
        documents=got["documents"],
        metadatas=[
            {k: v for k, v in meta.items() if k != "embedder_model"} or {"x": 1}
            for meta in got["metadatas"]
        ],
    )

    _touch(project.doc)
    project.mine()

    assert _closet_ids(project) == before


def test_a_remine_that_changes_text_rebuilds_the_closets(tmp_path):
    project = _project_with_closet(tmp_path)

    project.write(_paragraphs("beta", 3))
    _touch(project.doc)
    project.mine()

    ids = _closet_ids(project)
    assert "closet_llm_notes" not in ids
    assert ids


def test_a_retry_after_a_kill_rebuilds_the_closets(tmp_path, monkeypatch):
    project = _project_with_closet(tmp_path)

    project.write(_paragraphs("beta", 3))
    _touch(project.doc)
    _kill_updates(monkeypatch, project.palace)
    with pytest.raises(_Killed):
        project.mine()
    monkeypatch.undo()
    assert "closet_llm_notes" in _closet_ids(project)

    project.mine()

    assert "closet_llm_notes" not in _closet_ids(project)
    assert project.mined()


def test_a_file_with_no_closets_gets_them_on_a_remine_that_changes_no_text(tmp_path):
    project = _Project(tmp_path)
    project.write(_paragraphs("alpha", 3))
    project.mine()
    get_closets_collection(project.palace).delete(where={"source_file": str(project.doc)})
    assert _closet_ids(project) == []

    _touch(project.doc)
    project.mine()

    assert _closet_ids(project)


def test_a_failed_write_keeps_the_closets(tmp_path, monkeypatch):
    project = _project_with_closet(tmp_path)
    before = _closet_ids(project)

    project.write(_paragraphs("beta", 3))
    _touch(project.doc)
    _kill_upserts(monkeypatch, project.palace, error=RuntimeError)
    with pytest.raises(RuntimeError):
        project.mine()

    assert _closet_ids(project) == before


# -- write_file_drawers on each backend ---------------------------------------


def _backend_collection(name, tmp_path):
    """The collection a miner holds: the backend's, behind the embedding wrapper."""
    from mempalace.backends import PalaceRef, get_backend
    from mempalace.backends.embedding_wrapper import EmbeddingCollection

    return EmbeddingCollection(
        get_backend(name).get_collection(
            palace=PalaceRef(id=str(tmp_path), local_path=str(tmp_path)),
            collection_name="mempalace_drawers",
            create=True,
        )
    )


def _drawer(meta_extra=None):
    meta = {"source_file": "/src/notes.md", "chunk_index": 0, "wing": "w", "room": "r"}
    meta.update(meta_extra or {})
    return ("drawer_0", "verbatim text", meta)


def test_refiling_a_drawer_removes_keys_its_new_metadata_omits(tmp_path, backend):
    from mempalace.palace import write_file_drawers

    col = _backend_collection(backend, tmp_path)
    extra = {"entities": "Alice;Bob", "line_start": 3}
    write_file_drawers(col, "/src/notes.md", [_drawer(extra)])
    write_file_drawers(col, "/src/notes.md", [_drawer()])

    meta = col.get(ids=["drawer_0"], include=["metadatas"])["metadatas"][0]
    assert not set(extra) & set(meta), meta
    assert meta["chunk_total"] == 1


def test_a_failed_upsert_is_never_treated_as_written(tmp_path, backend):
    from mempalace.palace import write_file_drawers

    col = _backend_collection(backend, tmp_path)
    write_file_drawers(col, "/src/notes.md", [_drawer()])

    def fail(*args, **kwargs):
        raise ValueError("ID already exists")

    col.upsert = fail
    next_chunk = ("drawer_1", "new text", {**_drawer()[2], "chunk_index": 1})
    with pytest.raises(ValueError, match="already exists"):
        write_file_drawers(col, "/src/notes.md", [next_chunk])
    assert col.get(ids=["drawer_0"], include=["metadatas"])["ids"] == ["drawer_0"]


def test_a_dead_backend_during_the_prune_aborts_instead_of_failing_one_file(
    tmp_path, backend, monkeypatch
):
    from mempalace import palace
    from mempalace.backends import BackendError

    col = _backend_collection(backend, tmp_path)

    def dead(*args, **kwargs):
        raise BackendError("backend closed")

    monkeypatch.setattr(palace, "_source_file_delete_ids", dead)
    with pytest.raises(BackendError):
        palace.write_file_drawers(col, "/src/notes.md", [_drawer()])


def test_two_chunks_of_a_file_on_one_id_are_caught_across_batches(tmp_path, backend, monkeypatch):
    from mempalace.collision_scan import CollisionError
    from mempalace.palace import write_file_drawers

    monkeypatch.setattr("mempalace.palace.DRAWER_UPSERT_BATCH_SIZE", 1)
    col = _backend_collection(backend, tmp_path)
    first = _drawer()
    clash = ("drawer_0", "other text", {**_drawer()[2], "chunk_index": 1200})

    with pytest.raises(CollisionError):
        write_file_drawers(col, "/src/notes.md", [first, clash])
    assert col.get(ids=["drawer_0"])["ids"] == []


def test_writing_without_prune_removes_nothing_and_sets_no_marker(tmp_path, backend):
    from mempalace.palace import write_file_drawers

    col = _backend_collection(backend, tmp_path)
    write_file_drawers(col, "/src/notes.md", [_drawer()])
    other = ("drawer_1", "second", {**_drawer()[2], "chunk_index": 1})
    write_file_drawers(col, "/src/notes.md", [other], prune=False)

    assert sorted(col.get(where={"source_file": "/src/notes.md"})["ids"]) == [
        "drawer_0",
        "drawer_1",
    ]
    meta = col.get(ids=["drawer_1"], include=["metadatas"])["metadatas"][0]
    assert "chunk_total" not in meta


# -- format miner -------------------------------------------------------------


def test_format_failed_write_keeps_the_previous_drawers(tmp_path, monkeypatch):
    from mempalace import format_miner

    source = tmp_path / "report.pdf"
    source.write_text("placeholder\n", encoding="utf-8")
    palace = str(tmp_path / "palace")
    col = get_collection(palace)
    chunks = [{"content": f"page {i} " * 30, "chunk_index": i} for i in range(3)]
    format_miner._file_chunks_locked(
        col, str(source), chunks, "docs", "documents", "agent", source_mtime=1.0
    )
    assert len(_rows(palace, source)) == 3

    _kill_upserts(monkeypatch, palace)
    with pytest.raises(_Killed):
        format_miner._file_chunks_locked(
            col,
            str(source),
            [{"content": "rewritten " * 30, "chunk_index": 0}],
            "docs",
            "documents",
            "agent",
            source_mtime=2.0,
        )
    monkeypatch.undo()

    assert len(_rows(palace, source)) == 3


def test_format_remine_leaves_drawers_another_miner_filed(tmp_path):
    from mempalace import format_miner

    source = tmp_path / "report.pdf"
    source.write_text("placeholder\n", encoding="utf-8")
    palace = str(tmp_path / "palace")
    col = get_collection(palace)
    col.upsert(
        ids=["project_row"],
        documents=["filed by the project miner"],
        metadatas=[{"source_file": str(source), "chunk_index": 0, "extract_mode": "project"}],
    )
    chunks = [{"content": f"page {i} " * 30, "chunk_index": i} for i in range(2)]
    format_miner._file_chunks_locked(
        col, str(source), chunks, "docs", "documents", "agent", source_mtime=1.0
    )

    docs = [doc for doc, _ in _rows(palace, source)]
    assert "filed by the project miner" in docs
    assert len(docs) == 3
