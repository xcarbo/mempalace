"""Tests for the palace time machine (mempalace/asof.py)."""

import pytest

from mempalace.asof import _normalize_cutoff, snapshot


@pytest.fixture
def dated_collection(collection):
    """Drawers filed across three dates, one wing, plus a roadmap history."""
    collection.add(
        ids=[
            "drawer_proj_technical_early",
            "drawer_proj_technical_mid",
            "drawer_proj_decisions_mid",
            "drawer_proj_roadmap_v1",
            "drawer_proj_roadmap_v2",
            "drawer_proj_technical_late",
        ],
        documents=[
            "early technical note",
            "mid technical note",
            "mid decision record",
            "roadmap version one: ship the parser",
            "roadmap version two: ship the exporter",
            "late technical note",
        ],
        metadatas=[
            {"wing": "proj", "room": "technical", "filed_at": "2026-03-01T10:00:00"},
            {"wing": "proj", "room": "technical", "filed_at": "2026-04-10T10:00:00"},
            {"wing": "proj", "room": "decisions", "filed_at": "2026-04-12T10:00:00"},
            {"wing": "proj", "room": "roadmap", "filed_at": "2026-03-15T10:00:00"},
            {"wing": "proj", "room": "roadmap", "filed_at": "2026-05-20T10:00:00"},
            {"wing": "proj", "room": "technical", "filed_at": "2026-06-01T10:00:00"},
        ],
    )
    return collection


class TestNormalizeCutoff:
    def test_bare_date_becomes_end_of_day(self):
        assert _normalize_cutoff("2026-05-01") == "2026-05-01T23:59:59.999999"

    def test_iso_datetime_passthrough(self):
        assert _normalize_cutoff("2026-05-01T12:00:00") == "2026-05-01T12:00:00"

    def test_garbage_raises(self):
        with pytest.raises(ValueError):
            _normalize_cutoff("last tuesday")


class TestSnapshot:
    def test_counts_respect_cutoff(self, palace_path, dated_collection):
        snap = snapshot(palace_path, "2026-04-30", wing="proj")
        # early, mid x2, roadmap_v1 — not roadmap_v2 or late.
        assert snap["total_at_date"] == 4
        assert snap["total_now"] == 6
        assert snap["rooms"] == {"technical": 2, "decisions": 1, "roadmap": 1}

    def test_latest_sorted_desc(self, palace_path, dated_collection):
        snap = snapshot(palace_path, "2026-04-30", wing="proj", latest=3)
        filed = [e["filed_at"] for e in snap["latest"]]
        assert filed == sorted(filed, reverse=True)
        assert snap["latest"][0]["drawer_id"] == "drawer_proj_decisions_mid"

    def test_roadmap_as_it_stood(self, palace_path, dated_collection):
        snap = snapshot(palace_path, "2026-04-30", wing="proj")
        assert snap["roadmap"]["drawer_id"] == "drawer_proj_roadmap_v1"
        assert "version one" in snap["roadmap"]["content"]
        # After v2 lands, the same query at a later date sees v2.
        later = snapshot(palace_path, "2026-06-30", wing="proj")
        assert later["roadmap"]["drawer_id"] == "drawer_proj_roadmap_v2"

    def test_end_of_day_inclusive(self, palace_path, dated_collection):
        snap = snapshot(palace_path, "2026-03-01", wing="proj")
        assert snap["total_at_date"] == 1
        assert snap["latest"][0]["drawer_id"] == "drawer_proj_technical_early"

    def test_bad_date_raises(self, palace_path, dated_collection):
        with pytest.raises(ValueError, match="unparseable date"):
            snapshot(palace_path, "soonish", wing="proj")

    def test_kg_facts_at_date(self, palace_path, dated_collection, tmp_path):
        from mempalace.knowledge_graph import KnowledgeGraph

        kg_path = str(tmp_path / "kg.sqlite3")
        kg = KnowledgeGraph(db_path=kg_path)
        kg.add_triple("wing_proj", "uses_backend", "chroma", valid_from="2026-01-01")
        kg.add_triple(
            "wing_proj",
            "uses_backend",
            "milvus",
            valid_from="2026-06-01",
        )

        snap = snapshot(palace_path, "2026-04-30", wing="proj", kg_path=kg_path)
        objs = [f["object"] for f in snap["kg_facts"]]
        assert "chroma" in objs
        assert "milvus" not in objs
