"""Replaying an identical closed (historical) assertion must not add a row.

Upstream #2605 (663d8a12). ``add_triple`` only deduplicated open rows
(``valid_to IS NULL``). A retried ``kg-add`` with ``valid_to`` set (a daemon
batch re-run, a script re-run after an interruption) stored a second row
with a new id, so the same fact showed twice in every query of that window.
"""


def test_identical_closed_interval_replay_returns_existing_id(kg):
    kwargs = dict(
        valid_from="2025-01-01T00:00:00Z",
        valid_to="2025-02-01T00:00:00Z",
        source_closet="synthetic-replay-fixture",
    )
    tid1 = kg.add_triple("Pilot History", "was", "Archived", **kwargs)
    tid2 = kg.add_triple("Pilot History", "was", "Archived", **kwargs)
    assert tid1 == tid2
    rows = kg.query_entity("Pilot History", as_of="2025-01-15T00:00:00Z")
    assert len(rows) == 1


def test_date_only_closed_replay_returns_existing_id(kg):
    tid1 = kg.add_triple(
        "Alice", "worked_at", "Acme", valid_from="2020-01-01", valid_to="2021-01-01"
    )
    tid2 = kg.add_triple(
        "Alice", "worked_at", "Acme", valid_from="2020-01-01", valid_to="2021-01-01"
    )
    assert tid1 == tid2


def test_distinct_closed_intervals_stay_separate(kg):
    tid1 = kg.add_triple(
        "Alice",
        "lived_in",
        "Rome",
        valid_from="2020-01-01T00:00:00Z",
        valid_to="2020-12-31T00:00:00Z",
    )
    tid2 = kg.add_triple(
        "Alice",
        "lived_in",
        "Rome",
        valid_from="2023-01-01T00:00:00Z",
        valid_to="2023-12-31T00:00:00Z",
    )
    assert tid1 != tid2


def test_closed_replay_with_different_provenance_stays_separate(kg):
    common = dict(valid_from="2020-01-01T00:00:00Z", valid_to="2020-12-31T00:00:00Z")
    tid1 = kg.add_triple("Alice", "lived_in", "Rome", source_closet="notebook-a", **common)
    tid2 = kg.add_triple("Alice", "lived_in", "Rome", source_closet="notebook-b", **common)
    assert tid1 != tid2


def test_closed_replay_with_different_confidence_stays_separate(kg):
    common = dict(valid_from="2020-01-01", valid_to="2020-12-31")
    tid1 = kg.add_triple("Alice", "lived_in", "Rome", confidence=1.0, **common)
    tid2 = kg.add_triple("Alice", "lived_in", "Rome", confidence=0.5, **common)
    assert tid1 != tid2
