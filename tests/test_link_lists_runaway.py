"""Guards for the 2026-07-27 ``link_lists.bin`` runaway.

chroma-hnswlib's ``persistDirty`` skips non-dirty elements by seeking a stride
derived from in-memory element levels, while ``loadLinkLists`` reconstructs
those levels from the file itself with no validation (hnswalg.h:1365). One torn
record poisons the level table and every later persist seeks further past EOF,
growing the file sparsely without bound. On the canonical host it reached
231 GiB against a 58 MiB ceiling, filled the boot volume, starved macOS of swap
and caused three watchdog kernel panics.

Upstream: https://github.com/chroma-core/chroma/issues/7510
Recovery: docs/recovery/link-lists-runaway-disk-exhaustion.md
"""

import os
import struct


from mempalace.backends.chroma import (
    _link_lists_ceiling,
    reclaim_runaway_link_lists,
)

# Header values read from the segment that actually killed the host.
INCIDENT = {
    "cur_element_count": 178230,
    "maxlevel": 4,
    "maxM": 16,
    "size_data_per_element": 1676,
}


def _write_header(seg_dir, **overrides):
    """Write a chroma-hnswlib header.bin. Layout verified against live segments."""
    h = {**INCIDENT, **overrides}
    packed = struct.pack(
        "<I QQQQQQ iI QQQ d Q",
        1,  # version
        0,  # offset_level0
        262144,  # max_elements
        h["cur_element_count"],
        h["size_data_per_element"],
        1668,  # label_offset
        132,  # offset_data
        h["maxlevel"],
        45925,  # enterpoint_node
        h["maxM"],
        32,  # maxM0
        16,  # M
        0.3606737602222409,  # mult
        100,  # ef_construction
    )
    (seg_dir / "header.bin").write_bytes(packed)


def _sparse_file(path, apparent_bytes, allocated_bytes):
    """Create a sparse file: large apparent size, small real allocation.

    This is the shape that defeats ``st_size``-based detection, so the tests
    must reproduce it rather than just writing a big dense file.
    """
    with open(path, "wb") as fh:
        if allocated_bytes:
            fh.write(b"\xff" * allocated_bytes)
        if apparent_bytes > allocated_bytes:
            fh.seek(apparent_bytes - 1)
            fh.write(b"\0")


def test_ceiling_matches_hand_computation_for_the_incident_segment():
    """The bound is exact arithmetic from header.bin, not a tuned threshold."""
    # 178230 * (4 + (16*4+4) * (4+1)) = 178230 * 344
    assert _link_lists_ceiling(INCIDENT) == 178230 * 344 == 61_311_120


def test_reclaims_runaway_and_frees_the_blocks(tmp_path):
    seg = tmp_path / "9b682425-cac3-4409-a9fe-6e1cd4c48ef9"
    seg.mkdir()
    _write_header(seg)
    (seg / "data_level0.bin").write_bytes(b"\0" * 4096)
    runaway = seg / "link_lists.bin"
    # 512 MiB allocated — over the 256 MiB floor and ~35x the 58 MiB ceiling.
    _sparse_file(runaway, apparent_bytes=8 * 2**30, allocated_bytes=512 * 2**20)

    assert runaway.exists()
    reclaimed = reclaim_runaway_link_lists(str(tmp_path))

    assert str(runaway) in reclaimed
    assert not runaway.exists(), "the runaway file must actually be unlinked"
    # Everything salvageable is preserved — only the runaway file goes.
    assert (seg / "data_level0.bin").exists()
    assert (seg / "header.bin").exists()


def test_leaves_a_healthy_segment_untouched(tmp_path):
    """No false positives: a normal segment must survive the pass."""
    seg = tmp_path / "8b190816-4b80-41ca-97cc-fb62f3b22bab"
    seg.mkdir()
    _write_header(seg, cur_element_count=1515, maxlevel=3)
    (seg / "data_level0.bin").write_bytes(b"\0" * 2_539_140)
    healthy = seg / "link_lists.bin"
    healthy.write_bytes(b"\0" * 12_996)  # real size from a live segment

    assert reclaim_runaway_link_lists(str(tmp_path)) == []
    assert healthy.exists()


def test_small_files_never_trip_even_at_an_absurd_ratio(tmp_path):
    """A rebuild can leave the header describing fewer elements than
    link_lists.bin already covers. Without an absolute floor the guard would
    delete the index of the very rebuild it is protecting."""
    seg = tmp_path / "0c0502d3-0dc8-4768-968e-e3fd0f30c003"
    seg.mkdir()
    _write_header(seg, cur_element_count=5, maxlevel=0)  # ceiling = 5 * 72
    (seg / "data_level0.bin").write_bytes(b"\0" * 1024)
    small = seg / "link_lists.bin"
    small.write_bytes(b"\0" * (1024 * 1024))  # 1 MiB: ~2900x the ceiling

    assert reclaim_runaway_link_lists(str(tmp_path)) == []
    assert small.exists()


def test_reclaims_inside_a_quarantined_segment(tmp_path):
    """Quarantine renames aside but never frees — which is how 231 GiB of dead
    bytes sat unnoticed on the boot volume. The pass must reach those dirs."""
    seg = tmp_path / "9b682425-cac3-4409.corrupt-20260727-075647.drift-20260727-075647"
    seg.mkdir()
    _write_header(seg)
    (seg / "data_level0.bin").write_bytes(b"\0" * 4096)
    runaway = seg / "link_lists.bin"
    _sparse_file(runaway, apparent_bytes=4 * 2**30, allocated_bytes=400 * 2**20)

    assert str(runaway) in reclaim_runaway_link_lists(str(tmp_path))
    assert not runaway.exists()


def test_sparse_runaway_is_caught_on_blocks_not_apparent_size(tmp_path):
    """The reclaim pass must score allocated blocks.

    A runaway is sparse — 1.92 TiB apparent against 231 GiB real — so the
    ratio between the two is large and any check keying on one alone is
    reading a different number than the disk cares about. Here the allocation
    is what breaches the ceiling, and the file is 32x sparser than it looks.

    (Note: the incident write-up records a 258 MB apparent / 231 GiB allocated
    reading. That shape is not physically constructible — st_blocks cannot
    exceed the file length — which is why the recovery doc now treats that
    measurement as taken against the wrong path.)
    """
    seg = tmp_path / "11111111-2222-3333-4444-555555555555"
    seg.mkdir()
    _write_header(seg)
    (seg / "data_level0.bin").write_bytes(b"\0" * (4 * 2**20))
    runaway = seg / "link_lists.bin"
    _sparse_file(runaway, apparent_bytes=16 * 2**30, allocated_bytes=512 * 2**20)

    st = os.stat(runaway)
    assert st.st_blocks * 512 < st.st_size / 8, "precondition: file must be sparse"

    assert str(runaway) in reclaim_runaway_link_lists(str(tmp_path))
    assert not runaway.exists()


def test_missing_or_short_header_is_skipped(tmp_path):
    """No header means no exact bound, so the pass must not guess."""
    seg = tmp_path / "22222222-3333-4444-5555-666666666666"
    seg.mkdir()
    (seg / "data_level0.bin").write_bytes(b"\0" * 4096)
    big = seg / "link_lists.bin"
    _sparse_file(big, apparent_bytes=2 * 2**30, allocated_bytes=400 * 2**20)

    assert reclaim_runaway_link_lists(str(tmp_path)) == []
    assert big.exists()

    (seg / "header.bin").write_bytes(b"\x01\x02\x03")  # too short to unpack
    assert reclaim_runaway_link_lists(str(tmp_path)) == []
    assert big.exists()


def test_missing_palace_directory_is_silent(tmp_path):
    assert reclaim_runaway_link_lists(str(tmp_path / "nope")) == []
