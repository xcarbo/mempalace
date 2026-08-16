"""
conftest.py — Shared fixtures for MemPalace tests.

Provides isolated palace and knowledge graph instances so tests never
touch the user's real data or leak temp files on failure.

HOME is redirected to a temp directory at module load time — before any
mempalace imports — so that module-level initialisations (e.g.
``_kg = KnowledgeGraph()`` in mcp_server) write to a throwaway location
instead of the real user profile.
"""

import os
import hashlib
import math
import re
import shutil
import tempfile

# ── Isolate HOME before any mempalace imports ──────────────────────────
_original_env = {}
_session_tmp = tempfile.mkdtemp(prefix="mempalace_session_")

for _var in ("HOME", "USERPROFILE", "HOMEDRIVE", "HOMEPATH"):
    _original_env[_var] = os.environ.get(_var)

os.environ["HOME"] = _session_tmp
os.environ["USERPROFILE"] = _session_tmp
os.environ["HOMEDRIVE"] = os.path.splitdrive(_session_tmp)[0] or "C:"
os.environ["HOMEPATH"] = os.path.splitdrive(_session_tmp)[1] or _session_tmp

# Now it is safe to import mempalace modules that trigger initialisation.
import chromadb  # noqa: E402
import pytest  # noqa: E402

from mempalace.config import MempalaceConfig  # noqa: E402
from mempalace.knowledge_graph import KnowledgeGraph  # noqa: E402

_TEST_EMBED_DIM = 384
_TEST_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_REAL_EMBEDDING_TEST_MODULES = {
    "test_embedding",
    "test_embedding_api",
    "test_embeddinggemma",
    # Ours. The autouse fixture below replaces get_embedding_function and
    # _embed_texts with a deterministic stub for every module NOT listed here —
    # which is exactly what these modules assert on, so without the opt-out
    # they test the stub instead of the code (isinstance checks against
    # BgeSmallONNX / NomicEmbedONNX fail, and _embed_texts' is_query routing is
    # never reached). Added at the 3.7.1 merge, when the fixture arrived.
    "test_bge_small",
    "test_nomic_embed",
    "test_embedder_identity",
    "test_embedding_wrapper",
}


def _stable_test_embedding(text: str) -> list[float]:
    """Small deterministic embedding for tests that do not test ONNX itself."""
    vec = [0.0] * _TEST_EMBED_DIM
    tokens = _TEST_TOKEN_RE.findall((text or "").lower())
    if not tokens:
        tokens = [""]
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        vec[int.from_bytes(digest[:4], "little") % _TEST_EMBED_DIM] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class _StableTestEmbeddingFunction:
    @staticmethod
    def name() -> str:
        return "default"

    @staticmethod
    def build_from_config(config):
        _StableTestEmbeddingFunction.validate_config(config)
        return _StableTestEmbeddingFunction()

    @staticmethod
    def validate_config(config) -> None:
        return

    def get_config(self) -> dict:
        return {}

    def is_legacy(self) -> bool:
        return False

    def default_space(self) -> str:
        return "cosine"

    def supported_spaces(self) -> list[str]:
        return ["cosine", "l2", "ip"]

    def embed_query(self, input):
        return self(input=input)

    def __call__(self, input):
        return [_stable_test_embedding(str(text)) for text in list(input or [])]


# Redirect ChromaDB's ONNX model cache back to the real user's cache so tests
# don't re-download the 79 MB model on every run. The HOME redirect above
# would otherwise point ONNXMiniLM_L6_V2.DOWNLOAD_PATH at the empty temp dir.
try:
    from pathlib import Path  # noqa: E402
    from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import (  # noqa: E402
        ONNXMiniLM_L6_V2,
    )

    _real_home = _original_env.get("USERPROFILE") or _original_env.get("HOME")
    if _real_home:
        _real_cache = Path(_real_home) / ".cache" / "chroma" / "onnx_models" / "all-MiniLM-L6-v2"
        if _real_cache.exists():
            ONNXMiniLM_L6_V2.DOWNLOAD_PATH = _real_cache
except ImportError:
    pass


@pytest.fixture(autouse=True)
def _no_retrieval_log(monkeypatch):
    """Never append test noise to the user's real retrieval log.

    Tests that exercise the flight recorder opt back in by deleting the
    env var under a scratch HOME (see test_retrieval_log.scratch_home).
    """
    monkeypatch.setenv("MEMPALACE_RETRIEVAL_LOG", "0")


@pytest.fixture(autouse=True)
def _no_local_rerank(monkeypatch):
    """Never let the ambient shell's local reranker decide test outcomes.

    ``rerank_enabled()`` keys off MEMPALACE_RERANK_URL + MEMPALACE_RERANK_MODEL,
    which this machine's zshrc exports for every interactive shell and agent.
    With them set, ranking assertions were silently re-scored by LM Studio:
    ``test_effective_distance_clamped_to_valid_cosine_range`` and
    ``test_search_union_uses_sqlite_exact_lexical_search`` failed locally while
    passing in CI, and the results also depended on :1234 being up. Tests that
    exercise reranking set these vars themselves.
    """
    monkeypatch.delenv("MEMPALACE_RERANK_URL", raising=False)
    monkeypatch.delenv("MEMPALACE_RERANK_MODEL", raising=False)


# Ambient MEMPALACE_* variables that must never reach a test. Every one of
# these is read by production code, so an exported value silently re-points or
# re-configures the system under test. Deliberate opt-ins
# (``MEMPALACE_*_LIVE_URL`` / ``_LIVE_DSN``) are NOT listed — those exist to be
# set from the shell.
_AMBIENT_VARS_TO_SCRUB = (
    # Palace location. The dangerous one: see _no_ambient_config below.
    "MEMPALACE_PALACE_PATH",
    "MEMPAL_PALACE_PATH",
    "MEMPAL_DIR",
    # Storage backend selection — flips chroma tests onto another backend.
    "MEMPALACE_BACKEND",
    "MEMPALACE_BACKEND_EXPLICIT",
    # Embedding identity — changes vectors, and with them every ranking
    # assertion, exactly like the rerank leak did.
    "MEMPALACE_EMBEDDING_MODEL",
    "MEMPALACE_EMBEDDING_DEVICE",
    "MEMPALACE_EMBEDDING_THREADS",
    # Ingest shape.
    "MEMPALACE_MAX_CHUNKS_PER_FILE",
    "MEMPALACE_TOPIC_TUNNEL_MIN_COUNT",
    "MEMPALACE_SOURCE_DIR",
    # Write path.
    "MEMPALACE_WRITE_LOG",
    "MEMPALACE_WRITE_ROUTING",
    "MEMPALACE_CLI_WRITE_ROUTING",
    "MEMPALACE_HOOK_WRITE_ROUTING",
    # A low watchdog budget makes a slow write os._exit the whole test process.
    "MEMPALACE_WRITE_WATCHDOG_SECONDS",
    # MCP surface.
    "MEMPALACE_MCP_READ_ONLY",
    "MEMPALACE_MCP_ALLOW_PEER_WRITER",
    "MEMPALACE_MCP_HTTP_TOKEN",
    # Hooks / daemon / process plumbing.
    "MEMPALACE_HOOKS_AUTO_SAVE",
    "MEMPALACE_HOOKS_DAEMON",
    "MEMPALACE_DAEMON_STATE_ROOT",
    "MEMPALACE_MINE_PID_FILE",
    "MEMPALACE_MINE_TIMEOUT_HOURS",
    "MEMPALACE_PYTHON",
    "MEMPALACE_LOG_FILE",
    # Outbox — an exported URL would make tests emit to a real endpoint.
    "MEMPALACE_OUTBOX_URL",
    "MEMPALACE_OUTBOX_SECRET",
    "MEMPALACE_OUTBOX_SECRET_FILE",
    "MEMPALACE_OUTBOX_WINGS",
    # Ranking behavior — an exported fusion/blend override silently re-scores
    # every ranking assertion, exactly like the rerank leak did.
    "MEMPALACE_FUSION",
    "MEMPALACE_RRF_K",
    "MEMPALACE_RERANK_BLEND",
    "MEMPALACE_ARCHIVE_WINGS",
    "MEMPALACE_ARCHIVE_RANK_PENALTY",
)


@pytest.fixture(autouse=True)
def _no_ambient_config(monkeypatch):
    """Strip ambient MEMPALACE_* configuration from every test.

    HOME was already redirected, but ``MempalaceConfig`` reads
    ``MEMPALACE_PALACE_PATH`` from the environment and that value wins over the
    ``config`` fixture's config.json. With it exported, twelve tests — including
    ``test_clean_lone_surrogates.py::TestToolsAcceptSurrogates::test_add_drawer_content``
    and ``test_cli_api.py::test_full_crud_round_trip`` — stop using their scratch
    palace and read, write and DELETE against whatever palace the variable names.
    Pointed at ``~/.mempalace/palace`` that is the user's real 140k-drawer memory,
    and the suite still reports one failure out of 3,575.

    ``sandbox.env`` exports exactly that variable, so the hazard is one
    ``source`` away. The rest of the list is the same class of bug with a smaller
    blast radius: production reads them, so an exported value reconfigures the
    system under test without changing a line of code.

    Tests that exercise any of these set them explicitly with ``monkeypatch``,
    which runs after this fixture and therefore still wins.
    """
    for var in _AMBIENT_VARS_TO_SCRUB:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(autouse=True)
def _stable_embedding_function_for_tests(request, monkeypatch):
    """Keep ordinary tests off ChromaDB's native ONNX embedding path.

    Module-sized Windows runs were crashing inside onnxruntime after many raw
    Chroma add/query calls. The embedding-specific tests opt out below; every
    other test gets a deterministic in-process EF so it still exercises vector
    writes/search without loading native ONNX sessions.
    """
    module_name = getattr(getattr(request, "module", None), "__name__", "")
    if module_name in _REAL_EMBEDDING_TEST_MODULES:
        yield
        return

    ef = _StableTestEmbeddingFunction()

    import mempalace.backends.chroma as chroma_mod
    import mempalace.backends.embedding_wrapper as embedding_wrapper
    import mempalace.embedding as embedding_mod
    from chromadb.api.types import DefaultEmbeddingFunction

    monkeypatch.setattr(DefaultEmbeddingFunction, "__call__", lambda self, input: ef(input=input))
    monkeypatch.setattr(
        DefaultEmbeddingFunction, "embed_query", lambda self, input: ef(input=input)
    )
    monkeypatch.setattr(embedding_mod, "get_embedding_function", lambda *_, **__: ef)
    monkeypatch.setattr(
        chroma_mod.ChromaBackend, "_resolve_embedding_function", staticmethod(lambda: ef)
    )
    # Must accept ``is_query`` — our _embed_texts routes asymmetric models
    # (bge-small, nomic) through ef.embed_query, and every explicit-vector
    # backend read passes is_query=True. A bare ``lambda texts:`` stub raises
    # TypeError on those call sites. The stub EF is symmetric, so ignoring the
    # flag is behavior-neutral for tests.
    monkeypatch.setattr(
        embedding_wrapper,
        "_embed_texts",
        lambda texts, is_query=False: ef(input=list(texts)),
    )
    yield


@pytest.fixture(autouse=True)
def _reset_mcp_cache():
    """Reset cached MCP state between tests without importing mcp_server.

    If mempalace.mcp_server is already imported, close/clear its KG cache and
    Chroma client cache. If it has not been imported, leave it unloaded so
    fork/spawn-based tests do not inherit extra Chroma/SQLite state.
    """

    def _clear_cache():
        try:
            import sys

            mcp_server = sys.modules.get("mempalace.mcp_server")
            if mcp_server is not None:
                for kg in list(getattr(mcp_server, "_kg_by_path", {}).values()):
                    close = getattr(kg, "close", None)
                    if close is not None:
                        try:
                            close()
                        except Exception:
                            pass

                if hasattr(mcp_server, "_kg_by_path"):
                    mcp_server._kg_by_path.clear()

                # Close (not just dereference) the cached chromadb client so its
                # rust-side file handles are released; on Windows a bare deref
                # leaves them locked and leaks across the session (#1128).
                cached_client = getattr(mcp_server, "_client_cache", None)
                if cached_client is not None:
                    close = getattr(cached_client, "close", None)
                    if callable(close):
                        try:
                            close()
                        except Exception:
                            pass
                mcp_server._client_cache = None
                mcp_server._collection_cache = None
                if hasattr(mcp_server, "_collection_cache_backend"):
                    mcp_server._collection_cache_backend = None
                if hasattr(mcp_server, "_collection_cache_palace"):
                    mcp_server._collection_cache_palace = None
                if hasattr(mcp_server, "_collection_open_error"):
                    mcp_server._collection_open_error = None
        except AttributeError:
            pass

        try:
            # Reset the per-process quarantine gate so tests don't leak
            # state through ChromaBackend._quarantined_paths, and drop cached
            # HNSW capacity verdicts (#1471) for the same reason — a test that
            # reuses a palace path would otherwise inherit the previous test's
            # verdict.
            from mempalace.backends.chroma import ChromaBackend, reset_hnsw_capacity_cache

            ChromaBackend._quarantined_paths.clear()
            reset_hnsw_capacity_cache()
        except (ImportError, AttributeError):
            pass

        # Release chromadb clients opened through the backend layer. Many tests
        # reach the store via palace.get_collection() (sweep, repair, CLI, ...),
        # which caches one PersistentClient per palace_path on the long-lived
        # backend singleton and never closes it. chromadb frees the rust-side
        # SQLite/HNSW file handles only on client.close(); on POSIX the open
        # handles are harmless, but on Windows they stay locked and accumulate
        # across the session until a later test's HNSW segment write fails
        # (#1128 Windows CI). close_palace() closes the client and drops the
        # handle without marking the backend closed, so it stays reusable.
        try:
            from mempalace import palace as _palace

            backend = getattr(_palace, "_DEFAULT_BACKEND", None)
            clients = getattr(backend, "_clients", None)
            if clients:
                for path in list(clients):
                    try:
                        backend.close_palace(path)
                    except Exception:
                        pass
        except (ImportError, AttributeError):
            pass

    _clear_cache()
    yield
    _clear_cache()


@pytest.fixture(scope="session", autouse=True)
def _isolate_home():
    """Ensure HOME points to a temp dir for the entire test session.

    The env vars were already set at module level (above) so that
    module-level initialisations are captured.  This fixture simply
    restores the originals on teardown and cleans up the temp dir.
    """
    yield
    for var, orig in _original_env.items():
        if orig is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = orig
    shutil.rmtree(_session_tmp, ignore_errors=True)


@pytest.fixture
def tmp_dir():
    """Create and auto-cleanup a temporary directory."""
    d = tempfile.mkdtemp(prefix="mempalace_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def palace_path(tmp_dir):
    """Path to an empty palace directory inside tmp_dir."""
    p = os.path.join(tmp_dir, "palace")
    os.makedirs(p)
    return p


@pytest.fixture
def config(tmp_dir, palace_path):
    """A MempalaceConfig pointing at the temp palace."""
    cfg_dir = os.path.join(tmp_dir, "config")
    os.makedirs(cfg_dir)
    import json

    with open(os.path.join(cfg_dir, "config.json"), "w") as f:
        json.dump({"palace_path": palace_path}, f)
    return MempalaceConfig(config_dir=cfg_dir)


@pytest.fixture
def collection(palace_path):
    """A ChromaDB collection pre-seeded in the temp palace."""
    client = chromadb.PersistentClient(path=palace_path)
    col = client.get_or_create_collection("mempalace_drawers", metadata={"hnsw:space": "cosine"})
    yield col
    client.delete_collection("mempalace_drawers")
    # close() (not a bare dereference) releases chromadb's rust-side SQLite/HNSW
    # file handles. On Windows a mere `del` leaves them locked, so the temp
    # palace cannot be removed and handles leak across the whole test session
    # until a later test's HNSW write fails (#1128 Windows CI).
    client.close()


@pytest.fixture
def seeded_collection(collection):
    """Collection with a handful of representative drawers."""
    collection.add(
        ids=[
            "drawer_proj_backend_aaa",
            "drawer_proj_backend_bbb",
            "drawer_proj_frontend_ccc",
            "drawer_notes_planning_ddd",
        ],
        documents=[
            "The authentication module uses JWT tokens for session management. "
            "Tokens expire after 24 hours. Refresh tokens are stored in HttpOnly cookies.",
            "Database migrations are handled by Alembic. We use PostgreSQL 15 "
            "with connection pooling via pgbouncer.",
            "The React frontend uses TanStack Query for server state management. "
            "All API calls go through a centralized fetch wrapper.",
            "Sprint planning: migrate auth to passkeys by Q3. "
            "Evaluate ChromaDB alternatives for vector search.",
        ],
        metadatas=[
            {
                "wing": "project",
                "room": "backend",
                "source_file": "auth.py",
                "chunk_index": 0,
                "added_by": "miner",
                "filed_at": "2026-01-01T00:00:00",
            },
            {
                "wing": "project",
                "room": "backend",
                "source_file": "db.py",
                "chunk_index": 0,
                "added_by": "miner",
                "filed_at": "2026-01-02T00:00:00",
            },
            {
                "wing": "project",
                "room": "frontend",
                "source_file": "App.tsx",
                "chunk_index": 0,
                "added_by": "miner",
                "filed_at": "2026-01-03T00:00:00",
            },
            {
                "wing": "notes",
                "room": "planning",
                "source_file": "sprint.md",
                "chunk_index": 0,
                "added_by": "miner",
                "filed_at": "2026-01-04T00:00:00",
            },
        ],
    )
    return collection


@pytest.fixture
def kg(tmp_dir):
    """An isolated KnowledgeGraph using a temp SQLite file."""
    db_path = os.path.join(tmp_dir, "test_kg.sqlite3")
    graph = KnowledgeGraph(db_path=db_path)
    yield graph
    graph.close()


@pytest.fixture
def seeded_kg(kg):
    """KnowledgeGraph pre-loaded with sample triples."""
    kg.add_entity("Alice", entity_type="person")
    kg.add_entity("Max", entity_type="person")
    kg.add_entity("swimming", entity_type="activity")
    kg.add_entity("chess", entity_type="activity")

    kg.add_triple("Alice", "parent_of", "Max", valid_from="2015-04-01")
    kg.add_triple("Max", "does", "swimming", valid_from="2025-01-01")
    kg.add_triple("Max", "does", "chess", valid_from="2024-06-01")
    kg.add_triple("Alice", "works_at", "Acme Corp", valid_from="2020-01-01", valid_to="2024-12-31")
    kg.add_triple("Alice", "works_at", "NewCo", valid_from="2025-01-01")

    return kg
