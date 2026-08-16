import json

import pytest

import mempalace.outbox as outbox
from mempalace.config import MempalaceConfig


def _cfg(tmp_path, block):
    (tmp_path / "config.json").write_text(json.dumps({"outbox": block}))
    return MempalaceConfig(config_dir=str(tmp_path))


def test_outbox_config_property(tmp_path):
    cfg = _cfg(
        tmp_path,
        {
            "url": "http://localhost:3131/api/palace-event",
            "secret": "s",
            "wings": ["torq-terminal"],
        },
    )
    assert cfg.outbox["url"].endswith("/api/palace-event")
    assert cfg.outbox["wings"] == ["torq-terminal"]


def test_outbox_env_overrides(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, {})
    monkeypatch.setenv("MEMPALACE_OUTBOX_URL", "http://x/e")
    monkeypatch.setenv("MEMPALACE_OUTBOX_WINGS", "a,b")
    assert cfg.outbox["url"] == "http://x/e"
    assert cfg.outbox["wings"] == ["a", "b"]


def test_emit_posts_for_tracked_wing(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, {"url": "http://t/e", "secret": "sek", "wings": ["torq-terminal"]})
    sent = {}

    def fake_urlopen(req, timeout=None):
        sent["url"] = req.full_url
        sent["secret"] = req.get_header("X-palace-secret")
        sent["body"] = json.loads(req.data)

        class R:  # noqa: N801
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return b"{}"

        return R()

    monkeypatch.setattr(outbox.urllib.request, "urlopen", fake_urlopen)
    outbox.emit(cfg, "torq-terminal", "d1", "added")
    assert sent["body"] == {"wing": "torq-terminal", "drawer_id": "d1", "event": "added"}
    assert sent["secret"] == "sek"


def test_emit_skips_untracked_wing_and_missing_cfg(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, {"url": "http://t/e", "secret": "s", "wings": ["only-this"]})

    def boom(*a, **k):
        raise AssertionError("should not POST")

    monkeypatch.setattr(outbox.urllib.request, "urlopen", boom)
    outbox.emit(cfg, "other-wing", "d1", "added")  # no POST
    cfg2 = _cfg(tmp_path, {})
    outbox.emit(cfg2, "torq-terminal", "d1", "added")  # disabled → no POST


def test_emit_swallows_network_errors(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, {"url": "http://t/e", "secret": "s", "wings": ["w"]})

    def boom(*a, **k):
        raise OSError("connection refused")

    monkeypatch.setattr(outbox.urllib.request, "urlopen", boom)
    outbox.emit(cfg, "w", "d1", "added")  # must not raise


def test_outbox_secret_file_supplies_secret(tmp_path):
    """secret_file is what makes the secret reachable from cron and launchd.

    An env var only reaches processes whose shell exported it; ``memp`` also
    runs from cron and launchd, which read neither .zshenv nor .bashrc. Reading
    a path in-process works in every context, and lets the secret live on a
    filesystem where chmod is actually enforced.
    """
    secret_path = tmp_path / "outbox.secret"
    secret_path.write_text("s3cr3t-from-file\n")  # trailing newline must be stripped
    cfg = _cfg(tmp_path, {"url": "http://x/api", "wings": ["w"], "secret_file": str(secret_path)})

    ob = cfg.outbox

    assert ob["secret"] == "s3cr3t-from-file"
    assert "secret_file" not in ob, "the path must not leak into the emitted block"


def test_outbox_env_secret_beats_secret_file(tmp_path, monkeypatch):
    secret_path = tmp_path / "outbox.secret"
    secret_path.write_text("from-file")
    monkeypatch.setenv("MEMPALACE_OUTBOX_SECRET", "from-env")
    cfg = _cfg(tmp_path, {"url": "http://x/api", "wings": ["w"], "secret_file": str(secret_path)})

    assert cfg.outbox["secret"] == "from-env"


def test_outbox_secret_file_beats_inline_secret(tmp_path):
    """The whole point is to stop relying on the inline value, so the file wins."""
    secret_path = tmp_path / "outbox.secret"
    secret_path.write_text("from-file")
    cfg = _cfg(
        tmp_path,
        {
            "url": "http://x/api",
            "wings": ["w"],
            "secret": "inline",
            "secret_file": str(secret_path),
        },
    )

    assert cfg.outbox["secret"] == "from-file"


def test_outbox_secret_file_env_var_overrides_config_key(tmp_path, monkeypatch):
    chosen = tmp_path / "chosen.secret"
    chosen.write_text("chosen")
    ignored = tmp_path / "ignored.secret"
    ignored.write_text("ignored")
    monkeypatch.setenv("MEMPALACE_OUTBOX_SECRET_FILE", str(chosen))
    cfg = _cfg(tmp_path, {"url": "http://x/api", "wings": ["w"], "secret_file": str(ignored)})

    assert cfg.outbox["secret"] == "chosen"


@pytest.mark.parametrize(
    "state",
    ["missing", "empty", "whitespace"],
    ids=["missing-file", "empty-file", "whitespace-only"],
)
def test_outbox_unusable_secret_file_falls_back_and_never_raises(tmp_path, state):
    """Fail-soft by contract: outbox.emit swallows everything so a palace write
    is never broken by the webhook. Resolution must not raise either."""
    secret_path = tmp_path / "outbox.secret"
    if state == "empty":
        secret_path.write_text("")
    elif state == "whitespace":
        secret_path.write_text("   \n\t ")

    cfg = _cfg(
        tmp_path,
        {
            "url": "http://x/api",
            "wings": ["w"],
            "secret": "inline",
            "secret_file": str(secret_path),
        },
    )

    assert cfg.outbox["secret"] == "inline"


def test_outbox_secret_file_expands_user(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    (home / "outbox.secret").write_text("tilde-resolved")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("MEMPALACE_OUTBOX_SECRET_FILE", "~/outbox.secret")
    cfg = _cfg(tmp_path, {"url": "http://x/api", "wings": ["w"]})

    assert cfg.outbox["secret"] == "tilde-resolved"
