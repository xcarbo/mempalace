import json

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


def test_emit_skips_untracked_wing_and_missing_config(tmp_path, monkeypatch):
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
