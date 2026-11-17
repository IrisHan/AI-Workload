import json
from types import SimpleNamespace

import pytest
from app.lib.spec_studio import fetch


def test_fetch_resolves_every_file_to_immutable_sha(monkeypatch, tmp_path, qwen_cfg):
    sha = "a" * 40
    config = tmp_path / "config.json"
    config.write_text(json.dumps(qwen_cfg))
    calls = []

    def download(repo_id, filename, revision, **kwargs):
        calls.append((filename, revision))
        if filename == "config.json":
            return str(config)
        raise FileNotFoundError(filename)

    monkeypatch.setattr("huggingface_hub.model_info", lambda *a, **k: SimpleNamespace(sha=sha))
    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    monkeypatch.setattr(fetch, "_CACHE", {})
    bundle = fetch.fetch_bundle("test/model", revision="main")
    assert bundle.sha == sha
    assert calls
    assert all(revision == sha for _, revision in calls)


def test_fetch_refreshes_main_and_isolates_cached_objects(monkeypatch, tmp_path, qwen_cfg):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(qwen_cfg))
    state = {"sha": "a" * 40, "calls": 0}

    def info(*args, **kwargs):
        state["calls"] += 1
        return SimpleNamespace(sha=state["sha"])

    def download(repo_id, filename, **kwargs):
        if filename != "config.json":
            raise FileNotFoundError(filename)
        return str(path)

    monkeypatch.setattr("huggingface_hub.model_info", info)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    monkeypatch.setattr(fetch, "_CACHE", {})
    first = fetch.fetch_bundle("test/model")
    first.config["hidden_size"] = 999
    second = fetch.fetch_bundle("test/model")
    assert second.config["hidden_size"] == 1536
    state["sha"] = "b" * 40
    assert fetch.fetch_bundle("test/model").sha == "b" * 40
    assert state["calls"] == 3
    assert len(fetch._CACHE) == 2


def test_fetch_authorization_checked_before_cached_read(monkeypatch, tmp_path, qwen_cfg):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(qwen_cfg))

    def info(*args, **kwargs):
        if kwargs["token"] != "authorized-test-token":
            raise PermissionError("gated")
        return SimpleNamespace(sha="c" * 40)

    def download(repo_id, filename, **kwargs):
        if filename != "config.json":
            raise FileNotFoundError(filename)
        return str(path)

    monkeypatch.setattr("huggingface_hub.model_info", info)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    monkeypatch.setattr(fetch, "_CACHE", {})
    fetch.fetch_bundle("test/private", token="authorized-test-token")
    with pytest.raises(PermissionError):
        fetch.fetch_bundle("test/private")
