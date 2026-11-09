"""Tests for weights registry and path resolution."""

import pytest
from asteria.weights.registry import get_weight_metadata, WEIGHT_REGISTRY, LATEST_VERSION
from asteria.weights.manager import resolve_checkpoint_path


def test_registry_metadata():
    meta = get_weight_metadata("v0.1.0")
    assert meta.version == "v0.1.0"
    assert "drive.google.com" in meta.url
    assert "1HfPMtcR0jjEfHIlAC-gEWAQeIFoljBwC" in meta.url
    assert meta.in_channels == 3
    assert meta.num_classes == 3
    assert meta.epochs == 80
    assert meta.interior_iou > 0.7


def test_registry_latest_alias():
    meta_latest = get_weight_metadata("latest")
    assert meta_latest.version == LATEST_VERSION


def test_registry_unknown_version():
    with pytest.raises(KeyError, match="Unknown weight version"):
        get_weight_metadata("v999.0.0")


def test_resolve_local_file(tmp_path):
    dummy_file = tmp_path / "custom_weights.pt"
    dummy_file.write_text("mock binary weights")
    resolved = resolve_checkpoint_path(dummy_file)
    assert resolved == dummy_file.resolve()


def test_resolve_env_var(tmp_path, monkeypatch):
    env_file = tmp_path / "env_weights.pt"
    env_file.write_text("mock env weights")
    monkeypatch.setenv("ASTERIA_WEIGHTS_PATH", str(env_file))

    resolved = resolve_checkpoint_path("v0.1.0")
    assert resolved == env_file.resolve()


def test_resolve_url_custom(tmp_path, monkeypatch):
    """Verify resolve_checkpoint_path handles direct URLs and caches them."""
    import asteria.weights.manager as wm

    download_called = []

    def mock_download(url, dest):
        download_called.append((url, dest))
        dest.write_text("mock downloaded model")

    monkeypatch.setattr(wm, "_download_remote_file", mock_download)

    test_url = "https://drive.google.com/file/d/1HfPMtcR0jjEfHIlAC-gEWAQeIFoljBwC/view?usp=sharing"
    cached = resolve_checkpoint_path(test_url, cache_dir=tmp_path)
    assert cached.exists()
    assert len(download_called) == 1
    assert download_called[0][0] == test_url

    # Second call should use cache without downloading again
    cached2 = resolve_checkpoint_path(test_url, cache_dir=tmp_path)
    assert cached2 == cached
    assert len(download_called) == 1
