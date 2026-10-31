"""Tests for weights registry and path resolution."""

import pytest
from asteria.weights.registry import get_weight_metadata, WEIGHT_REGISTRY, LATEST_VERSION
from asteria.weights.manager import resolve_checkpoint_path


def test_registry_metadata():
    meta = get_weight_metadata("v0.1.0")
    assert meta.version == "v0.1.0"
    assert meta.in_channels == 3
    assert meta.num_classes == 3
    assert meta.epochs == 20
    assert meta.interior_iou > 0.8


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
