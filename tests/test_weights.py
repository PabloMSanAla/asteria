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


def test_download_weight_file_success(tmp_path, monkeypatch):
    """Verify download_weight_file saves and validates checkpoint."""
    import asteria.weights.manager as wm
    from asteria.weights.manager import download_weight_file, compute_sha256

    dest = tmp_path / "weights" / "unetpp_mock.pt"

    def mock_download(url, temp_dest):
        temp_dest.parent.mkdir(parents=True, exist_ok=True)
        temp_dest.write_text("mock unetpp weights content")

    monkeypatch.setattr(wm, "_download_remote_file", mock_download)

    # Compute hash of the mock content
    temp_check = tmp_path / "temp_check.pt"
    temp_check.write_text("mock unetpp weights content")
    expected_hash = compute_sha256(temp_check)

    from asteria.weights.registry import WeightMetadata

    mock_meta = WeightMetadata(
        version="v0.1.0",
        filename="unetpp_mock.pt",
        url="https://drive.google.com/mock",
        sha256=expected_hash,
        in_channels=3,
        num_classes=3,
        features=(32, 64),
        deep_supervision=False,
        epochs=80,
        training_hardware="GH200",
        dataset_description="mock",
        interior_iou=0.75,
        boundary_iou_tol2=0.51,
        soft_boundary_iou=0.40,
        macro_iou=0.63,
        release_date="2026-11-01",
    )

    res = download_weight_file(mock_meta, dest)
    assert res.exists()
    assert res == dest


def test_resolve_checkpoint_path_triggers_download(tmp_path, monkeypatch):
    """Verify resolve_checkpoint_path triggers official weight download when cache is empty."""
    import asteria.weights.manager as wm

    download_records = []

    def mock_download(url, dest):
        download_records.append((url, dest))
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("downloaded unetpp weights")

    monkeypatch.setattr(wm, "_download_remote_file", mock_download)
    # Patch compute_sha256 to match registry sha256
    monkeypatch.setattr(wm, "compute_sha256", lambda path: "71feeee2bc372693c361c234d2c114a1cc2d0805e03bc9e2c1c95004b3d21df4")

    cached_file = resolve_checkpoint_path("v0.1.0", cache_dir=tmp_path)
    assert cached_file.exists()
    assert len(download_records) == 1
    assert "drive.google.com" in download_records[0][0]

    # Subsequent resolution uses cache
    cached_again = resolve_checkpoint_path("v0.1.0", cache_dir=tmp_path)
    assert cached_again == cached_file
    assert len(download_records) == 1


def test_starnet3_download_disabled():
    """Verify StarNet3 automatic remote download is disabled and guides user."""
    from asteria.weights.starnet3_manager import resolve_starnet3_checkpoint_path, download_starnet3_model

    with pytest.raises(RuntimeError, match="not currently published for automatic remote download"):
        resolve_starnet3_checkpoint_path("latest")

    with pytest.raises(RuntimeError, match="not currently published for automatic remote download"):
        download_starnet3_model("latest")
