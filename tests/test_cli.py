"""Tests for Asteria CLI entry points."""

import pytest
import torch
from asteria import __version__
from asteria.cli import main
from asteria.models.unetpp import UNetPlusPlus
from asteria.utils.synthetic import save_synthetic_fits


@pytest.fixture
def cli_test_setup(tmp_path):
    """Set up small synthetic FITS and mock checkpoint for CLI testing."""
    fits_path = tmp_path / "simulated_scene.fits"
    save_synthetic_fits(fits_path, size=128, seed=42)

    ckpt_path = tmp_path / "cli_mock_unetpp.pt"
    model = UNetPlusPlus(
        in_channels=3,
        num_classes=3,
        features=(16, 32, 64, 128, 256),
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "in_channels": 3,
            "num_classes": 3,
            "features": (16, 32, 64, 128, 256),
            "deep_supervision": False,
        },
        ckpt_path,
    )
    return fits_path, ckpt_path


def test_cli_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert f"asteria {__version__}" in captured.out or f"asteria {__version__}" in captured.err


def test_cli_download_flag(monkeypatch, tmp_path, capsys):
    """Verify asteria --download pre-downloads and caches UNet++ weights."""
    import asteria.weights.manager as wm

    mock_ckpt = tmp_path / "mock_unetpp.pt"
    mock_ckpt.write_text("mock unetpp")
    monkeypatch.setattr(wm, "resolve_checkpoint_path", lambda version: mock_ckpt)

    ret = main(["--download"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "Pre-downloading official UNet++ weights" in captured.out
    assert "Successfully cached UNet++ weights at" in captured.out


def test_cli_help(capsys):
    ret = main([])
    assert ret == 0
    captured = capsys.readouterr()
    assert "Asteria" in captured.out


def test_cli_full_run(cli_test_setup, tmp_path):
    fits_path, ckpt_path = cli_test_setup
    out_png = tmp_path / "cli_output.png"
    out_fits = tmp_path / "cli_segmap.fits"

    argv = [
        str(fits_path),
        "--checkpoint", str(ckpt_path),
        "--device", "cpu",
        "--tile_size", "128",
        "--overlap", "32",
        "--batch_size", "2",
        "--output", str(out_png),
        "--export_fits", str(out_fits),
        "--no_display",
    ]

    ret = main(argv)
    assert ret == 0
    assert out_png.exists()
    assert out_png.stat().st_size > 0
    assert out_fits.exists()
    assert out_fits.stat().st_size > 0


def test_cli_default_batch_size_adaptive(cli_test_setup, tmp_path):
    """Verify CLI succeeds without providing --batch_size (batch_size=None adaptive)."""
    fits_path, ckpt_path = cli_test_setup
    out_png = tmp_path / "cli_adaptive_output.png"

    argv = [
        str(fits_path),
        "--checkpoint", str(ckpt_path),
        "--device", "cpu",
        "--tile_size", "128",
        "--overlap", "32",
        "--output", str(out_png),
        "--no_display",
    ]

    ret = main(argv)
    assert ret == 0
    assert out_png.exists()
    assert out_png.stat().st_size > 0


def test_cli_sky_flag(cli_test_setup, tmp_path):
    """Verify CLI accepts and processes explicit --sky argument."""
    fits_path, ckpt_path = cli_test_setup
    out_png = tmp_path / "cli_sky_output.png"

    argv = [
        str(fits_path),
        "--checkpoint", str(ckpt_path),
        "--device", "cpu",
        "--tile_size", "128",
        "--overlap", "32",
        "--sky", "5.0",
        "--output", str(out_png),
        "--no_display",
    ]

    ret = main(argv)
    assert ret == 0
    assert out_png.exists()
    assert out_png.stat().st_size > 0


def test_cli_pixel_clarity_no_display(cli_test_setup, monkeypatch):
    """Verify --no_clarity suppresses pixel clarity launcher."""
    import asteria.cli as cli

    clarity_called = []
    monkeypatch.setattr(cli, "open_pixel_clarity", lambda **kwargs: clarity_called.append(kwargs))

    fits_path, ckpt_path = cli_test_setup
    argv = [
        str(fits_path),
        "--checkpoint", str(ckpt_path),
        "--device", "cpu",
        "--tile_size", "128",
        "--overlap", "32",
        "--no_clarity",
    ]

    ret = main(argv)
    assert ret == 0
    assert len(clarity_called) == 0
