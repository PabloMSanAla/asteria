"""Tests for Asteria CLI entry points."""

import pytest
import torch
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
    assert "asteria 0.1.0" in captured.out or "asteria 0.1.0" in captured.err


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
