"""Tests for Asteria CLI entry points."""

import pytest
import torch
from asteria.cli import main, starnet3_main
from asteria.models.unetpp import UNetPlusPlus
from asteria.models.starnet3 import StarNet3Model
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
        "--deblend",
        "--no_display",
    ]

    ret = main(argv)
    assert ret == 0
    assert out_png.exists()
    assert out_png.stat().st_size > 0
    assert out_fits.exists()
    assert out_fits.stat().st_size > 0


@pytest.fixture
def starnet_cli_test_setup(tmp_path):
    fits_path = tmp_path / "starnet_scene.fits"
    save_synthetic_fits(fits_path, size=128, seed=42)

    ckpt_file = tmp_path / "mock_starnet3.pt"
    model = StarNet3Model(
        dim=12,
        dim_mults=(1, 2, 4),
        unet_scales=(2, 4),
        refiner_n_feat=16,
        refiner_num_orb=1,
        refiner_orb_num_cab=1,
        refiner_branch_width=4,
        adapter_hidden_dim=8,
        sam_hidden_dim=8,
        stage_input_channels=3,
        use_pos_guide=False,
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "dim": 12,
            "dim_mults": (1, 2, 4),
            "unet_scales": (2, 4),
            "refiner_n_feat": 16,
            "refiner_num_orb": 1,
            "refiner_orb_num_cab": 1,
            "refiner_branch_width": 4,
            "adapter_hidden_dim": 8,
            "sam_hidden_dim": 8,
            "stage_input_channels": 3,
            "use_pos_guide": False,
        },
        ckpt_file,
    )
    return fits_path, ckpt_file


def test_cli_starnet3_help(capsys):
    ret = starnet3_main([])
    assert ret == 1
    captured = capsys.readouterr()
    assert "StarNet3" in captured.out


def test_cli_starnet3_execution(starnet_cli_test_setup, tmp_path):
    fits_path, ckpt_path = starnet_cli_test_setup
    out_fits = tmp_path / "starless.fits"
    mask_fits = tmp_path / "starmask.fits"

    argv = [
        str(fits_path),
        "--checkpoint", str(ckpt_path),
        "--device", "cpu",
        "--tile_size", "64",
        "--overlap", "16",
        "-o", str(out_fits),
        "--mask", str(mask_fits),
        "--no_display",
    ]

    ret = starnet3_main(argv)
    assert ret == 0
    assert out_fits.exists()
    assert mask_fits.exists()


def test_cli_asteria_starnet_subcommand(starnet_cli_test_setup, tmp_path):
    fits_path, ckpt_path = starnet_cli_test_setup
    out_fits = tmp_path / "subcommand_starless.fits"

    argv = [
        "starnet",
        str(fits_path),
        "--checkpoint", str(ckpt_path),
        "--device", "cpu",
        "--tile_size", "64",
        "--overlap", "16",
        "-o", str(out_fits),
        "--no_display",
    ]

    ret = main(argv)
    assert ret == 0
    assert out_fits.exists()


def test_cli_starnet3_export_mef_fits(starnet_cli_test_setup, tmp_path):
    fits_path, ckpt_path = starnet_cli_test_setup
    export_fits = tmp_path / "starnet_mef.fits"

    argv = [
        str(fits_path),
        "--checkpoint", str(ckpt_path),
        "--device", "cpu",
        "--tile_size", "64",
        "--overlap", "16",
        "--export_fits", str(export_fits),
        "--no_display",
    ]

    ret = starnet3_main(argv)
    assert ret == 0
    assert export_fits.exists()

    from astropy.io import fits
    with fits.open(export_fits) as hdul:
        assert len(hdul) == 2
        assert hdul[0].header.get("EXTNAME") == "ORIGINAL"
        assert hdul[0].data.shape == (128, 128)
        assert hdul[1].name == "STARLESS"
        assert hdul[1].data.shape == (128, 128)


def test_cli_starnet3_with_sky_and_pixel_scale(starnet_cli_test_setup, tmp_path):
    """Verify starnet3 CLI resolves stretch kwargs with explicit --sky and --pixel_scale."""
    fits_path, ckpt_path = starnet_cli_test_setup
    export_fits = tmp_path / "starnet_sky_mef.fits"

    argv = [
        str(fits_path),
        "--checkpoint", str(ckpt_path),
        "--device", "cpu",
        "--tile_size", "64",
        "--overlap", "16",
        "--pixel_scale", "1.0",
        "--sky", "0.5",
        "--export_fits", str(export_fits),
        "--no_display",
    ]

    ret = starnet3_main(argv)
    assert ret == 0
    assert export_fits.exists()
