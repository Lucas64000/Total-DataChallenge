"""Shared fixtures for ETL unit tests."""

import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from pipeline.etl.config import PathConfig, PreprocessingConfig


@pytest.fixture
def sandbox(request: pytest.FixtureRequest) -> Path:
    """
    Isolated temporary directory scoped to one test.

    Each test gets its own folder (named after the module + test function) so
    tests that write real files to disk never interfere with each other. The
    directory is removed both before (in case a previous run crashed) and after
    the test to keep the workspace clean.
    """
    module = request.node.module.__name__.split(".")[-1]
    name = request.node.name
    root = Path(f"_sandbox_{module}_{name}")
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    yield root
    shutil.rmtree(root, ignore_errors=True)


@pytest.fixture
def preprocessing_config(sandbox: Path) -> PreprocessingConfig:
    """Build a PreprocessingConfig rooted inside the test sandbox."""
    return PreprocessingConfig(
        paths=PathConfig(
            source_dir=sandbox / "source",
            output_dir=sandbox / "output",
            backup_dir=sandbox / "output" / "backup",
        )
    )


@pytest.fixture
def etl_filenames() -> SimpleNamespace:
    """Canonical filename samples reused across ETL unit tests."""
    return SimpleNamespace(
        # transform/filename_parser samples
        boly_standard=(
            "FR_N0431710-871_W0000230-087_20231005_Ardea-cinerea_IMAG0023_SmallestMaxSize.jpg"
        ),
        reconyx_standard=(
            "FR_N0431652-111_W0000251-205_20220725_Canis-lupus-familiaris_RCNX0107_"
            "SmallestMaxSize.jpg"
        ),
        reconyx_ardea_sample=("FR_N123-456_W0000248-181_20240101_Ardea-cinerea_RCNX123_0001.jpg"),
        multi_species_reconyx=(
            "FR_N0431703-408_W0000243-7760_20230901_Ardea-cinerea-Martes-martes_"
            "RCNX0025_SmallestMaxSize.jpg"
        ),
        multi_species_boly=(
            "FR_N0431652-111_W0000251-205_20210429_Homo-sapiens-Canis-lupus-familiaris_"
            "IMAG0359_SmallestMaxSize.jpg"
        ),
        alt_unknown_camera=(
            "FR_N0431702-522_W0000243-861_20200619_Canis-lupus-familiaris_"
            "chien-2020-06-16-16-31-32_SmallestMaxSize.jpg"
        ),
        coord_suffix_unknown=(
            "FR_N0431656-068_W0000248-181b_20200603_Genetta-genetta_"
            "genette-2020-05-25-05-21-18_SmallestMaxSize.jpg"
        ),
        coord_suffix_reconyx=(
            "FR_N0431704-006_W0000242-723b_20200603_Martes-martes_RCNX0035_SmallestMaxSize.jpg"
        ),
        genette_alt=(
            "FR_N0431659-873_W0000246-345_20200619_Genetta-genetta_"
            "genette-2020-06-18-01-04-12_SmallestMaxSize.jpg"
        ),
        invalid="random_file_name.jpg",
        # timestamp_ocr samples
        reconyx_ocr_template=("FR_N0431652-111_W0000251-205_20220725_Fox_RCNX{index:04d}.jpg"),
        # detect_camera_type() direct inputs
        detect_reconyx_id="RCNX0107",
        detect_reconyx_filename="FR_N0431652-111_RCNX0107.jpg",
        detect_boly_id="IMAG0023",
        detect_boly_filename="FR_N0431710-871_IMAG0023.jpg",
        detect_unknown_filename="some_other_camera.jpg",
        detect_unknown_alt="chien-2020-06-16",
    )
