"""Makes the project root importable for pytest and gives tests a synthetic dataset."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))


@pytest.fixture(scope="session")
def synthetic_dir(tmp_path_factory):
    import synthetic

    folder = tmp_path_factory.mktemp("gl_data")
    synthetic.write_cache(folder)
    return folder


@pytest.fixture()
def df(synthetic_dir, monkeypatch):
    import data

    monkeypatch.setenv("GLOBALLENS_DATA_DIR", str(synthetic_dir))
    return data.load_dataset()
