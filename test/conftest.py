import logging
import sys
from pathlib import Path

import pytest

from data_store import DataStore
from utils import logger

@pytest.fixture(scope="session", autouse=True)
def set_log_level():
    logger.setLevel(logging.ERROR)

@pytest.fixture(scope="session", autouse=True)
def no_data_store_use():
    data_store_use = DataStore.USE_STORE
    DataStore.USE_STORE = False
    yield
    DataStore.USE_STORE = data_store_use


@pytest.fixture(scope="session", autouse=True)
def import_build_plots():
    # import from scripts to get correct coverage
    sys.path.append(Path(__file__).parent.parent.joinpath("scripts").as_posix())
    import build_plots  # noqa: E402