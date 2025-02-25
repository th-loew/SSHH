import os
from pathlib import Path
from typing import Any

import numpy as np

from utils import logger, FloatArray, ComplexArray

DATA_STORE_DIR = os.getenv("DATA_STORE_DIR", None)


StoreType = FloatArray | ComplexArray | np.ndarray  # check: ignore


class DataStoreMeta(type):

    DEFAULT_DATA_STORE_PATH = Path(DATA_STORE_DIR) if DATA_STORE_DIR else Path(__file__).parent.parent.joinpath("data")

    def __new__(mcs, name, bases, attrs):
        if name not in ("DataStore", "FloatDataStore"):
            root = attrs.get("ROOT")
            for b in bases:
                if isinstance(b, DataStoreMeta):
                    try:
                        root = root or b.ROOT
                    except AttributeError:
                        pass
            root = root or mcs.DEFAULT_DATA_STORE_PATH.joinpath(name)
            attrs.setdefault("ROOT", root)
            attrs.setdefault("SUBDIRS", tuple())
            root.mkdir(parents=True, exist_ok=True)
            for subdir in attrs["SUBDIRS"]:
                root.joinpath(subdir).mkdir(parents=True, exist_ok=True)
        return super().__new__(mcs, name, bases, attrs)


class DataStore(metaclass=DataStoreMeta):
    ROOT: Path
    SUBDIRS: tuple[str, ...]
    _ERRORS = FileNotFoundError, EOFError, ValueError, OSError  # OSError because of filename too long
    _MAX_NBYTES = 5e6
    USE_STORE = True

    def __init__(self, do_not_compute: bool = False):
        self._do_not_compute = do_not_compute

    class LoadException(Exception):
        pass

    class LoadExceptionDoNotCompute(Exception):
        pass

    def load_data(self, naming: Any, subdir: str | None = None) -> StoreType:
        if not self.USE_STORE:
            raise self.LoadException("Data store disabled")
        path = self._get_path(naming, subdir)
        try:
            return np.load(path)
        except self._ERRORS as e:
            if self._do_not_compute:
                raise self.LoadExceptionDoNotCompute() from e
            logger.warning("Failed to load data from %s", path)
            raise self.LoadException() from e

    def save_data(self, data: StoreType, naming: Any, subdir: str | None = None) -> None:
        if not self.USE_STORE:
            return
        if data.nbytes > self._MAX_NBYTES:
            # logger.warning(f"Data size {data.nbytes} exceeds maximum {self.MAX_NBYTES}")
            return
        path = self._get_path(naming, subdir)
        try:
            np.save(path, data)
        except self._ERRORS:
            logger.warning("Failed to save data to %s (filename too long?)", path)

    def _get_path(self, naming: Any, subdir: str | None = None) -> str:
        assert subdir is None or subdir in self.SUBDIRS, f"Unknown subdir {subdir}"
        filename = self._get_filename(naming)
        path = self.ROOT
        if subdir:
            path = path.joinpath(subdir)
        if subsubdir := self._get_subsubdir():
            path = path.joinpath(subsubdir)
            path.mkdir(exist_ok=True)
        path = path.joinpath(filename)
        return str(path)


    def _get_subsubdir(self) -> str | None:
        return None


    def _get_filename(self, thing) -> str:
        def get_filename_blank(thing) -> str:
            if isinstance(thing, str):
                return thing
            from chain_model import ChainModel
            if isinstance(thing, ChainModel.Params):
                return str(thing).replace(" b=0.0 ", " b=0 ").replace(" trap=None ", " ")
            from params_range import ParamsRange
            if isinstance(thing, ParamsRange):
                return f"{thing}"
            raise ValueError(f"Don't know how to name {thing}")
        filename = get_filename_blank(thing)
        if not filename.endswith(".npy"):
            filename += ".npy"
        return filename


class FloatDataStore(DataStore):

    def load_data(self, naming: Any, subdir: str | None = None) -> FloatArray:
        return super().load_data(naming, subdir)  # type: ignore[return-value]
