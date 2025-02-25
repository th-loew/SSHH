import numpy as np
import pytest

from chain_model import ChainModel
from data_store import DataStore as DataStore, FloatDataStore, DataStoreMeta
from harper import Harper
from params_range import ParamsRange

@pytest.fixture(scope="module")
def data():
    return np.array((1, 2.3, 4))

@pytest.fixture(scope="module")
def filename():
    return "test.npy"

@pytest.fixture(scope="module", autouse=True)
def UseDataStoreInTmpDir(tmp_path_factory):
    default_data_store_path = DataStoreMeta.DEFAULT_DATA_STORE_PATH
    use_store = DataStore.USE_STORE
    DataStoreMeta.DEFAULT_DATA_STORE_PATH = tmp_path_factory.mktemp("test_data")
    DataStore.USE_STORE = True
    yield
    DataStoreMeta.DEFAULT_DATA_STORE_PATH = default_data_store_path
    DataStore.USE_STORE = use_store

@pytest.fixture(scope="module")
def data_store(data, filename) -> DataStore:
    class DS1(DataStore):
        SUBDIRS = "save", "load"
    store = DS1()
    np.save(store.ROOT.joinpath("load").joinpath(filename), data)
    return store


@pytest.fixture(scope="module")
def data_store_with_subsubdir() -> DataStore:
    class DS2(DataStore):
        SUBDIRS = "save", "load"
        def _get_subsubdir(_self) -> str:
            return "subsubdir"
    return DS2()


def test_subdirs(data_store):
    for subdir in data_store.SUBDIRS:
        path = data_store.ROOT.joinpath(subdir)
        assert path.is_dir(), path

@pytest.mark.parametrize("subdir", [None, "save"])
def test_subsubdirs(data_store_with_subsubdir, data, filename, subdir):
    store = data_store_with_subsubdir
    kwargs = {} if subdir is None else {"subdir": subdir}
    store.save_data(data, filename, **kwargs)
    path = store.ROOT if subdir is None else store.ROOT.joinpath(subdir)
    path = path.joinpath("subsubdir").joinpath(filename)
    assert path.is_file()
    data_loaded = store.load_data(filename, **kwargs)
    assert np.all(data_loaded == data)

def test_load(data, filename, data_store):
    loaded_data = data_store.load_data(filename, subdir="load")
    assert np.all(loaded_data == data)

def test_load_fail(data_store):
    with pytest.raises(data_store.LoadException):
        data_store.load_data("non_existent_file", subdir="load")

def test_save(data_store, filename, data):
    data_store.save_data(data, filename, subdir="save")
    path = data_store.ROOT.joinpath("save").joinpath(filename)
    assert path.is_file()

def test_filename_params(data_store):
    params1 = ChainModel.Params(N=100, phi_hop=1)
    params2 = ChainModel.Params(N=100, t_inter=1, t_intra=1)
    filename1 = data_store._get_filename(params1)
    filename2 = data_store._get_filename(params2)
    assert filename1 != filename2

def test_filename_params_range(data_store):
    params_range1 = ParamsRange("phi_H", (0, 2, 1), N=100, phi_hop=1)
    params_range2 = ParamsRange("phi_H", (0, 2, 1), N=100, phi_hop=2)
    filename1 = data_store._get_filename(params_range1)
    filename2 = data_store._get_filename(params_range2)
    assert filename1 != filename2

@pytest.mark.parametrize("phi_H", np.arange(0, 2.01, .1))
@pytest.mark.parametrize("phi_hop", np.arange(0, 2.01, .1))
def test_load_save_params(data_store, phi_hop, phi_H):
    params = ChainModel.Params(N=6, phi_hop=phi_hop, harper=Harper(phi_H=phi_H))
    data = ChainModel(params).eigh(eigvals_only=True)
    data_store.save_data(data, params, subdir="save")
    data_loaded = data_store.load_data(params, subdir="save")
    assert np.all(data_loaded == data)

def test_float_data_store():
    class FDS1(FloatDataStore):
        pass
    data_store = FDS1()
    assert data_store.ROOT.name == "FDS1"
