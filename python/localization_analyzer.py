import numpy as np

from bands import BandGetter, iterate_bands
from chain_model import ChainModel
from data_store import FloatDataStore
from localization_measure import LocalizationMeasure
from params_range import ParamsRange
from utils import FloatArray


class LocalizationAnalyzer(FloatDataStore):

    SUBDIRS = "bands",

    def __init__(self,
                 method: LocalizationMeasure.Method = LocalizationMeasure.Method.BIN,
                 band_getter: BandGetter | None = None,
                 **kwargs):
        self._localization_measure = LocalizationMeasure(method)
        self._band_getter = band_getter or BandGetter()
        super().__init__(**kwargs)

    def _get_subsubdir(self) -> str | None:
        return (f"{self._localization_measure._get_subsubdir()}, "  # pylint: disable=protected-access
                f"{self._band_getter._get_subsubdir()}")  # pylint: disable=protected-access

    def analyze_bands(self, params: ChainModel.Params | ParamsRange) -> tuple[FloatArray, FloatArray]:
        bands = self._band_getter.get_bands_system(params) if isinstance(params, ChainModel.Params) \
            else self._band_getter.get_bands_params_range(params)

        try:
            return bands, self.load_data(params, subdir="bands")
        except self.LoadException:
            pass

        params_list = list(params) if isinstance(params, ParamsRange) else [params]
        bands = bands.reshape((-1, len(params_list)))
        loc_bands = np.empty_like(bands)
        for j, p in enumerate(params_list):
            assert isinstance(p, ChainModel.Params)
            loc_raw = self._localization_measure.measure_all(p)
            eigvals = ChainModel(p).eigh(eigvals_only=True)
            for (i, i_upper), (lower, upper) in iterate_bands(bands[:, j]):
                idx = np.logical_and(eigvals >= lower, eigvals <= upper)
                loc_bands[i:i_upper, j] = loc_raw[idx].mean()

        if isinstance(params, ChainModel.Params):
            bands = bands.reshape(-1)
            loc_bands = loc_bands.reshape(-1)
        self.save_data(loc_bands, params, subdir="bands")
        return bands, loc_bands
