from typing import Generator

import numpy as np

from chain_model import ChainModel
from data_store import FloatDataStore
from params_range import ParamsRange
from utils import FloatArray

DEFAULT_MINIMUM_BANDGAP = 1e-2
DEFAULT_N = 40000


def iterate_bands(bands: FloatArray) -> Generator[tuple[tuple[int, int], tuple[float, float]], None, None]:
    assert len(bands.shape) == 1
    i = 0
    while i < len(bands):
        i_upper = i+1
        while np.isnan(bands[i_upper]):
            i_upper += 1
        yield (i, i_upper+1), (bands[i], bands[i_upper])
        i = i_upper + 1


class BandGetter(FloatDataStore):

    SUBDIR_SYSTEMS = "systems"
    SUBDIR_PARAMS_RANGES = "params_ranges"
    SUBDIRS = SUBDIR_SYSTEMS, SUBDIR_PARAMS_RANGES


    def __init__(self, minimum_bandgap: float = DEFAULT_MINIMUM_BANDGAP, N: int = DEFAULT_N, **kwargs):
        self.minimum_bandgap = minimum_bandgap
        self.N = N
        super().__init__(**kwargs)

    def _get_subsubdir(self) -> str | None:
        return f"N={self.N} minimum_bandgap={self.minimum_bandgap}"

    def get_bands_params_range(self, params_range: ParamsRange) -> FloatArray:
        naming = str(params_range).replace(str(params_range[0]),
                                           str(params_range[0].copy_with_changed_N(self.N)))
        try:
            return self.load_data(naming, subdir=self.SUBDIR_PARAMS_RANGES)
        except self.LoadException:
            bands = np.empty(0)
            for params in params_range:
                new_bands = self.get_bands_system(params)
                bands = new_bands if not bands.size else self._concatenate_bands(bands, new_bands)
            self.save_data(bands, naming, subdir=self.SUBDIR_PARAMS_RANGES)
            return bands

    def get_nonan_bands_params_range(self, params_range: ParamsRange) -> FloatArray:
        bands = self.get_bands_params_range(params_range)
        return self._get_nonan_bands(bands)

    @staticmethod
    def _get_nonan_bands(bands: FloatArray) -> FloatArray:
        bands = bands if len(bands.shape) == 2 else bands.reshape(len(bands), 1)
        for j in range(bands.shape[1]):
            i = 0
            while i < bands.shape[0]:
                if np.isnan(bands[i, j]):
                    i1 = i+1
                    while np.isnan(bands[i1, j]):
                        i1 += 1
                    for k in range(i, i1):
                        bands[k, j] = (bands[i-1, j] + bands[i1, j]) / 2
                    i = i1
                else:
                    i += 1
        return bands

    def get_bands_system(self, params: ChainModel.Params) -> FloatArray:
        params = params.copy_with_changed_N(self.N)
        try:
            return self.load_data(params, subdir=self.SUBDIR_SYSTEMS)
        except self.LoadException:
            try:
                bands = self.get_bands_system_analytic(params)
            except ValueError:
                bands = self.get_bands_system_numeric(params)
            self.save_data(bands, params, subdir=self.SUBDIR_SYSTEMS)
            return bands

    def get_bands_system_numeric(self, params: ChainModel.Params) -> FloatArray:
        params = params.copy_with_changed_N(self.N)
        spectrum = ChainModel(params).eigh(eigvals_only=True)
        return self.get_bands_from_spectrum(spectrum)

    @staticmethod
    def get_bands_system_analytic(params: ChainModel.Params) -> FloatArray:
        name = ChainModel(params).name
        if name == "plain lattice":  # t_intra == t_inter
            return np.array([-2*params.t_intra, 2*params.t_intra])
        if name == "SSH":
            v = params.t_intra
            w = params.t_inter
            return np.array([-v-w, -abs(v-w), abs(v-w), v+w])
        raise ValueError(f"Band calculation for {name} model not possible.")

    def get_bands_from_spectrum(self, spectrum: FloatArray) -> FloatArray:
        """
        Get bands from spectrum.
        :param spectrum: Energies of the system. Assumed to be sorted.
        """
        bands: list[float] = []
        previous_in_band = False

        for i_, _E in np.ndenumerate(spectrum):
            E = float(_E)
            i = i_[0]
            in_band = False if i in (0, len(spectrum)-1) else \
                (E - spectrum[i-1] < self.minimum_bandgap and spectrum[i+1] - E  < self.minimum_bandgap)
            if in_band and not previous_in_band:
                bands.append(E)
                previous_in_band = True
            elif not in_band and previous_in_band:
                bands.append(float(spectrum[i-1]))
                previous_in_band = False
        return np.array(bands)

    def _concatenate_bands(self, bands_old: FloatArray, bands_new: FloatArray) -> FloatArray:
        bands_old = bands_old if len(bands_old.shape) == 2 else bands_old.reshape(len(bands_old), -1)
        n_bands_old = bands_old.shape[0] // 2
        n_bands_new = bands_new.shape[0] // 2
        bands = np.concatenate((bands_old[:1, :], bands_new[:1].reshape((-1, 1))), axis=1)
        def append(b_old: FloatArray, b_new: FloatArray):
            nonlocal bands
            bands_to_add = np.concatenate((b_old, b_new.reshape((-1, 1))), axis=1)
            bands = np.concatenate((bands, bands_to_add), axis=0)
        i_old = 1
        i_new = 1
        while i_old < n_bands_old and i_new < n_bands_new:
            slice_old = bands_old[2*i_old-1:, -1]
            slice_new = bands_new[2*i_new-1:]
            gap_old = slice_old[:2]
            gap_new = slice_new[:2]
            new_to_old = self._find_best_match(gap_new, slice_old)
            if new_to_old is None:  # only nans in slice_old
                append(bands_old[2*i_old-1:2*i_old+1, :], bands_new[2*i_new-1:2*i_new+1])
                i_old += 1
                i_new += 1
            elif np.isnan(gap_old[0]):
                gap_old_best = slice_old[2*new_to_old:2*new_to_old+2]
                best_old_to_new = self._find_best_match(gap_old_best, slice_new)
                if best_old_to_new == 0:
                    append(bands_old[2*i_old-1:2*i_old+1, :], np.nan*np.ones((2, 1)))
                    i_old += 1
                else:
                    append(bands_old[2*i_old-1:2*i_old+1, :], bands_new[2*i_new-1:2*i_new+1])
                    i_old += 1
                    i_new += 1
            else:
                old_to_new = self._find_best_match(gap_old, slice_new)
                assert old_to_new is not None
                if old_to_new == new_to_old == 0:
                    append(bands_old[2*i_old-1:2*i_old+1, :], bands_new[2*i_new-1:2*i_new+1])
                    i_new += 1
                    i_old += 1
                elif new_to_old == 0:
                    append(np.nan*np.ones((2*old_to_new, bands_old.shape[1])), bands_new[2*i_new-1:2*(i_new+old_to_new)-1])
                    i_new += old_to_new
                elif old_to_new == 0:
                    append(bands_old[2*i_old-1:2*(i_old+new_to_old)-1, :], np.nan*np.ones((2*new_to_old, 1)))
                    i_old += new_to_old
                else:
                    raise RuntimeError("This case should not happen.")
        if i_old < n_bands_old:
            append(bands_old[2*i_old-1:-1, :], np.nan*np.ones((2*(n_bands_old-i_old), 1)))
        elif i_new < n_bands_new:
            append(np.nan*np.ones((2*(n_bands_new-i_new), bands_old.shape[1])), bands_new[2*i_new-1:-1])
        append(bands_old[-1:, :], bands_new[-1:])
        return bands

    @staticmethod
    def _find_best_match(a: FloatArray, b: FloatArray) -> int | None:
        d = np.inf
        i_best = None
        for i in range(len(b) // 2):
            d_ = float(np.linalg.norm(a - b[2*i:2*i+2]))
            if np.isnan(d_):
                continue
            elif d_ < d:
                d = d_
                i_best = i
            else:
                break
        return i_best
