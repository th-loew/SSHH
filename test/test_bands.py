from pathlib import Path

import numpy as np
import pytest

from bands import BandGetter, iterate_bands
from chain_model import ChainModel
from params_range import ParamsRange


@pytest.fixture(scope="module")
def band_getter():
    return BandGetter(N=4000)


@pytest.fixture(scope="module")
def band_getter_large():
    return BandGetter()


@pytest.mark.parametrize("t_inter, t_intra, lower, upper", [
    (.5, 1, .5, 1.5),
    (1, 1, None, 2),
    (1, .5, .5, 1.5)
])
def test_get_bands_ssh(t_inter, t_intra, lower, upper):
    bands = BandGetter.get_bands_system_analytic(ChainModel.Params(N=100, t_inter=t_inter, t_intra=t_intra))
    assert bands[-1] == upper == -bands[0]
    if lower is not None:
        assert bands[-2] == lower == -bands[1]

def test_get_bands_ssh_against_latex(band_getter):
    params = ParamsRange("phi_hop", (.01, 1.99, .02), N=100)
    def latex_func(phi):
        _phi = np.pi * phi / 4
        return np.sqrt(1-np.sin(2*_phi)), np.sqrt(1+np.sin(2*_phi))
    bands = band_getter.get_bands_params_range(params)
    l, u = latex_func(params.param_values)
    assert np.allclose(l, bands[-2, :], atol=1e-10)
    assert np.allclose(u, bands[-1, :], atol=1e-10)

@pytest.mark.parametrize("phi_hop", np.arange(0, 2, .1))
def test_get_bands_for_ssh_model(phi_hop, band_getter):
    delta_E = band_getter.minimum_bandgap * 2
    params = ChainModel.Params(N=band_getter.N, phi_hop=phi_hop)
    bands_analytic = band_getter.get_bands_system_analytic(params)
    bands_numeric = band_getter.get_bands_system_numeric(params)
    assert bands_analytic.shape == bands_numeric.shape, ChainModel(params).eigh(eigvals_only=True)
    assert np.all(np.isnan(bands_analytic) == np.isnan(bands_numeric))
    assert np.allclose(bands_analytic[~np.isnan(bands_analytic)], bands_numeric[~np.isnan(bands_numeric)], atol=delta_E)


@pytest.mark.parametrize("bands_old, bands_new, bands_expected", [
    ([1, 2], [1.001, 2.001], [[1, 1.001], [2, 2.001]]),
    ([1, 2, 3, 4], [1, 2.001, 3.001, 3.999], [[1, 1], [2, 2.001], [3, 3.001], [4, 3.999]]),
    ([1, 2, 3, 6], [1, 2, 3, 4, 5, 6], [[1, 1], [2, 2], [3, 3], [np.nan, 4], [np.nan, 5], [6, 6]]),
    ([1, 4, 5, 6], [1, 2, 3, 4, 5, 6], [[1, 1], [np.nan, 2], [np.nan, 3], [4, 4], [5, 5], [6, 6]]),
    ([1, 2, 3, 4, 5, 6], [1, 4, 5, 6], [[1, 1], [2, np.nan], [3, np.nan], [4, 4], [5, 5], [6, 6]]),
    ([1, 2, 3, 4, 5, 8], [1, 4, 5, 6, 7, 8],
     [[1, 1], [2, np.nan], [3, np.nan], [4, 4], [5, 5], [np.nan, 6], [np.nan, 7], [8, 8]]),
    ([1, np.nan, np.nan, 2, 3, np.nan, np.nan, np.nan, np.nan, 4, 5, 6],
     [1, 2, 3, 4, 5, 6],
     [[1, 1], [np.nan, np.nan], [np.nan, np.nan], [2, 2], [3, 3],
      [np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan], [4, 4], [5, 5], [6, 6]
     ]),
    ([1, 2, 2.1, 3], [1, 3], [[1, 1], [2, np.nan], [2.1, np.nan], [3, 3]]),
    ([1, 3], [1, 2, 2.1, 3], [[1, 1], [np.nan, 2], [np.nan, 2.1], [3, 3]]),
    ([1, np.nan, np.nan, 2, 3, 6], [1, 2, 3, 4, 5, 6],
     [[1, 1], [np.nan, np.nan],[np.nan, np.nan], [2, 2], [3, 3], [np.nan, 4], [np.nan, 5], [6, 6]]),
    ([1, np.nan, np.nan, 4, 5, 6], [1, 2, 3, 4, 5, 6],
     [[1, 1], [np.nan, 2],[np.nan, 3], [4, 4], [5, 5], [6, 6]]),
])
def test__concatenate_bands(bands_old, bands_new, bands_expected, band_getter):
    bands_old = np.array(bands_old)
    bands_new = np.array(bands_new)
    bands_expected = np.array(bands_expected)
    bands = band_getter._concatenate_bands(bands_old, bands_new)
    msg = bands, bands_expected
    assert bands.shape == bands_expected.shape, msg
    assert np.all(np.isnan(bands) == np.isnan(bands_expected)), msg
    assert np.all(bands[~np.isnan(bands)] == bands_expected[~np.isnan(bands_expected)]), msg


def test_get_bands_params_range_ssh(band_getter):
    params_range = ParamsRange("phi_hop", (0, 2, .5), N=10)
    bands = band_getter.get_bands_params_range(params_range)
    assert bands.shape == (4, 5)
    assert np.all(np.isnan(bands[1:3, 2]))
    for i in (0, 1, 3, 4):
        bands_single = band_getter.get_bands_system(ChainModel.Params(N=10, phi_hop=.5*i))
        assert np.all(bands_single == bands[:, i])
    assert bands[0, 2] == pytest.approx(-np.sqrt(2), abs=band_getter.minimum_bandgap)
    assert bands[3, 2] == pytest.approx(np.sqrt(2), abs=band_getter.minimum_bandgap)


def test_get_nonan_bands():
    bands = np.array([1, np.nan, np.nan, 2, 3, np.nan, np.nan, np.nan, np.nan, 4, 5, 6])
    nonan_bands = BandGetter._get_nonan_bands(bands)
    nonan_bands_expected = np.array([1, 1.5, 1.5, 2, 3, 3.5, 3.5, 3.5, 3.5, 4, 5, 6]).reshape(12, 1)
    assert np.allclose(nonan_bands, nonan_bands_expected), (nonan_bands, nonan_bands_expected)


@pytest.mark.parametrize("phi_H", (.5, 1.5))
@pytest.mark.parametrize("phi_hop", (.5, 1, 1.5))
@pytest.mark.skip("This test fails")
def test_bands_params_range_harper_beta(band_getter_large, phi_H, phi_hop):
    params_range = ParamsRange("beta", (0, 2*np.pi, np.pi/5),
                               N=band_getter_large.N, phi_H=phi_H, phi_hop=phi_hop)
    bands = []
    directory = Path(__file__).parent / "res" / "ChainModel" / "eigvals"
    for p in params_range:
        eigvals = np.load(directory / f"{p}.npy")
        bands.append(band_getter_large.get_bands_from_spectrum(eigvals))

    for i, p in enumerate(params_range[1:]):
        msg = bands[0], bands[i+1]
        assert bands[0].shape == bands[i+1].shape, msg
        assert np.allclose(bands[0], bands[i+1], atol=band_getter_large.minimum_bandgap), msg

@pytest.mark.parametrize("bands, tuples_expected", [
    (np.array([1, 2, 3, 4]), [((0, 2), (1, 2)), ((2, 4), (3, 4))]),
    (np.array([1, np.nan, np.nan, 3.3]), [((0, 4), (1, 3.3))]),
    (np.array([1, np.nan, np.nan, np.nan, np.nan, 3.3, 4, np.nan, np.nan, 5, 6, 7]),
     [((0, 6), (1, 3.3)), ((6, 10), (4, 5)), ((10, 12), (6, 7))]),
])
def test_iterate_bands(bands, tuples_expected):
    tuples = tuple(iterate_bands(bands))
    msg = tuples, tuples_expected
    assert len(tuples) == len(tuples_expected), msg
    for i in range(len(tuples)):
        assert tuples[i][0] == tuples_expected[i][0], msg
        assert tuples[i][1][0] == tuples_expected[i][1][0], msg
        assert tuples[i][1][1] == tuples_expected[i][1][1], msg

