import numpy as np
import pytest

from chain_model import ChainModel
from harper import Harper
from localization_analyzer import LocalizationAnalyzer
from params_range import ParamsRange
from test_bands import band_getter


@pytest.mark.parametrize("phi_hop", np.arange(.6, 1.5, .2))
@pytest.mark.parametrize("phi_H, loc_expected", [(0, 0), (.01, 0), (1.99, 1), (2, 1)])
def test_analyze_bands_extremal_values(phi_hop, phi_H, loc_expected, band_getter):
    params = ChainModel.Params(N=500, phi_hop=phi_hop, harper=Harper(phi_H=phi_H))
    localization_analyzer = LocalizationAnalyzer(band_getter=band_getter)
    bands, loc_bands = localization_analyzer.analyze_bands(params)
    assert np.all(bands == band_getter.get_bands_system(params))
    assert np.all(loc_bands == loc_expected)


@pytest.mark.parametrize("phi_H, loc_expected", [(0, 0), (.01, 0), (1.99, 1), (2, 1)])
def test_analyze_bands_extremal_values_range(phi_H, loc_expected, band_getter):
    params = ParamsRange("phi_hop", (.6, 1.4, .2), N=500, phi_H=phi_H)
    localization_analyzer = LocalizationAnalyzer(band_getter=band_getter)
    bands, loc_bands = localization_analyzer.analyze_bands(params)
    assert np.all(loc_bands == loc_expected)
