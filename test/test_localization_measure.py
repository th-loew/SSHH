import numpy as np
import pytest

from chain_model import ChainModel
from localization_measure import LocalizationMeasure
from harper import Harper

@pytest.fixture(params=LocalizationMeasure.METHODS)
def measure(request) -> LocalizationMeasure:
    return LocalizationMeasure(method=request.param)

@pytest.mark.parametrize("lamda", (0., .001, .01, .1, 1., np.inf))
def test_N_to_infty(lamda, measure):
    N = 100000000000001
    finite_value = measure.theoretical_value(lamda, N)
    infinite_value = measure.theoretical_value(lamda)
    assert finite_value == pytest.approx(infinite_value)

@pytest.mark.parametrize("N", (21, 201, 2001))
@pytest.mark.parametrize("lamda", (0, .001, .01, .1, 1., 10., 100., np.inf))
def test_special_states(N, measure, lamda):
    loc_val_theo = measure.theoretical_value(lamda, N)
    loc_val_theo_min = (1/N if measure.method==LocalizationMeasure.Method.IPR else 0.0)
    if lamda == 0:  # fully localized
        state = np.zeros(N)
        state[0] = 1.0
        assert loc_val_theo == 1
    elif lamda == np.inf:  # fully delocalized
        state = np.ones(N) / np.sqrt(N)
        assert loc_val_theo == loc_val_theo_min
    else:  # exponentially localized
        assert N % 2 == 1
        state = np.exp(-lamda**(-1) * np.abs(np.arange(N)-(N//2)))
        state /= np.linalg.norm(state)
        assert loc_val_theo_min <= loc_val_theo <= 1
    assert measure.measure_one(state) == pytest.approx(loc_val_theo)

@pytest.mark.parametrize("N", (200, 500, 1000, 2000))
def test_random(measure, N):
    state = np.random.random(N)
    state /= np.linalg.norm(state)
    loc_val = measure.measure_one(state)
    if measure.method == LocalizationMeasure.Method.BIN:
        assert loc_val == 0
    else:
        assert 0.0 <= loc_val < .1

def test_2d(measure):
    N = 2000
    states = np.zeros((N, 3))
    states[:, 0] = 1 / np.sqrt(N)
    states[0, 1] = 1
    random_state = np.random.random(N)
    random_state /= np.linalg.norm(random_state)
    states[:, 2] = random_state
    loc_val = measure.measure_all(states)
    assert loc_val.shape == (3, )
    assert loc_val[0] == pytest.approx(0, abs=1e-2)
    assert loc_val[1] == pytest.approx(1, abs=1e-6)
    assert loc_val[2] == pytest.approx(0, abs=1e-1)

def test_ipr_small_array():
    state = np.asarray([0.1, 0.3, 0.3, 0.4, 0.8, 0.1])
    norm_theo = 1.0
    assert norm_theo, pytest.approx(np.linalg.norm(state), abs=1e-6)
    loc_val_theo = 1e-4 *(1 + 2 * 81 + 16*16 + 64*64 + 1)
    loc_val = LocalizationMeasure(method=LocalizationMeasure.Method.IPR).measure_one(state)
    assert loc_val_theo, pytest.approx(loc_val, abs=1e-6)

def test_measure_systems(measure):
    N = 200
    p1 = ChainModel.Params(N=N, phi_hop=1.1, harper=Harper(phi_H=1.999))
    p2 = ChainModel.Params(N=2000, phi_hop=1.1, harper=Harper(phi_H=0.001))
    params = p1, p1, p2
    l1 = 1.0
    abs = 1e-2
    L_all = measure.measure_all(p1)
    L_system_one = measure.measure_system(p1)
    L_system_all = measure.measure_systems(params)

    assert L_all.shape == (N, )
    assert np.mean(L_all) == pytest.approx(l1, abs=abs)

    assert L_system_one == pytest.approx(l1, abs=abs)

    assert L_system_all.shape == (3, )
    assert L_system_one == L_system_all[0]
    assert L_system_one == L_system_all[1]
    assert L_system_all[2] == pytest.approx(measure.theoretical_value(np.inf, N), abs=abs)

def test_measure_systems_all_measures():

    methods = LocalizationMeasure.METHODS
    params = (
        ChainModel.Params(N=20, phi_hop=1, harper=Harper(phi_H=1.9)),
        ChainModel.Params(N=20, phi_hop=1.1, harper=Harper(phi_H=1)),
        ChainModel.Params(N=20, phi_hop=1.1, harper=Harper(phi_H=1.9)),
    )

    results = {}
    for m in methods:
        measure = LocalizationMeasure(m)
        results[m] = tuple(float(x) for x in measure.measure_systems(params))

    results_all_np = LocalizationMeasure.measure_systems_all_methods(params)
    results_all = {key: tuple(float(x) for x in value) for key, value in results_all_np.items()}

    assert results == results_all

def test_measure_system_all_measures():

    methods = LocalizationMeasure.METHODS
    params = ChainModel.Params(N=20, phi_hop=1.1, harper=Harper(phi_H=1.9))

    results = {}
    for m in methods:
        measure = LocalizationMeasure(m)
        results[m] = measure.measure_system(params)

    results_all = LocalizationMeasure.measure_system_all_methods(params)

    assert results == results_all


def test_edge():
    N = 100
    states = np.zeros((N, 5))
    states[0, 0] = 1
    states[-1, 1] = 1
    states[:, 2] = 1 / np.sqrt(N)
    states[0, 3] = states[-1, 3] = 1 / np.sqrt(2)
    states[:N//2, 4] = 1 / np.sqrt(N//2)
    edge_measure = LocalizationMeasure(method=LocalizationMeasure.Method.EDGE)
    loc_val = edge_measure.measure_all(states)
    assert loc_val.shape == (5, )
    assert loc_val[0] == pytest.approx(-1, abs=1e-6)
    assert loc_val[1] == pytest.approx(1, abs=1e-6)
    assert abs(loc_val[2]) < .1
    assert loc_val[3] == pytest.approx(0, abs=1e-6)
    assert -.9 < loc_val[4] < -.1

def test_edge_shh():
    N = 100
    # add minimal harper to SSH to avoid degeneracy (left plus right)
    params = ChainModel.Params(N=N, phi_hop=1.9, harper=Harper(phi_H=.001))
    edge_measure = LocalizationMeasure(method=LocalizationMeasure.EDGE_METHOD)
    loc_val = edge_measure.measure_all(params)
    loc_val_edge = loc_val[N//2-1:N//2+1]
    assert abs(loc_val_edge[0]) == pytest.approx(1, abs=1/N)
    assert abs(loc_val_edge[1]) == pytest.approx(1, abs=1/N)
    abs_loc_val = np.abs(loc_val)
    abs_loc_val[N//2-1:N//2+1] = 0
    assert np.max(abs_loc_val) < .5