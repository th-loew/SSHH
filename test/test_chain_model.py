import numpy as np
import pytest

from chain_model import ChainModel, Localizations
from harper import Harper

r = np.sqrt

@pytest.mark.parametrize("kwargs", [{"alpha": 1}, {"varphi": 1}, {"v": 1, "w": 2}, {"phi_H": 2}])
def test_invalid_args(kwargs):
    with pytest.raises(ValueError):
        ChainModel.Params(N=10, **kwargs)

@pytest.mark.parametrize("kwargs", [{"N": 10, "phi_hop": .1}])
def test_valid_args(kwargs):
    params = ChainModel.Params(**kwargs)
    for key, value in kwargs.items():
        assert getattr(params, key) == value

@pytest.mark.parametrize("phi_H, phi_hop, V, t_intra, t_inter", [
    (0, 0, 0, 1, 0), (1, 0, 1, r(.5), 0), (2, 0, r(2), 0, 0),
    (0, 1, 0, r(.5), r(.5)), (1, 1, 1, .5, .5), (2, 1, r(2), 0, 0),
    (0, 2, 0, 0, 1), (1, 2, 1, 0, r(.5)), (2, 2, r(2), 0, 0)
])
def test_param_phases_special_values(phi_H, phi_hop, V, t_intra, t_inter):
    alpha = Harper.IrrationalFrequency.GOLDEN_NUMBER_REDUCED.value
    cos0 = np.cos(2 * np.pi * alpha)
    model = ChainModel(ChainModel.Params(N=10, harper=Harper(phi_H=phi_H, alpha=alpha), phi_hop=phi_hop))
    assert model.H_diag0[0] == pytest.approx(cos0 * V)
    assert model.H_diag1[0] == t_intra
    assert model.H_diag1[1] == t_inter

def test_potential_not_symmetric():
    N = 10
    pot = ChainModel(ChainModel.Params(N=N, harper=Harper(phi_H=2))).H_diag0
    for i in range(N//2):
        assert 0 != pytest.approx(pot[i] - pot[N-1-i])

@pytest.mark.parametrize("N", [4, 10, 100, 128, 200])
@pytest.mark.parametrize("kwargs, ssh_type", [({"t_intra": 1, "t_inter": 0}, "trivial"),
                                              ({"t_intra": 0, "t_inter": 1}, "topological"),
                                              ({"phi_hop": 0}, "trivial"),
                                              ({"phi_hop": 2}, "topological")])
def test_phi_hop_trivial_and_topological(N, kwargs, ssh_type):

    offset = 1 * (ssh_type == "trivial")

    hopping = ChainModel(ChainModel.Params(N=N, **kwargs)).H_diag1
    assert not np.any(hopping[offset::2])

@pytest.mark.parametrize("phi_hop", np.arange(0, 2, .001))
def test_hopping_amplitudes_add_to_one(phi_hop):
    params = ChainModel.Params(N=10, phi_hop=phi_hop)
    assert params.t_intra**2 + params.t_inter**2 == pytest.approx(1)

@pytest.mark.parametrize("N", [4, 10, 100, 128, 200, 500, 1000])
def test_eigvals_near_SSH(N):
    ssh = ChainModel(ChainModel.Params(N=N, phi_hop=1))
    nearly_ssh = ChainModel(ChainModel.Params(N=N, phi_hop=1.001))
    eigvals_ssh = ssh.eigh(eigvals_only=True)
    eigvals_nearly_ssh = nearly_ssh.eigh(eigvals_only=True)
    assert np.allclose(eigvals_ssh, eigvals_nearly_ssh, atol=1e-3)

@pytest.mark.parametrize("N", [4, 10, 100])
@pytest.mark.parametrize("phi_hop", np.arange(0, 2, .1))
@pytest.mark.parametrize("phi_H", np.arange(0, 2, .1))
def test_eigvecs(N, phi_hop, phi_H):
    model = ChainModel(ChainModel.Params(N=N, phi_hop=phi_hop, harper=Harper(phi_H=phi_H)))
    eigvals, eigvecs = model.eigh(eigvals_only=False)
    y = np.matmul(model.H, eigvecs)
    for i in range(N):
        assert np.allclose(y[:, i], eigvals[i] * eigvecs[:, i])

@pytest.mark.parametrize("kwargs, expected", [
    ({"phi_hop": 1}, "plain lattice"),
    ({"phi_hop": 1, "harper": Harper(phi_H=.01)}, "Harper"),
    ({"phi_hop": .756}, "SSH"),
    ({"phi_hop": .756, "harper": Harper(phi_H=.01)}, "SSH-Harper"),
    pytest.param({"nnn_factor": .5}, None)
])
def test_name(kwargs, expected):
    model = ChainModel(ChainModel.Params(N=100, **kwargs))
    if expected:
        assert model.name == expected
    else:
        with pytest.raises(NotImplementedError):
            getattr(model, "name")

@pytest.mark.parametrize("localization, expected", [
    (Localizations.LATTICE_SITE, (1, 2, 3, 4)),
    (Localizations.UNIT_CELL, (1, 1, 2, 2)),
])
def test_sites(localization, expected):
    model = ChainModel(ChainModel.Params(N=4, localization=localization))
    assert tuple(model.sites) == expected

def test_equal():
    params_1 = ChainModel.Params(N=100, phi_hop=1)
    params_2 = ChainModel.Params(N=100, t_inter=np.sqrt(.5), t_intra=np.sqrt(.5))
    assert params_1 == params_2


@pytest.mark.parametrize("nnn_factor, expected", [(0, 0), (.5, .5*np.sqrt(.5))])
def test_diag_2(nnn_factor, expected):
    model = ChainModel(ChainModel.Params(N=4, nnn_factor=nnn_factor))
    diag2 = np.diag(model.H, 2)
    assert np.all(diag2 == expected)

def test_N_M():
    assert ChainModel.Params(N=8).M == 4
    assert ChainModel.Params(M=4).N == 8


@pytest.mark.parametrize("phi_hop", np.arange(0, 2, .1))
@pytest.mark.parametrize("phi_H", np.arange(0, 2, .1))
@pytest.mark.parametrize("N", [20, 100, 200])
def test_frobenius_norm_and_quadratic_mean_of_eigenvalues(N, phi_hop, phi_H):
    params = ChainModel.Params(N=N, phi_hop=phi_hop, harper=Harper(phi_H=phi_H))
    model = ChainModel(params)
    tol = 1 / (N-1)
    frobenius_norm = np.linalg.norm(model.H, ord='fro')
    assert pytest.approx(np.sqrt(N), rel=tol) == frobenius_norm
    eigvals = model.eigh(eigvals_only=True)
    quadratic_mean = np.sqrt(np.mean(eigvals**2))
    assert pytest.approx(1, abs=tol) == quadratic_mean
    assert pytest.approx(frobenius_norm) == quadratic_mean * np.sqrt(N)


def test_params_copy():
    params = ChainModel.Params(N=10, phi_hop=1.43,
                               harper=Harper(phi_H=1.45, alpha=Harper.IrrationalFrequency.GOLDEN_NUMBER, beta=.23))
    params_copy = params.copy_with_changed_N(10)
    assert params == params_copy
    params_altered_copy = params.copy_with_changed_N(12)
    assert params != params_altered_copy
    assert params_altered_copy.N == 12
    assert params_altered_copy.M == 6
    with pytest.raises(ValueError):
        params.copy_with_changed_N(11)

def test_harper_not_none():

    params = ChainModel.Params(N=10, harper=Harper(phi_H=.7, alpha=Harper.IrrationalFrequency.PI_THIRD, beta=.23))
    assert params.harper is not None
    assert params.harper_not_none == params.harper

    params = ChainModel.Params(N=10, harper=None)
    assert params.harper is None
    with pytest.raises(ValueError):
        getattr(params, "harper_not_none")
