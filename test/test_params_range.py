import numpy as np
import pytest

from chain_model import ChainModel
from harper import Harper, AlphaGetter
from params_range import ParamsRange


def test_wrong_args():
    with pytest.raises(ValueError):
        ParamsRange("not_a_param", (1, 3, 1), N=10)
    with pytest.raises(ValueError):
        ParamsRange("phi_H", (1, 3, 1), phi_H=.1, N=10)
    with pytest.raises(ValueError):
        ParamsRange("phi_H", (3, 1, 1), phi_H=.1, N=10)
    with pytest.raises(ValueError):
        ParamsRange("phi_H", (1, 2.5, 1), phi_H=.1, N=10)
    with pytest.raises(ValueError):
        ParamsRange("phi_H", (1, 2, .2), harper=Harper(alpha=Harper.IrrationalFrequency.PI), N=10)
    with pytest.raises(ValueError):
        ParamsRange("alpha", (1, 2.5, 1), alpha=.1, N=10)
    with pytest.raises(ValueError):
        ParamsRange("alpha", (1, 2, .2), harper=Harper(alpha=Harper.IrrationalFrequency.PI), N=10)
    with pytest.raises(ValueError):
        ParamsRange("phi_hop", (1, 2, .2), phi_H=.1, harper=Harper(alpha=Harper.IrrationalFrequency.PI), N=10)
    with pytest.raises(ValueError):
        ParamsRange("phi_hop", (1, 2, .2), alpha=Harper.IrrationalFrequency.PI, harper=Harper(phi_H=.1), N=10)
    with pytest.raises(ValueError):
        ParamsRange("phi_hop", (1, 2, .2, Harper.IrrationalFrequency.GOLDEN_NUMBER), alpha=Harper.IrrationalFrequency.PI, harper=Harper(phi_H=.1), N=10)

def test_phi_H():
    params = ParamsRange("phi_H", (0, 2, .1), N=10)
    assert len(params) == 21
    p0 = params[0]
    for i, p in enumerate(params):
        assert p.harper.phi_H == pytest.approx(.1 * i)
        for field in ChainModel.Params.model_fields:
            if field not in ("harper", "t_intra", "t_inter"):
                assert getattr(p, field) == getattr(p0, field), f"field: {field}"
        for field in Harper.model_fields:
            if field not in ("phi_H", "amplitude"):
                assert getattr(p.harper, field) == getattr(p0.harper, field), f"field: {field}"

def test_phi_hop():
    params = tuple(ParamsRange("phi_hop", (0, 2, .1), N=10, phi_H=.5))
    assert len(params) == 21
    p0 = params[0]
    for i, p in enumerate(params):
        assert p.phi_hop == pytest.approx(.1 * i)
        for field in ChainModel.Params.model_fields:
            if field not in ("phi_hop", "t_intra", "t_inter"):
                assert getattr(p, field) == getattr(p0, field), f"field: {field}"

def test_repr_with_harper_none():
    params1 = ParamsRange("phi_H", (0, 2, .1), N=10, alpha=Harper.IrrationalFrequency.GOLDEN_NUMBER)
    params2 = ParamsRange("phi_H", (0, 2, .1), N=10, alpha=Harper.IrrationalFrequency.GOLDEN_NUMBER_REDUCED)
    assert str(params1) != str(params2)


def test_beta():
    beta_range = np.arange(0, 2.0000001*np.pi, np.pi/100)
    params = ParamsRange("beta", (0, 2*np.pi, np.pi/100), N=10, phi_H=.5)
    assert len(params) == 201
    assert np.allclose(beta_range, np.array([p.harper.beta for p in params]))

def test_alpha_irrational():
    alpha_range = np.arange(0, 2.00001*np.pi, .1*np.pi)
    params = ParamsRange("alpha", (0, 2, .1, Harper.IrrationalFrequency.PI), N=10, phi_H=.5)
    assert len(params) == 21
    assert np.allclose(alpha_range, np.array([p.harper.alpha for p in params]))

def test_alpha_getter():
    count = 10
    alpha_getter_kwargs = {"search_factor": 10}
    alpha_range = AlphaGetter(**alpha_getter_kwargs).get_range(count=count)
    params = ParamsRange("alpha", (None, None, count), N=5000, phi_H=.5,
                         alpha_getter_kwargs=alpha_getter_kwargs)
    assert len(params) == 10
    print(params)
    assert np.all(alpha_range == np.array([p.harper.alpha for p in params]))