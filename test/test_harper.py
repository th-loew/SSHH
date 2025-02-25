import numpy as np
import pytest

from harper import Harper, AlphaGetter

from python.harper import beta_symmetric_harper_hamiltonian


def test_repr_none():
    assert repr(Harper(phi_H=0)) == "None"

@pytest.mark.parametrize("phi", np.arange(0, 2, 0.1))
def test_repr_no_numpy(phi):
    assert "numpy" not in repr(Harper(phi_H=phi))

def test_equal():
    assert Harper(phi_H=1) == Harper(amplitude=1)

@pytest.mark.parametrize("field_name, value",
                         [
                             ("alpha", 1),
                             ("frequency", Harper.IrrationalFrequency.GOLDEN_NUMBER_REDUCED),
                             ("offset_periods", .1),
                             ("varphi", .1),
                             ("phi", .1),
                             ("phi_hop", .1),
                         ]
                         )
def test_invalid_args(field_name, value):
    with pytest.raises(ValueError):
        Harper(**{field_name: value})

@pytest.mark.parametrize("kwargs",
                         [
                             {"alpha": Harper.IrrationalFrequency.GOLDEN_NUMBER_REDUCED, "beta": .1, "amplitude": .1},
                             {"alpha": Harper.IrrationalFrequency.SEC_MOST_IRRAT, "beta": .55, "phi_H": .1},
                         ]
                         )
def test_valid_args(kwargs):
    harper = Harper(**kwargs)
    for key, value in kwargs.items():
        assert getattr(harper, key) == value


@pytest.mark.parametrize("key", ("phi_H", "amplitude"))
@pytest.mark.parametrize("value", np.linspace(0, 1, 11))
def test_localization_infinite(key, value):
    harper = Harper(**{key: value})
    assert np.isinf(harper.localization_length)

@pytest.mark.parametrize("kwargs", [{"phi_H": 2}, {"amplitude": np.inf}])
def test_localization_zero(kwargs):
    harper = Harper(**kwargs)
    assert harper.localization_length == 0

@pytest.mark.parametrize("key, values", [("phi_H", np.linspace(1, 2, 11)),
                                         ("amplitude", [np.exp(v) for v in np.arange(10)])])
def test_localization_decreasing(key, values):
    localization_lengths = [Harper(**{key: value}).localization_length for value in values]
    assert np.all(np.diff(localization_lengths) < 0)

def test_frequency():
    value = 0.4456
    f = Harper.Frequency(value)
    assert f.name == "0.4456"
    assert type(f.value) == float
    assert f.value == value
    assert str(f) == "0.4456"

def test_irrational_frequency():
    f = Harper.IrrationalFrequency.GOLDEN_NUMBER
    assert f.name == "GOLDEN_NUMBER"
    assert f.value == (1 + 5**(1/2)) / 2
    assert str(f) == "GOLDEN_NUMBER"
    assert isinstance(f, Harper.Frequency)

@pytest.mark.parametrize("N", [50, 1000, 5000, 40000])
@pytest.mark.parametrize("alpha", [1, .5, .1, 2, .02])
def test_alpha_getter_degree_of_irrationality_rational(N, alpha):
    alpha_getter = AlphaGetter()
    assert alpha_getter.degree_of_irrationality(alpha, N) == 0

@pytest.mark.parametrize("N", [5000, 40000])
@pytest.mark.parametrize("alpha", list(Harper.IrrationalFrequency))
def test_alpha_getter_degree_of_irrationality_irrational(N, alpha):
    alpha_getter = AlphaGetter()
    assert alpha_getter.degree_of_irrationality(alpha, N) == pytest.approx(1, abs=1e-2)

@pytest.mark.parametrize("N", [1000, 5000, 40000])
def test_alpha_getter_degree_of_irrationality_all_values(N):
    alphas = np.arange(.1, .11, 1e-5)
    alpha_getter = AlphaGetter()
    values = [alpha_getter.degree_of_irrationality(alpha, N) for alpha in alphas]
    for a, b in zip(alphas, values):
        print(a, b)
    assert min(values) >= 0
    assert max(values) <= 1
    tol = 1e-3
    assert min(values) == pytest.approx(0, abs=tol)
    assert max(values) == pytest.approx(1, abs=tol)


@pytest.mark.parametrize("modus", ["uniform", Harper.IrrationalFrequency.GOLDEN_NUMBER])
def test_alpha_getter_get_range(modus):
    start, stop = .1, .2
    tol = 1e-3
    alpha_getter = AlphaGetter(modus=modus, search_factor=10)
    alphas = alpha_getter.get_range(start=start, stop=stop)
    assert len(alphas) == 100
    assert alphas[0] == pytest.approx(start, abs=tol)
    assert alphas[-1] == pytest.approx(stop, abs=tol)

def test_alpha_getter_get_range_mean_doi():
    alpha_getter_uniform = AlphaGetter(modus="uniform", search_factor=10)
    alpha_getter_golden_number = AlphaGetter(modus=Harper.IrrationalFrequency.GOLDEN_NUMBER, search_factor=10)
    alphas_uniform = alpha_getter_uniform.get_range(count=20)
    alphas_golden_number = alpha_getter_golden_number.get_range(count=20)
    mean_doi_uniform = np.mean([alpha_getter_uniform.degree_of_irrationality(alpha, 5000) for alpha in alphas_uniform])
    mean_doi_golden_number = np.mean([alpha_getter_golden_number.degree_of_irrationality(alpha, 5000) for alpha in alphas_golden_number])
    assert mean_doi_uniform < mean_doi_golden_number

def test_beta_symmetric_harper_hamiltonian():
    N = 2
    alpha = Harper.Frequency(.3 / np.pi)
    beta_0 = np.pi - .9
    assert beta_symmetric_harper_hamiltonian(alpha, N) == (beta_0, beta_0 + np.pi)
