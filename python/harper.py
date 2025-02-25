from __future__ import annotations

from enum import Enum
from typing import Any, Iterable, Literal

import numpy as np
from pydantic import BaseModel, Field, model_validator

from data_store import FloatDataStore
from utils import ParamPhase, param_phase_to_values, logger, FloatArray


class Harper(BaseModel):

    class Config:
        arbitrary_types_allowed = True

    def __repr__(self):
        if not self:
            return "None"
        return super().__repr__()

    class Frequency(float):

        def __getattr__(self, item):
            if item == "name":
                return str(self)
            if item == "value":
                return float(self)
            return getattr(super(), item)

    class IrrationalFrequency(Frequency, Enum):
        GOLDEN_NUMBER = (1 + 5**(1/2)) / 2
        GOLDEN_NUMBER_REDUCED = 5**(1/2) / 2
        SEC_MOST_IRRAT = 1 + 2**(1/2)
        SEC_MOST_IRRAT_RED = 2**(1/2)
        PI_THIRD = np.pi / 3
        SQRT_2_NORMALIZED = np.sqrt(2) / 1.4
        SQRT_1_2_NORMALIZED = 1.4 / np.sqrt(2)
        PI = np.pi

        def __repr__(self):
            return self.name

        def __str__(self):
            return self.name

    amplitude: float = Field(None, description="Harper amplitude")  # type: ignore[assignment]
    phi_H: ParamPhase = Field(None, description="parameter phase (in units of pi/4)")
    alpha: Frequency = Field(IrrationalFrequency.GOLDEN_NUMBER_REDUCED, description="Harper frequency")
    beta: float = Field(0, ge=0, le=2*np.pi, description="Harper potential offset")

    @model_validator(mode='before')
    @classmethod
    def check_deprecated_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            assert "phi" not in data, "phi is deprecated, use phi_H instead"
            assert "varphi" not in data, "varphi is deprecated, use phi_H instead"
            assert "frequency" not in data, "frequency is deprecated, use alpha instead"
            assert "offset_periods" not in data, "offset_periods is deprecated, use beta instead"
        return data

    @model_validator(mode='before')
    @classmethod
    def check_wrong_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            assert "phi_hop" not in data,\
                f"phi_hop is not a valid parameter for {cls.__name__}, did you mean phi_H?"
        return data

    @model_validator(mode='before')
    @classmethod
    def check_fields_defined(cls, data: Any) -> Any:
        if isinstance(data, dict):
            assert "amplitude" in data or "phi_H" in data, "One of amplitude or phi_H must be defined"
            assert "amplitude" not in data or "phi_H" not in data, "Only one of amplitude or phi_H can be defined"
        return data

    @model_validator(mode='after')
    def compute_amplitude(self) -> Harper:
        if self.phi_H is not None:
            if self.phi_H == 1:
                self.amplitude = 1
            else:
                self.amplitude = float(np.sqrt(2) * param_phase_to_values(self.phi_H)[0])
        return self

    @model_validator(mode='before')
    @classmethod
    def check_alpha(cls, data: Any) -> Any:
        if isinstance(data, dict):
            try:
                alpha = data["alpha"]
            except KeyError:
                return data
            if isinstance(alpha, Harper.Frequency):
                return data
            if isinstance(alpha, float):
                data["alpha"] = Harper.Frequency(alpha)
                return data

    def __bool__(self):
        return bool(self.amplitude)

    def __eq__(self, other):
        if not other:
            return not self
        if other.__class__ is self.__class__:
            return self.amplitude == other.amplitude and self.alpha == other.alpha \
                and self.beta == other.beta
        logger.warning("Comparing Harper potential with %s is not recommended.", other.__class__.__name__)
        return False

    @property
    def hopping_amplitude_factor(self) -> float | None:
        if self.phi_H is None:
            return None
        return param_phase_to_values(self.phi_H)[1]

    @property
    def localization_length(self) -> float:
        if self.phi_H is None:
            V = self.amplitude
            t = 1.0
        else:
            V, t = param_phase_to_values(self.phi_H)
        V , t = abs(V), abs(t)
        if V <= t:
            return np.inf
        if t == 0:
            return 0
        return 1 / np.log(V / t)


class AlphaGetter(FloatDataStore):

    def __init__(self, with_inverse: bool = True, search_factor: int = 100,
                 modus: Literal["uniform"] | Harper.IrrationalFrequency = Harper.IrrationalFrequency.GOLDEN_NUMBER,
                 **data_store_kwargs):
        self._with_inverse = with_inverse
        self._search_factor = search_factor
        self._modus = modus
        super().__init__(**data_store_kwargs)

    def _get_subsubdir(self) -> str | None:
        return f"with_inverse={self._with_inverse}_search_factor={self._search_factor}_modus={self._modus}"

    def get_range(self,
                   Ns: Iterable[int] = (5000, 40000),
                   start: float | None = None,
                   stop: float | None = None,
                   count: int = 100) -> FloatArray:
        Ns = tuple(Ns)
        min_alpha = round(2 / min(Ns) * count) / count
        max_alpha = round((1 - min_alpha) *count) / count
        start = min_alpha if start is None else start
        stop = max_alpha if stop is None else stop
        if min_alpha > start or max_alpha < stop:
            raise ValueError("Invalid range. Must be inside [2/N, 1-2/N]")
        if start != round(start *count) /count or stop != round(stop * count) / count:
            raise ValueError("Start and stop must be multiples of step")
        filename = f"AlphaRange({Ns}_{(start, stop, count)})"
        try:
            return self.load_data(filename)
        except self.LoadException:
            pass

        if self._modus == "uniform":
            alphas_raw = self._get_raw_alphas_uniform(start, stop, count)
        elif isinstance(self._modus, Harper.IrrationalFrequency):
            alphas_raw = self._get_raw_alphas_alpha0(self._modus, start, stop, count)
        else:
            raise ValueError("Invalid modus")

        doi_values = [min(self.degree_of_irrationality(alpha, N) for N in Ns) for alpha in alphas_raw]
        values = tuple(zip(alphas_raw, doi_values))
        alphas = []
        for i in range(count):
            alpha_min = start + i / count * (stop - start)
            alpha_max = start + (i + 1) / count * (stop - start)
            v = tuple((a, d) for a, d in values if alpha_min <= a < alpha_max)
            alphas.append(max(v, key=lambda x: x[1])[0])
        data = np.array(alphas)

        self.save_data(data, filename)
        return data

    def _get_raw_alphas_uniform(self, start: float, stop: float, count: int):
        return np.linspace(start, stop, count*self._search_factor + 1)

    def _get_raw_alphas_alpha0(self, alpha0: float, start: float, stop: float, count: int):
        n = round(count * self._search_factor * alpha0)
        return np.arange(start, stop, (stop-start) / n * alpha0)

    def degree_of_irrationality(self, alpha: float, N: int, intern: bool = False) -> float:
        alpha = alpha % 1
        if alpha == 0:
            return 0
        if self._with_inverse and not intern:
            logger.debug("Computing degree of irrationality for alpha=%s, N=%s", alpha, N)
            return min(self.degree_of_irrationality(alpha, N, intern=True),
                       self.degree_of_irrationality(1/alpha, N, intern=True))
        values = np.mod(np.arange(N) * alpha, 1)
        dor_modulo_jump = 1 - np.max(values)
        idx = values > .5
        values[idx] = 1 - values[idx]
        values = np.sort(values)
        dor = np.max(np.diff(values))
        dor = max(dor, dor_modulo_jump)
        return 1 - (.5 / N - dor) / (.5 / N - 1)


def beta_symmetric_harper_hamiltonian(alpha: Harper.Frequency, N: int) -> tuple[float, float]:
    beta_0 = (-np.pi * alpha * (N+1)) % np.pi
    return beta_0, beta_0 + np.pi
