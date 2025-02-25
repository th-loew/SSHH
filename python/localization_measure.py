from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Iterable

import numpy as np

from chain_model import ChainModel
from data_store import FloatDataStore
from utils import ComplexArray, FloatArray, compute_parallel


class _Method(ABC):

    @property
    def latex_name(self) -> str:
        return r"$L_\mathrm{" + self.name + r"}$"

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def measure(self, state: ComplexArray) -> FloatArray:
        pass

    @abstractmethod
    def theoretical_value(self, lamda: float, N: int | None = None, strict: bool = True) -> float:
        pass


class Ipr(_Method):

    @property
    def name(self) -> str:
        return "IPR"

    def measure(self, state: ComplexArray) -> FloatArray:
        return np.sum(np.abs(state)**4, axis=0)

    def theoretical_value(self, lamda: float, N: int | None = None, strict: bool = True) -> float:
        if lamda == 0:
            return 1
        if lamda == np.inf:
            return 1/N if N is not None else 0
        q = np.exp(-2 * lamda**(-1))
        if isinstance(N, int):
            if strict:
                assert N % 2 == 1
            q_K = q**((N+1)/2)
        else:
            q_K = 0
        return (1 - q) / (1+q) * (1+q**2-2*q_K**2) / (1 + q - 2*q_K)**2


class Std(_Method):

    @property
    def name(self) -> str:
        return "Std"

    def measure(self, state: ComplexArray) -> FloatArray:
        # each column one state
        N = state.shape[0]
        l = np.arange(N) + 1
        alpha_square = np.abs(state)**2
        S2 = np.dot(l**2, alpha_square)
        S1 = np.dot(l, alpha_square)
        sigma = np.sqrt(12 / (N**2 - 1) * (S2 - S1**2))
        sigma[sigma > 1] = 1
        return 1 - sigma

    def theoretical_value(self, lamda: float, N: int | None = None, strict: bool = True) -> float:
        if lamda == np.inf:
            return 0
        if lamda == 0:
            return 1
        if N is None or N == np.inf:
            return 1
        if strict:
            assert N % 2 == 1
        K = (N+1) / 2
        q = np.exp(-2 * lamda**(-1))
        f = (K-1)**2 * q**(K+2) + (-2*K**2+2*K+1) * q**(K+1) + K**2 * q**K - q**2 - q
        return float(
            1 - np.sqrt(-12 * 2 / (N**2-1) * f / (1+q-2*q**K)) / (1-q)
        )


class Bin(_Method):

    c = 1e-2
    C = .5

    @property
    def name(self) -> str:
        return "bin"

    def measure(self, state: ComplexArray) -> FloatArray:
        count = np.sum(np.abs(state) < self.c * np.max(np.abs(state), axis=0), axis=0)
        return (count / len(state) > self.C).astype(np.float64)

    def theoretical_value(self, lamda: float, N: int | None = None, strict: bool = True) -> float:
        if lamda == np.inf:
            return 0
        if lamda == 0:
            return 1
        if N is None or N == np.inf:
            return 1
        return float(lamda < N*(self.C-1)/np.log(self.c))


class EdgeMethod(_Method):

    @property
    def name(self) -> str:
        return "edge"

    def measure(self, state: ComplexArray) -> FloatArray:
        # each column one state
        N = state.shape[0]
        l = np.arange(N) + 1
        alpha_square = np.abs(state)**2
        S1 = np.dot((l-1)**2, alpha_square)
        SN = np.dot((l-N)**2, alpha_square)
        sigma1 = np.sqrt(S1) / (N-1)
        sigmaN = np.sqrt(SN) / (N-1)
        sigma1[sigma1 > 1] = 1
        sigmaN[sigmaN > 1] = 1
        return sigma1 - sigmaN

    def theoretical_value(self, lamda: float, N: int | None = None, strict: bool = True) -> float:
        raise NotImplementedError("EdgeMethod does not have a theoretical value")


class LocalizationMeasure(FloatDataStore):

    class Method(Enum):
        IPR = "Ipr"
        STD = "Std"
        BIN = "Bin"
        EDGE = "Edge"

        @property
        def instance(self) -> _Method:
            if self.value == "Ipr":
                return Ipr()
            if self.value == "Std":
                return Std()
            if self.value == "Bin":
                return Bin()
            if self.value == "Edge":
                return EdgeMethod()
            raise ValueError(f"Unknown method {self.value}")
        
    METHODS = Method.IPR, Method.STD, Method.BIN
    EDGE_METHOD = Method.EDGE

    def __init__(self, method: Method, **kwargs):
        self._method = method
        self._method_instance: _Method = method.instance
        super().__init__(**kwargs)

    @property
    def method(self) -> Method:
        return self._method

    def _get_subsubdir(self):
        return self._method_instance.name

    def measure_one(self, state: ComplexArray) -> float:
        assert state.ndim == 1
        N = len(state)
        state = state.reshape((N, 1))
        loc = self._method_instance.measure(state)
        assert loc.shape == (1, )
        return float(loc[0])

    def measure_all(self, states: ComplexArray | ChainModel.Params) -> FloatArray:
        params = None
        if isinstance(states, ChainModel.Params):
            params = states
            try:
                return self.load_data(params)
            except self.LoadException:
                pass
            state_array = ChainModel(params).eigh(eigvals_only=False)[1]
        else: state_array = states
        assert state_array.ndim == 2
        loc = self._method_instance.measure(state_array)
        if params is not None:
            self.save_data(loc, params)
        return loc

    @staticmethod
    def _state_measure_to_system_measure(state_data: FloatArray, modus: str) -> float:
        try:
            return getattr(state_data, modus)()
        except AttributeError as e:
            raise ValueError(f"Unknown modus {modus}") from e

    def measure_system(self, params: ChainModel.Params, modus: str = 'mean') -> float:
        state_data = self.measure_all(params)
        return self._state_measure_to_system_measure(state_data, modus)

    def measure_systems(self, params: Iterable[ChainModel.Params], modus: str = 'mean') -> FloatArray:
        return np.array(compute_parallel(self.measure_system, params, modus=modus))

    @classmethod
    def measure_system_all_methods(cls, params: ChainModel.Params, modus: str = 'mean') -> \
            dict[Method, float]:
        instances = {m: cls(m) for m in cls.METHODS}
        values: dict[LocalizationMeasure.Method, float] =  {}
        methods: list[LocalizationMeasure.Method] = []
        for m, instance in instances.items():
            try:
                state_data = instance.load_data(params)
            except instance.LoadException:
                methods.append(m)
            else:
                values[m] = cls._state_measure_to_system_measure(state_data, modus)
        if methods:
            _, states = ChainModel(params).eigh(eigvals_only=False)
            for m in methods:
                instance = instances[m]
                state_data = instance.measure_all(states)
                instance.save_data(state_data, params)
                values[m] = cls._state_measure_to_system_measure(state_data, modus)
        return values


    @classmethod
    def measure_systems_all_methods(cls, params: Iterable[ChainModel.Params], modus: str = 'mean') -> \
            dict[Method, FloatArray]:
        data: list[dict[LocalizationMeasure.Method, float]] = compute_parallel(cls.measure_system_all_methods, params, modus=modus)
        return {m: np.array([d[m] for d in data]) for m in cls.METHODS}


    def theoretical_value(self, lamda: float | FloatArray, N: int | None = None,
                          strict: bool | None = None) -> FloatArray:
        if strict is None:
            fun = lambda l: self._method_instance.theoretical_value(l, N)
        else:
            fun = lambda l: self._method_instance.theoretical_value(l, N, strict=strict)
        return np.array([fun(l) for l in np.atleast_1d(lamda)])
