import re
from typing import Generator, Literal, Any

import numpy as np

from chain_model import ChainModel
from harper import Harper, AlphaGetter
from utils import param_phase_validator, FloatArray


# this should be a ParamsRange2D in the future
ParamsArray2D = np.ndarray  # check: ignore


class ParamsRange:

    def _param_value_validator(self, value: float) -> float:
        factor = (np.pi / 4) if self.param == "beta" else 1
        return factor * param_phase_validator(value / factor, modulo=False)

    def __init__(self, param_name: Literal["phi_H", "phi_hop", "beta", "alpha"],
                 param_range_inclusive: tuple[float, float, float] | tuple[float | None, float | None, int] | tuple[float, float, float, Harper.IrrationalFrequency],
                 alpha_getter_kwargs: dict[str, Any] | None = None,
                 **kwargs):

        alpha_getter_kwargs = alpha_getter_kwargs or {}
        if param_name not in ("phi_H", "phi_hop", "beta", "alpha"):
            raise ValueError(f"Unknown range parameter {param_name}")
        if param_name in kwargs:
            raise ValueError(f"Parameter {param_name} is set twice")
        self.param = param_name
        self._factor = param_range_inclusive[3] if len(param_range_inclusive) == 4 else \
            (np.pi / 4) if param_name == "beta" else 1
        self._float_range: tuple[float, float, float] = .0, .0, .0
        self._int_range: tuple[float | None, float | None, int] = None, None, 0
        self._alpha_getter = AlphaGetter()
        if self.param == "alpha" and isinstance(param_range_inclusive[2], int):
            start = None if param_range_inclusive[0] is None else self._param_value_validator(param_range_inclusive[0])
            stop = None if param_range_inclusive[1] is None else self._param_value_validator(param_range_inclusive[1])
            count = param_range_inclusive[2]
            self._int_range = start, stop, count
            self._alpha_getter = AlphaGetter(**alpha_getter_kwargs)
        else:
            pri: list[float] = []
            for p in param_range_inclusive[:3]:
                if not isinstance(p, (float, int)):
                    raise ValueError("Invalid range: %s" % str(param_range_inclusive))
                pri.append(self._param_value_validator(p))
            self._float_range = start, stop, step = pri[0], pri[1], pri[2]
            if stop <= start \
                    or self._param_value_validator(self._param_value_validator(stop - start) % step) % step != 0:
                raise ValueError("Invalid range: %s" % str(param_range_inclusive))
        self._kwargs = kwargs
        try:
            self._params = tuple(self._get_params())
        except ValueError as e:
            raise ValueError("Invalid value in ParamsRange initialization") from e

    @property
    def param_values(self) -> FloatArray:
        if self.param in Harper.model_fields:
            return np.array([getattr(p.harper, self.param) for p in self._params])
        return np.array([getattr(p, self.param) for p in self._params])

    def __len__(self):
        return len(self._params)

    def __getitem__(self, item):
        return self._params[item]

    def __iter__(self):
        return iter(self._params)

    def _get_values(self) -> Generator[float, None, None]:
        if self.param == "alpha" and self._int_range[-1]:
            start, stop, count = self._int_range
            for alpha in self._alpha_getter.get_range(start=start, stop=stop, count=count):
                yield alpha
        else:
            factor = float(self._factor) if isinstance(self._factor, Harper.IrrationalFrequency) else 1
            start, stop, step = self._float_range
            current = start
            while current <= stop:
                yield current * factor
                if current == stop:
                    break
                current = self._param_value_validator(current + step)

    def _get_params(self) -> Generator[ChainModel.Params, None, None]:
        for current in self._get_values():
            kwargs_harper = {k: v for k, v in self._kwargs.items() if k in Harper.model_fields}
            kwargs = {k: v for k, v in self._kwargs.items() if k not in Harper.model_fields}
            if self.param in Harper.model_fields:
                kwargs_harper[self.param] = current
            else:
                kwargs[self.param] = current
            if kwargs.get("harper") is not None and kwargs_harper:
                raise ValueError("harper parameter is set twice")
            try:
                harper_from_args = Harper(**kwargs_harper)
            except ValueError:
                harper_from_args = None
            kwargs["harper"] = kwargs.get("harper") or harper_from_args
            yield ChainModel.Params(**kwargs)

    def __repr__(self) -> str:
        params_range_repr: tuple
        if self.param == "alpha":
            params_range_repr = self._int_range
        else:
            def _(value: float) -> str:
                return f"{param_phase_validator(value/np.pi, modulo=False)}pi" if self.param == "beta" else f"{value}"
            params_range_repr = tuple(_(v) for v in self._float_range)
            if isinstance(self._factor, Harper.IrrationalFrequency):
                params_range_repr += (self._factor,)
        example_idx = 1 if self.param == "phi_H" and self._float_range[0] == 0 else 0
        s = f"ParamsRange({self.param}, {params_range_repr}, {self[example_idx]})"
        return re.sub(r", N=\d+ M=\d+ ", ", ", s)
