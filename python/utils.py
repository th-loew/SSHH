from __future__ import annotations

import functools
import logging
import os
from multiprocessing import Pool, cpu_count
from typing import Annotated, Callable, Iterable, overload

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field, AfterValidator


logger = logging.getLogger("Masterarbeit")
log_level = os.getenv("LOG_LEVEL", "WARNING")
logger.setLevel(getattr(logging, log_level))
console_handler = logging.StreamHandler()
console_handler.setLevel(log_level)
logger.addHandler(console_handler)

result_logger = logging.getLogger('Masterarbeit Results')
result_logger.setLevel(log_level)
file_handler = logging.FileHandler('results.log')
file_handler.setLevel(log_level)
result_logger.addHandler(file_handler)


IntArray = NDArray[np.signedinteger]  # check: ignore
FloatArray = NDArray[np.float64]  # check: ignore
ComplexArray = NDArray[np.complex128]  # check: ignore


def compute_parallel(fun: Callable, args: Iterable, **kwargs) -> list:

    nodes = os.getenv("NODES", None)

    if nodes is None:
        n_logical_cores = cpu_count()  # includes hyperthreading
        n_physical_cores = n_logical_cores // 2
        n_pool = max(1, n_physical_cores - 1)  # leave one core for the main process
    else:
        n_pool = int(nodes) - 1

    args = list(args)
    with Pool(n_pool) as pool:
        return pool.map(functools.partial(fun, **kwargs), args)

def format_float(value: float) -> str:
    s = f"{value:.2e}"
    a, b = s.split("e")
    exp = int(b)
    if exp == 0:
        return a
    return f"{a} \\cdot 10^{{{exp}}}"


@overload
def param_phase_validator(phi: float, modulo: bool = True) -> float: ...
@overload
def param_phase_validator(phi: None, modulo: bool = True) -> None: ...
def param_phase_validator(phi: float | None, modulo: bool = True) -> float | None:
    if phi is None:
        return None
    if modulo:
        phi = phi % 8
        if phi > 4:
            phi -= 8
    digits = 6
    phi_rounded = round(phi, digits)
    assert abs(phi_rounded - phi) < 10 ** (-2 * digits), f"Rounding error: {phi} -> {phi_rounded}"
    return phi_rounded

ParamPhase = Annotated[float | None, AfterValidator(param_phase_validator)]

def param_phase_to_values(phi: float) -> tuple[float, float]:
    phi = phi % 8  # nominal values in [-4, 4)
    if phi == int(phi):
        r2 = float(np.sqrt(1 / 2))
        return [
            (0.0, 1.0), (r2, r2), (1.0, 0.0), (r2, -r2),
            (0.0, -1.0), (-r2, -r2), (-1.0, 0.0), (-r2, r2)
        ][int(phi)]
    else:
        phi *= np.pi / 4
        return float(np.sin(phi)), float(np.cos(phi))


class Trap(BaseModel):
    def __bool__(self):
        raise NotImplementedError()

    def __eq__(self, other):
        raise NotImplementedError()


class ParabolicTrap(Trap):
    depth: float = Field(0, ge=0)
    offset_relative: float = Field(0, ge=-1, le=1)

    def __bool__(self):
        return self.depth > 0

    def __eq__(self, other):
        if not other:
            return not self
        if other.__class__ is self.__class__:
            return self.depth == other.depth and self.offset_relative == other.offset_relative
        logger.warning("Comparing ParabolicTrap potential with %s is not recommended.",
                       other.__class__.__name__)
        return False
