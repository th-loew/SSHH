from __future__ import annotations

from enum import Enum
from typing import Any, overload, Literal

import numpy as np
import scipy
from pydantic import BaseModel, Field, model_validator

from data_store import DataStore
from utils import logger, Trap, param_phase_to_values, ParamPhase, IntArray, FloatArray, ComplexArray
from harper import Harper

class Localizations(Enum):
    LATTICE_SITE = "LatticeSite"
    UNIT_CELL = "UnitCell"

    def __repr__(self):
        return self.name


# noinspection PyPep8Naming
class ChainModel(DataStore):
    SUBDIRS = "eigvecs", "eigvals"

    def load_eigvals(self, naming: Params) -> FloatArray:
        return self.load_data(naming, subdir="eigvals")  # type: ignore[return-value]

    def load_eigvecs(self, naming: Params) -> ComplexArray:
        return self.load_data(naming, subdir="eigvecs")  # type: ignore[return-value]


    class Params(BaseModel):
        N: int = Field(None, ge=4, multiple_of=2, description="Number of lattice sites")  # type: ignore[assignment]
        M: int = Field(None, ge=2, description="Number of unit cells")  # type: ignore[assignment]
        phi_hop: ParamPhase = Field(None, description="Coupling phase (in units of pi/4)")
        t_intra: float = Field(None, description="intra-cell hopping amplitude")  # type: ignore[assignment]
        t_inter: float = Field(None, description="inter-cell hopping amplitude")  # type: ignore[assignment]
        localization: Localizations = Field(Localizations.LATTICE_SITE, description="Localization method")
        harper: Harper | None = Field(None, description="Harper quasiperiodic potential")
        b: float = Field(0, ge=0, le=1, description="bulkiness")
        trap: Trap | None = Field(None, description="Trap potential")
        nnn_factor: float = Field(0, description="Next-nearest-neighbor coupling factor")

        @property
        def harper_not_none(self) -> Harper:
            if self.harper is None:
                raise ValueError("Harper potential is not specified")
            return self.harper

        def copy_with_changed_N(self, N: int) -> ChainModel.Params:
            if N < 4 or N % 2:
                raise ValueError(f"N={N} is not a valid number of lattice sites")
            return self.model_copy(update={"N": N, "M": N // 2})



        def __eq__(self, other):
            if isinstance(other, self.__class__):
                return (self.M == other.M and self.t_intra == other.t_intra and self.t_inter == other.t_inter
                        and self.b == other.b and self.trap == other.trap and self.harper == other.harper
                        and self.localization == other.localization and self.nnn_factor == other.nnn_factor)
            logger.warning("Comparing %s with %s is not recommended.",
                           self.__class__.__name__, other.__class__.__name__)
            return False

        @model_validator(mode='before')
        @classmethod
        def check_deprecated_fields(cls, data: Any) -> Any:
            if isinstance(data, dict):
                assert "alpha" not in data, "alpha is deprecated, use phi_hop instead"
                assert "phi" not in data, "phi is deprecated, use phi_hop instead"
                assert "varphi" not in data, "varphi is deprecated, use phi_hop instead"
                assert "v" not in data, "v is deprecated, use t_intra instead"
                assert "w" not in data, "w is deprecated, use t_inter instead"
            return data

        @model_validator(mode='before')
        @classmethod
        def check_wrong_fields(cls, data: Any) -> Any:
            if isinstance(data, dict):
                assert "phi_H" not in data,\
                    f"phi_H is not a valid parameter for {cls.__name__}, did you mean phi_hop?"
            return data

        @model_validator(mode='before')
        @classmethod
        def check_fields_defined(cls, data: Any) -> Any:

            if isinstance(data, dict):

                assert "M" in data or "N" in data, "Either M or N must be specified"
                assert not ("M" in data and "N" in data), "Either M or N must be specified, not both"

                if "phi_hop" not in data and "t_intra" not in data and "t_inter" not in data:
                    data["phi_hop"] = 1
                elif "phi_hop" not in data:
                    assert "t_intra" in data and "t_inter" in data,\
                        "If phi_hop is not specified, both t_intra and t_inter must be specified"
                elif "phi_hop" in data:
                    assert "t_intra" not in data and "t_inter" not in data,\
                        "If phi_hop is specified, neither t_intra nor t_inter must be specified"

                harper: Harper | None = data.get("harper")
                if harper is not None:
                    if harper.phi_H is not None:
                        assert "phi_hop" in data,\
                            "If harper is specified and phi_H is specified, phi_hop must be specified"
                    else:
                        assert "phi_hop" not in data,\
                            "If harper is specified and phi_H is not specified, phi_hop must not be specified"

                if "nnn_factor" in data:
                    assert data.get("phi_hop") == 1,\
                        "Next-nearest-neighbor coupling is only supported for t_inter == t_intra, i.e. phi_hop=1"

                if "b" in data:
                    assert "nnn_factor" not in data and "trap" not in data and "harper" not in data,\
                        ("Bulkiness is not supported in combination with next-nearest-neighbor coupling, "
                         "trap potential or Harper potential")

            return data

        @model_validator(mode='after')
        def compute_M_N(self) -> ChainModel.Params:
            if self.N is None:
                self.N = 2 * self.M
            else:
                self.M = self.N // 2
            return self

        @model_validator(mode='after')
        def compute_hopping_amplitudes(self) -> ChainModel.Params:

            if self.phi_hop is not None:
                self.t_inter, self.t_intra = param_phase_to_values(self.phi_hop)
                if isinstance(self.harper, Harper):
                    factor = self.harper.hopping_amplitude_factor
                    if self.phi_hop == self.harper.phi_H == 1:
                        self.t_inter = self.t_intra = .5
                    elif factor is not None:
                        self.t_intra, self.t_inter = factor * self.t_intra, factor * self.t_inter


            return self

    def __init__(self, params: Params, **kwargs):
        self._params = params
        super().__init__(**kwargs)

    @property
    def params(self) -> Params:
        return self._params

    @property
    def name(self) -> Literal["plain lattice", "Harper", "SSH", "SSH-Harper"]:
        p = self.params
        if p.b or p.trap or p.nnn_factor:
            raise NotImplementedError("Unknown name for the model")
        if not p.harper and p.phi_hop == 1:
            return "plain lattice"
        if self.params.phi_hop == 1:
            return "Harper"
        if not self.params.harper:
            return "SSH"
        return "SSH-Harper"

    @property
    def sites(self) -> IntArray:
        if self.params.localization == Localizations.LATTICE_SITE:
            return 1 + np.arange(self.params.N)
        if self.params.localization == Localizations.UNIT_CELL:
            return 1 + np.arange(self.params.N) // 2
        raise NotImplementedError(f"Localization method {self.params.localization} is not supported")

    @property
    def H_diag0(self) -> FloatArray:
        x = self.sites
        diag0 = np.zeros(self.params.N, dtype=np.float64)
        if self.params.trap:
            raise NotImplementedError(f"Trap potential {self.params.trap.__class__.__name__} is not supported")
        if self.params.harper:
            harper = self.params.harper
            diag0 += harper.amplitude * np.cos(2 * np.pi * harper.alpha * x + harper.beta)
        return diag0

    @property
    def H_diag1(self) -> FloatArray:
        diag1 = self.params.t_inter * np.ones(self.params.N - 1, dtype=np.float64)
        diag1[::2] = self.params.t_intra
        return diag1

    @property
    def H_diag2(self) -> FloatArray:
        if not self.params.nnn_factor:
            return np.zeros(self.params.N - 2, dtype=np.float64)
        assert self.params.t_intra == self.params.t_inter
        return self.params.t_intra * self.params.nnn_factor * np.ones(self.params.N - 2, dtype=np.float64)

    @property
    def H(self) -> FloatArray:
        H = scipy.sparse.diags([self.H_diag0, self.H_diag1, self.H_diag1, self.H_diag2, self.H_diag2],
                               [0, 1, -1, 2, -2], dtype=np.float64).toarray()
        H[0, -1] = H[-1, 0] = self.params.b * self.params.t_inter
        return H


    def _eigh(self, eigvals_only=False) -> FloatArray | tuple[FloatArray, ComplexArray]:
        try:
            return scipy.linalg.eigh_tridiagonal(self.H_diag0, self.H_diag1, eigvals_only=eigvals_only)
        except scipy.linalg.LinAlgError:
            return scipy.linalg.eigh_tridiagonal(self.H_diag0, self.H_diag1, eigvals_only=eigvals_only,
                                                 lapack_driver="stev")


    @overload
    def eigh(self) -> FloatArray: ...
    @overload
    def eigh(self, eigvals_only: Literal[True]) -> FloatArray: ...
    @overload
    def eigh(self, eigvals_only: Literal[False]) -> tuple[FloatArray, ComplexArray]: ...
    @overload
    def eigh(self, eigvals_only: bool) -> FloatArray | tuple[FloatArray, ComplexArray]: ...
    def eigh(self, eigvals_only=False) -> FloatArray | tuple[FloatArray, ComplexArray]:
        res: FloatArray | tuple[FloatArray, ComplexArray]
        try:
            vals = self.load_eigvals(self.params)
            res = vals if eigvals_only else (vals, self.load_eigvecs(self.params))
        except self.LoadException:
            if self.params.b or any(self.H_diag2):
                res = scipy.linalg.eigh(self.H, eigvals_only=eigvals_only)
            elif any(self.H_diag0):
                res = self._eigh(eigvals_only=eigvals_only)
            else:
                M = self.params.M
                tol = np.inf
                tol_step = 1e-5
                res = self._eigh(eigvals_only=eigvals_only)
                eigvals: FloatArray = res[0] if isinstance(res, tuple) else res
                select_range = np.mean(eigvals[M-2:M]), np.mean(eigvals[M:M+2])
                try:
                    while tol > np.min(np.abs(eigvals)) * tol_step:
                        logger.debug("tol_iteration tol: %s", "inf" if tol == np.inf else f"{tol:.2e}")
                        tol = np.min(np.abs(eigvals[M-1:M+1])) * tol_step
                        res_subset = scipy.linalg.eigh_tridiagonal(self.H_diag0, self.H_diag1, eigvals_only=eigvals_only,
                                                                   tol=tol, lapack_driver="stebz",
                                                                   select="v", select_range=select_range)
                        if isinstance(res, tuple):
                            res[0][M-1:M+1] = res_subset[0]
                            res[1][:, M-1:M+1] = res_subset[1]
                        else:
                            res[M-1:M+1] = res_subset
                except ValueError:
                    pass
            if isinstance(res, tuple):
                self.save_data(res[0], self.params, subdir="eigvals")
                self.save_data(res[1], self.params, subdir="eigvecs")
            else:
                self.save_data(res, self.params, subdir="eigvals")
        return res
