import os
from pathlib import Path

import numpy as np

from chain_model import ChainModel
from params_range import ParamsRange

CALCULATIONS_ONLY = os.getenv("GENERATE_TEST_DATA_CALCULATIONS_ONLY", "0") == "1"
N = 40000

directory = Path(__file__).parent.parent / "test" / "res" / "ChainModel" / "eigvals"


for phi_H in (.5, 1.5):
    for phi_hop in (.5, 1, 1.5):
        for p in ParamsRange("beta", (0, 2 * np.pi, np.pi / 5), N=N, phi_H=phi_H, phi_hop=phi_hop):
            eigvals = ChainModel(p).eigh(eigvals_only=True)
            if not CALCULATIONS_ONLY:
                directory.mkdir(parents=True, exist_ok=True)
                np.save(directory / f"{p}.npy", eigvals)
