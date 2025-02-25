import json
from multiprocessing import Pool
import time

from data_store import DataStore
from harper import Harper
from localization_measure import LocalizationMeasure
from params_range import ParamsRange
from utils import logger

DataStore.USE_STORE = False

data: dict[int, dict[int, dict[str, int | float]]] = {}

alpha = list(Harper.IrrationalFrequency)[-2]

for N in (50, 100, 1000, 2000):

    data[N] = {}

    params = ParamsRange("phi_H", (0, 2, .2), N=N, alpha=alpha)

    for n_pool in range(1, 13):

        t0 = time.time()
        with Pool(n_pool) as pool:
            pool.map(LocalizationMeasure.measure_system_all_methods, params)
        t1 = time.time()

        data[N][n_pool] = {
            'time': t1 - t0,
            'number_of_systems': len(params),
            'number_of_methods': len(LocalizationMeasure.METHODS),
            'time_per_system': (t1 - t0) / len(params),
        }

        with open('time_data.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)

        logger.info('N=%d, n_pool=%d, time=%.2f s', N, n_pool, t1 - t0)
