
# ========================================
# library
# ========================================
from scipy.optimize import minimize
import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import logging
import sys
from contextlib import contextmanager
import time

# ==================
# Constant
# ==================
ex = "_ensemble"
TRAIN_PATH = "../input/fb3/train_folds.csv"
if not os.path.exists(f"../output/ex/ex{ex}"):
    os.makedirs(f"../output/ex/ex{ex}")

LOGGER_PATH = f"../output/ex/ex{ex}/ex{ex}.txt"

# ===============
# Functions
# ===============
def MCRMSE(y_trues, y_preds):
    scores = []
    idxes = y_trues.shape[1]
    for i in range(idxes):
        y_true = y_trues[:, i]
        y_pred = y_preds[:, i]
        score = mean_squared_error(y_true, y_pred, squared=False)  # RMSE
        scores.append(score)
    mcrmse_score = np.mean(scores)
    return mcrmse_score, scores

def setup_logger(out_file=None, stderr=True, stderr_level=logging.INFO, file_level=logging.DEBUG):
    LOGGER.handlers = []
    LOGGER.setLevel(min(stderr_level, file_level))

    if stderr:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(FORMATTER)
        handler.setLevel(stderr_level)
        LOGGER.addHandler(handler)

    if out_file is not None:
        handler = logging.FileHandler(out_file)
        handler.setFormatter(FORMATTER)
        handler.setLevel(file_level)
        LOGGER.addHandler(handler)

    LOGGER.info("logger set up")
    return LOGGER


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    LOGGER.info(f'[{name}] done in {time.time() - t0:.0f} s')

LOGGER = logging.getLogger()
FORMATTER = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
setup_logger(out_file=LOGGER_PATH)

# ================================
# Main
# ================================
data_dict = pd.read_csv(TRAIN_PATH)
y_cols = ['cohesion', 'syntax', 'vocabulary',
          'phraseology', 'grammar', 'conventions']
y = data_dict[y_cols].values
fold_array = data_dict['fold'].values

ex008 = np.load("../output/ex/ex008/ex008_oof.npy")
ex023 = np.load("../output/ex/ex023/ex023_oof.npy")
ex028 = np.load("../output/ex/ex028/ex028_oof.npy")
# ex037 = np.load("../output/ex/ex037/ex037_oof.npy")
# ex038 = np.load("../output/ex/ex038/ex038_oof.npy")
ex044 = np.load("../output/ex/ex044/ex044_oof.npy")
ex051 = np.load("../output/ex/ex051/ex051_oof.npy")
def f(x):
    pred1 = ex008 * x[0] + ex023 * x[1] + ex028 * x[2] +  ex044 * x[3] + ex051 * x[4]
    score = MCRMSE(y, pred1)
    return score[0]


with timer("ensemble"):
    weight_init = [1 / 5 for _ in range(5)]
    result = minimize(f, weight_init, method="Nelder-Mead")
    final_x = list(result.x)
    final_cv = f(result.x)
    LOGGER.info(f'ensemble_weight:{final_x}')
    LOGGER.info(f'ensemble cv score: {final_cv}')