#! /usr/local/bin/python3

import pandas as pd
import random
import os
import sys
import numpy as np
import logging
import pickle
from tqdm import tqdm
from functools import partial
from matplotlib import pyplot as plt
import multiprocessing
import joblib
from timlinucb import (
    generate_node2vec_fetures,
    timlinucb,
    timlinucb_parallel_oim,
    timlinucb_parallel_t,
)
from helpers import tim_t, tqdm_joblib, tim_t_parallel, run_algorithm
from rsb import rsb2

# --------------------------------------------------------------------------------------
# %% ------------------------------ Initial setup --------------------------------------
# --------------------------------------------------------------------------------------


# Fancy plots in matplotlib and pandas

random.seed(42)
plt.style.use("ggplot")
pd.options.mode.chained_assignment = None  # default='warn'
NUMEXPR_MAX_THREADS = 8  # For RSB

# Setting up logging
VERBOSE = True
LOGGING_FMT = (
    "%(levelname)s | %(asctime)s | line %(lineno)s | %(funcName)s | %(message)s"
)
LOGGING_DATEFMT = "%H:%M:%S"

logging_conf = partial(
    logging.basicConfig, format=LOGGING_FMT, datefmt=LOGGING_DATEFMT, stream=sys.stdout
)

if VERBOSE:
    logging_conf(level=logging.DEBUG)
else:
    logging_conf(level=logging.INFO)


# --------------------------------------------------------------------------------------
# %% ------------------------------ Dataset setup --------------------------------------
# --------------------------------------------------------------------------------------

DATASET_DIR = os.path.join("..", "Datasets")
if not os.path.exists(DATASET_DIR):
    print("Can't find the dataset directory!")
DATASET_FACEBOOK = os.path.join(
    DATASET_DIR, "fb-wosn-friends", "fb-wosn-friends-clean.edges"
)
DATASET_DIGG = os.path.join(DATASET_DIR, "digg-friends", "digg-friends.edges")
DATASET_SCHOOL = os.path.join(
    DATASET_DIR, "ia-primary-school-proximity", "ia-primary-school-proximity.edges"
)
DATASET_PHONES = os.path.join(DATASET_DIR, "mit", "out.mit")

NUMBER_OF_DAYS = 20
# NUMBER_OF_DAYS = "ALL"
NUM_FEATURES_NODE2VEC = 20
NUM_SEEDS_TO_FIND = 5


# --------------------------------------------------------------------------------------
# %% ------------------------- Preparing the Digg friends dataset ----------------------
# --------------------------------------------------------------------------------------

logging.debug("Processing the Digg dataset...")
logging.debug("Getting the graph...")
df_digg = pd.read_csv(
    DATASET_DIGG, sep=" ", names=["source", "target", "timestamp"], skiprows=1,
)

# Processing the dataframe
logging.debug("Splitting the graph into time steps...")
df_digg["day"] = pd.to_datetime(df_digg["timestamp"], unit="s").dt.floor("d")
df_digg = df_digg.sort_values(by=["timestamp", "source", "target"])
df_digg = df_digg[["source", "target", "day"]]
df_digg_nodes = np.sort(np.unique(np.hstack((df_digg["source"], df_digg["target"]))))

# Getting the true weights
logging.debug('Generating "true" activation probabilities...')
df_digg["probab"] = np.random.uniform(0, 0.1, size=df_digg.shape[0])
# df_digg['probab'] = np.random.uniform(0, 1, size=df_digg.shape[0])
if NUMBER_OF_DAYS == "ALL":
    df_digg_times = np.sort(np.unique(df_digg["day"]))[0:]
else:
    df_digg_times = np.sort(np.unique(df_digg["day"]))[0:NUMBER_OF_DAYS]

# TIMLinUCB only
logging.debug("Generating node2vec features...")
df_digg_feats = generate_node2vec_fetures(
    df_digg, dataset_name="digg", num_features=NUM_FEATURES_NODE2VEC
)

DATASET = df_digg
DATASET_FEATS = df_digg_feats
DATASET_TIMES = df_digg_times
DATASET_NODES = df_digg_nodes


# --------------------------------------------------------------------------------------
# %% ----------------------- Comparing TIMLinUCB w/ RSB and t_TIM  ---------------------
# ----------- Details: 20 seeds, start->30 days ----------------------------------------

logging.debug("Comparing TIMLinUCB with t_TIM and RSB - Digg dataset (30 days)")
NUMBER_OF_DAYS = 30


DATASET = df_digg
DATASET_FEATS = df_digg_feats
DATASET_TIMES = np.sort(np.unique(DATASET["day"]))[0:NUMBER_OF_DAYS]
NUM_SEEDS_TO_FIND = 20

# RSB parameters
OPTIMAL_C_RSB = 10
OPTIMAL_GAMMA_RSB = 0.4
NUM_REPEATS_EXPECT_RSB = 20

# TIMLinUCB parameters
OPTIMAL_EPS_TLU = 0.5
OPTIMAL_SIGMA_TLU = 1
OPTIMAL_C_TLU = 1
OPTIMAL_NUM_REPEATS_OIM_TLU = 20
OPTIMAL_NUM_REPEATS_REW_TLU = 20

# TIM_T parameters
OPTIMAL_EPS_TIM_T = 0.5
OPTIMAL_NUM_REPEATS_REW_TIM_T = 20

setup_array = []

# Setting up TIM_t (offline algorithm)
setup_array.append(
    {
        "algo_name": "tim_t",
        "function": tim_t_parallel,
        "args": [DATASET, DATASET_NODES, DATASET_TIMES],
        "kwargs": {
            "num_seeds": NUM_SEEDS_TO_FIND,
            "num_repeats_reward": OPTIMAL_NUM_REPEATS_REW_TIM_T,
            "epsilon": OPTIMAL_EPS_TIM_T,
        },
    }
)

# Setting up RSB with persistent features
setup_array.append(
    {
        "algo_name": "rsb_persist",
        "function": rsb2,
        "args": [DATASET, DATASET_NODES, DATASET_TIMES],
        "kwargs": {
            "num_seeds": NUM_SEEDS_TO_FIND,
            "C": OPTIMAL_C_RSB,
            "gamma": OPTIMAL_GAMMA_RSB,
            "num_repeats_expect": NUM_REPEATS_EXPECT_RSB,
            "persist_params": True,
            "hide_tqdm": True,
        },
    }
)

# Setting up RSB without persistent features
setup_array.append(
    {
        "algo_name": "rsb_nopersist",
        "function": rsb2,
        "args": [DATASET, DATASET_NODES, DATASET_TIMES],
        "kwargs": {
            "num_seeds": NUM_SEEDS_TO_FIND,
            "C": OPTIMAL_C_RSB,
            "gamma": OPTIMAL_GAMMA_RSB,
            "num_repeats_expect": NUM_REPEATS_EXPECT_RSB,
            "persist_params": False,
            "hide_tqdm": True,
        },
    }
)

# Setting up TIMLinUCB
setup_array.append(
    {
        "algo_name": "timlinucb_nopersist",
        "function": timlinucb_parallel_oim,
        "args": [DATASET, DATASET_FEATS, DATASET_TIMES, DATASET_NODES],
        "kwargs": {
            "num_seeds": NUM_SEEDS_TO_FIND,
            "num_repeats_oim": OPTIMAL_NUM_REPEATS_OIM_TLU,
            "num_repeats_oim_reward": OPTIMAL_NUM_REPEATS_REW_TLU,
            "sigma": OPTIMAL_SIGMA_TLU,
            "c": OPTIMAL_C_TLU,
            "epsilon": OPTIMAL_EPS_TLU,
            "hide_tqdm": True,
            "process_id": "digg1",
            "max_jobs": -5,
        },
    }
)

# Setting up TIMLinUCB (persistent parameters)
setup_array.append(
    {
        "algo_name": "timlinucb_persist",
        "function": timlinucb_parallel_t,
        "args": [DATASET, DATASET_FEATS, DATASET_TIMES, DATASET_NODES],
        "kwargs": {
            "num_seeds": NUM_SEEDS_TO_FIND,
            "num_repeats_oim": OPTIMAL_NUM_REPEATS_OIM_TLU,
            "num_repeats_oim_reward": OPTIMAL_NUM_REPEATS_REW_TLU,
            "sigma": OPTIMAL_SIGMA_TLU,
            "c": OPTIMAL_C_TLU,
            "epsilon": OPTIMAL_EPS_TLU,
            "persist": True,
            "process_id": "digg2",
        },
    }
)


with tqdm_joblib(tqdm(desc="Algorithm comparison (start)", total=len(setup_array))):
    results_array = joblib.Parallel(n_jobs=len(setup_array))(
        joblib.delayed(run_algorithm)(setup_dict) for setup_dict in setup_array
    )


with open("comparison_digg_20s_30t_start_1.pickle", "wb") as f:
    pickle.dump(results_array, f)

# --------------------------------------------------------------------------------------
# %% ----------------------- Comparing TIMLinUCB w/ RSB and t_TIM  ---------------------
# --------------------------- Details: 20 seeds, 30 days->end --------------------------

logging.debug("Comparing TIMLinUCB with t_TIM and RSB - Digg dataset (30 days)")
NUMBER_OF_DAYS = 30
DATASET_TIMES = np.sort(np.unique(DATASET["day"]))[0:NUMBER_OF_DAYS]
NUM_SEEDS_TO_FIND = 20

# RSB parameters
OPTIMAL_C_RSB = 10
OPTIMAL_GAMMA_RSB = 0.4
NUM_REPEATS_EXPECT_RSB = 20

# TIMLinUCB parameters
OPTIMAL_EPS_TLU = 0.5
OPTIMAL_SIGMA_TLU = 1
OPTIMAL_C_TLU = 1
OPTIMAL_NUM_REPEATS_OIM_TLU = 20
OPTIMAL_NUM_REPEATS_REW_TLU = 20

# TIM_T parameters
OPTIMAL_EPS_TIM_T = 0.5
OPTIMAL_NUM_REPEATS_REW_TIM_T = 20

setup_array = []

# Setting up TIM_t (offline algorithm)
setup_array.append(
    {
        "algo_name": "tim_t",
        "function": tim_t_parallel,
        "args": [DATASET, DATASET_NODES, DATASET_TIMES],
        "kwargs": {
            "num_seeds": NUM_SEEDS_TO_FIND,
            "num_repeats_reward": OPTIMAL_NUM_REPEATS_REW_TIM_T,
            "epsilon": OPTIMAL_EPS_TIM_T,
        },
    }
)


# Setting up RSB with persistent features
setup_array.append(
    {
        "algo_name": "rsb_persist",
        "function": rsb2,
        "args": [DATASET, DATASET_NODES, DATASET_TIMES],
        "kwargs": {
            "num_seeds": NUM_SEEDS_TO_FIND,
            "C": OPTIMAL_C_RSB,
            "gamma": OPTIMAL_GAMMA_RSB,
            "num_repeats_expect": NUM_REPEATS_EXPECT_RSB,
            "persist_params": True,
            "hide_tqdm": True,
        },
    }
)

# Setting up RSB without persistent features
setup_array.append(
    {
        "algo_name": "rsb_nopersist",
        "function": rsb2,
        "args": [DATASET, DATASET_NODES, DATASET_TIMES],
        "kwargs": {
            "num_seeds": NUM_SEEDS_TO_FIND,
            "C": OPTIMAL_C_RSB,
            "gamma": OPTIMAL_GAMMA_RSB,
            "num_repeats_expect": NUM_REPEATS_EXPECT_RSB,
            "persist_params": False,
            "hide_tqdm": True,
        },
    }
)

# Setting up TIMLinUCB
setup_array.append(
    {
        "algo_name": "timlinucb_nopersist",
        "function": timlinucb_parallel_oim,
        "args": [DATASET, DATASET_FEATS, DATASET_TIMES, DATASET_NODES],
        "kwargs": {
            "num_seeds": NUM_SEEDS_TO_FIND,
            "num_repeats_oim": OPTIMAL_NUM_REPEATS_OIM_TLU,
            "num_repeats_oim_reward": OPTIMAL_NUM_REPEATS_REW_TLU,
            "sigma": OPTIMAL_SIGMA_TLU,
            "c": OPTIMAL_C_TLU,
            "epsilon": OPTIMAL_EPS_TLU,
            "hide_tqdm": True,
            "process_id": "digg1",
            "max_jobs": -5,
        },
    }
)

# Setting up TIMLinUCB (persistent parameters)
setup_array.append(
    {
        "algo_name": "timlinucb_persist",
        "function": timlinucb_parallel_t,
        "args": [DATASET, DATASET_FEATS, DATASET_TIMES, DATASET_NODES],
        "kwargs": {
            "num_seeds": NUM_SEEDS_TO_FIND,
            "num_repeats_oim": OPTIMAL_NUM_REPEATS_OIM_TLU,
            "num_repeats_oim_reward": OPTIMAL_NUM_REPEATS_REW_TLU,
            "sigma": OPTIMAL_SIGMA_TLU,
            "c": OPTIMAL_C_TLU,
            "epsilon": OPTIMAL_EPS_TLU,
            "persist": True,
            "process_id": "digg2",
        },
    }
)


with tqdm_joblib(tqdm(desc="Algorithm comparison (start)", total=len(setup_array))):
    results_array = joblib.Parallel(n_jobs=len(setup_array))(
        joblib.delayed(run_algorithm)(setup_dict) for setup_dict in setup_array
    )


with open("comparison_digg_20s_30t_end_1.pickle", "wb") as f:
    pickle.dump(results_array, f)
