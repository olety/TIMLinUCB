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
# %% ------------------------- Preparing the Facebook friends dataset ------------------
# --------------------------------------------------------------------------------------

#
# logging.debug("Processing the Facebook dataset...")
# logging.debug("Getting the graph...")
# df_facebook = pd.read_csv(
#     DATASET_FACEBOOK,
#     sep=" ",
#     # ??? is probably weights, they are always 1
#     names=["source", "target", "???", "timestamp"],
#     skiprows=2,
# )
#
# # Processing the dataframe
# logging.debug("Splitting the graph into time steps...")
# df_facebook["day"] = pd.to_datetime(df_facebook["timestamp"], unit="s").dt.floor("d")
# df_facebook = df_facebook.sort_values(by=["timestamp", "source", "target"])
# df_facebook = df_facebook[["source", "target", "day"]]
# df_facebook_nodes = np.sort(
#     np.unique(np.hstack((df_facebook["source"], df_facebook["target"])))
# )
#
# # Getting the true weights
# logging.debug('Generating "true" activation probabilities...')
# df_facebook["probab"] = np.random.uniform(0, 0.1, size=df_facebook.shape[0])
# # df_facebook['probab'] = np.random.uniform(0, 1, size=df_facebook.shape[0])
# if NUMBER_OF_DAYS == "ALL":
#     df_facebook_times = np.sort(np.unique(df_facebook["day"]))[0:]
# else:
#     df_facebook_times = np.sort(np.unique(df_facebook["day"]))[0:NUMBER_OF_DAYS]
#
# # TIMLinUCB only
# logging.debug("Generating node2vec features...")
# df_facebook_feats = generate_node2vec_fetures(
#     df_facebook, num_features=NUM_FEATURES_NODE2VEC
# )

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

# --------------------------------------------------------------------------------------
# %% --------------------------- Preparing the School dataset --------------------------
# --------------------------------------------------------------------------------------
#
# logging.debug("Processing the School dataset...")
# logging.debug("Getting the graph...")
# df_school = pd.read_csv(
#     DATASET_SCHOOL, sep=",", names=["source", "target", "timestamp"]
# )
#
# # Processing the dataframe
# logging.debug("Splitting the graph into time steps...")
# df_school["day"] = pd.to_datetime(df_school["timestamp"], unit="s").dt.floor("d")
# df_school = df_school.sort_values(by=["timestamp", "source", "target"])
# df_school = df_school[["source", "target", "day"]]
# df_school_nodes = np.sort(
#     np.unique(np.hstack((df_school["source"], df_school["target"])))
# )
#
# # Getting the true weights
# logging.debug('Generating "true" activation probabilities...')
# df_school["probab"] = np.random.uniform(0, 0.1, size=df_school.shape[0])
# # df_school['probab'] = np.random.uniform(0, 1, size=df_school.shape[0])
# if NUMBER_OF_DAYS == "ALL":
#     df_school_times = np.sort(np.unique(df_school["day"]))[0:20]
# else:
#     df_school_times = np.sort(np.unique(df_school["day"]))[0:]
#
# # TIMLinUCB only
# logging.debug("Generating node2vec features...")
# df_school_feats = generate_node2vec_fetures(
#     df_school, dataset_name="school", num_features=NUM_FEATURES_NODE2VEC
# )

# --------------------------------------------------------------------------------------
# %% ------------------------ Dataset to use for starting experiments ------------------
# --------------------------------------------------------------------------------------


# DATASET = df_facebook
# DATASET_FEATS = df_facebook_feats
# DATASET_TIMES = df_facebook_times
# DATASET_NODES = df_facebook_nodes

# --------------------------------------------------------------------------------------
# %% ------------------------- Comparing diff num_repeats_oim --------------------------
# --------------------------------------------------------------------------------------
#
# logging.debug("Starting search for best num_repeats_oim - TIMLinUCB/Facebook")
# OPTIMAL_SIGMA = 5
# OPTIMAL_C = 0.1
# OPTIMAL_EPS = 0.4
# OPTIMAL_NUM_REPEATS_REWARD = 20
# NUMBER_REPEATS_ARR = [1, 5, 7, 10, 20, 30]
# results_array = []
#
# for num_repeats in tqdm(
#     NUMBER_REPEATS_ARR, desc="Repeats search", leave=True, file=sys.stderr
# ):
#     results_array.append(
#         {
#             "sigma": OPTIMAL_SIGMA,
#             "c": OPTIMAL_C,
#             "epsilon": OPTIMAL_EPS,
#             "num_repeats": num_repeats,
#             "df": timlinucb(
#                 DATASET,
#                 DATASET_FEATS,
#                 DATASET_TIMES,
#                 DATASET_NODES,
#                 num_seeds=NUM_SEEDS_TO_FIND,
#                 num_repeats_oim=num_repeats,
#                 num_repeats_oim_reward=OPTIMAL_NUM_REPEATS_REWARD,
#                 sigma=OPTIMAL_SIGMA,
#                 c=OPTIMAL_C,
#                 epsilon=OPTIMAL_EPS,
#             ),
#         }
#     )
#
# with open("num_reps_facebook.pickle", "wb") as f:
#     pickle.dump(results_array, f)

# --------------------------------------------------------------------------------------
# %% ------------------------ Comparing diff num_repeats_reward ------------------------
# --------------------------------------------------------------------------------------

# logging.debug("Starting search for best num_repeats_reward - TIMLinUCB/Facebook")
# OPTIMAL_SIGMA = 5
# OPTIMAL_C = 0.1
# OPTIMAL_EPS = 0.4
# OPTIMAL_NUM_REPEATS_OIM = 5
#
# NUMBER_REPEATS_REWARD_ARR = [1, 5, 10, 20, 50, 100, 150]
# results_array = []
#
# for num_repeats in tqdm(
#     NUMBER_REPEATS_REWARD_ARR, desc="Repeats reward search", leave=True, file=sys.stderr
# ):
#     results_array.append(
#         {
#             "sigma": OPTIMAL_SIGMA,
#             "c": OPTIMAL_C,
#             "epsilon": OPTIMAL_EPS,
#             "num_repeats": OPTIMAL_NUM_REPEATS_OIM,
#             "num_repeats_reward": num_repeats,
#             "df": timlinucb(
#                 DATASET,
#                 DATASET_FEATS,
#                 DATASET_TIMES,
#                 DATASET_NODES,
#                 num_seeds=NUM_SEEDS_TO_FIND,
#                 num_repeats_oim=OPTIMAL_NUM_REPEATS_OIM,
#                 num_repeats_oim_reward=num_repeats,
#                 sigma=OPTIMAL_SIGMA,
#                 c=OPTIMAL_C,
#                 epsilon=OPTIMAL_EPS,
#             ),
#         }
#     )
#
# with open("num_reps_reward_facebook.pickle", "wb") as f:
#     pickle.dump(results_array, f)

# --------------------------------------------------------------------------------------
# %% -------------------- [Sequential] Grid search for best params ---------------------
# --------------------------------------------------------------------------------------
#
# logging.debug("Starting sequential grid search for best params - TIMLinUCB/Facebook")
# sigma_array = [0.01, 0.1, 0.5, 1, 5, 10]  # was 4
# c_array = [0.01, 0.1, 0.5, 1, 5, 10]  # was 0.1
# epsilon_array = [1, 5, 10, 50]  # was 0.4 [0.1, 0.5]
# results_array = []
#
# OPTIMAL_NUM_REPEATS_REWARD = 20
# OPTIMAL_NUM_REPEATS_OIM = 5
#
# for sigma in tqdm(sigma_array, desc="Sigma search", leave=True, file=sys.stderr):
#     for c in tqdm(c_array, desc="C search", leave=True, file=sys.stderr):
#         for eps in tqdm(epsilon_array, desc="E search", leave=True, file=sys.stderr):
#             results_array.append(
#                 {
#                     "sigma": sigma,
#                     "c": c,
#                     "epsilon": eps,
#                     "df": timlinucb_parallel_t(
#                         DATASET,
#                         DATASET_FEATS,
#                         DATASET_TIMES,
#                         DATASET_NODES,
#                         num_seeds=NUM_SEEDS_TO_FIND,
#                         num_repeats_oim=OPTIMAL_NUM_REPEATS_OIM,
#                         num_repeats_oim_reward=OPTIMAL_NUM_REPEATS_REWARD,
#                         sigma=sigma,
#                         c=c,
#                         epsilon=eps,
#                     ),
#                 }
#             )
#
#
# with open("grid_facebook_seq.pickle", "wb") as f:
#     pickle.dump(results_array, f)

# --------------------------------------------------------------------------------------
# %% --------------------- [Parallel] Grid search for best params ----------------------
# --------------------------------------------------------------------------------------
#
# logging.debug("Starting parallel grid search for best params - TIMLinUCB/Facebook")
# sigma_array = [0.01, 0.1, 0.5, 1, 5, 10]  # was 4
# c_array = [0.01, 0.1, 0.5, 1, 5, 10]  # was 0.1
# epsilon_array = [0.1]  # was 0.4 [0.1, 0.5, 1, 5, 10, 50]
# params_array = []
# results_array = []
#
#
# OPTIMAL_NUM_REPEATS_REWARD = 20
# OPTIMAL_NUM_REPEATS_OIM = 5
#
#
# for sigma in sigma_array:
#     for c in c_array:
#         for epsilon in epsilon_array:
#             params_array.append({"c": c, "sigma": sigma, "epsilon": epsilon})
#
#
# def parallel_grid(i, sigma, epsilon, c):
#     return {
#         "sigma": sigma,
#         "c": c,
#         "epsilon": epsilon,
#         "df": timlinucb_parallel_t(
#             DATASET,
#             DATASET_FEATS,
#             DATASET_TIMES,
#             DATASET_NODES,
#             num_seeds=NUM_SEEDS_TO_FIND,
#             num_repeats_oim=OPTIMAL_NUM_REPEATS_OIM,
#             num_repeats_oim_reward=OPTIMAL_NUM_REPEATS_REWARD,
#             sigma=sigma,
#             c=c,
#             epsilon=epsilon,
#             process_id=i,
#         ),
#     }
#
#
# with tqdm_joblib(tqdm(desc="Grid search", total=len(params_array))):
#     results_array = joblib.Parallel(n_jobs=-2)(
#         joblib.delayed(parallel_grid)(
#             i,
#             params_array[i]["sigma"],
#             params_array[i]["epsilon"],
#             params_array[i]["c"],
#         )
#         for i in range(len(params_array))
#     )
#
#
# with open("grid_par_smalleps_facebook.pickle", "wb") as f:
#     pickle.dump(results_array, f)


# --------------------------------------------------------------------------------------
# %% ------------------------- Finding best params of RSB ------------------------------
# --------------------------------------------------------------------------------------

#
# logging.debug("Starting grid search for best params - RSB/Facebook")
# c_array = [0.01, 0.1, 0.5, 1, 5, 10, 50]  # was 0.1
# gamma_array = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1]  # was 0.4
# results_array = []
#
#
# for gamma in tqdm(gamma_array, desc="Gamma search", leave=True, file=sys.stderr):
#     for c in tqdm(c_array, desc="C search", leave=True, file=sys.stderr):
#         results_array.append(
#             {
#                 "gamma": gamma,
#                 "c": c,
#                 "df": rsb2(
#                     DATASET,
#                     DATASET_NODES,
#                     DATASET_TIMES,
#                     num_seeds=NUM_SEEDS_TO_FIND,
#                     C=c,
#                     gamma=gamma,
#                     num_repeats_expect=25,
#                     persist_params=False,
#                 ),
#             }
#         )
#
# with open("grid_rsb.pickle", "wb") as f:
#     pickle.dump(results_array, f)

# --------------------------------------------------------------------------------------
# %% ---------------------- Looking at RSB's num_repeats_expect ------------------------
# --------------------------------------------------------------------------------------

# logging.debug("Starting num_repeats search - RSB/Facebook")
# OPTIMAL_C_RSB = 1
# OPTIMAL_GAMMA_RSB = 0.4
# num_repeats_array = [1, 5, 10, 20, 50, 75, 100, 125, 150]
# results_array = []
#
# for num_repeats_expect in tqdm(
#     num_repeats_array, desc="Num repeats", leave=True, file=sys.stderr
# ):
#     results_array.append(
#         {
#             "gamma": OPTIMAL_GAMMA_RSB,
#             "c": OPTIMAL_C_RSB,
#             "num_repeats_expect": num_repeats_expect,
#             "df": rsb2(
#                 DATASET,
#                 DATASET_NODES,
#                 DATASET_TIMES,
#                 num_seeds=NUM_SEEDS_TO_FIND,
#                 C=OPTIMAL_C_RSB,
#                 gamma=OPTIMAL_GAMMA_RSB,
#                 num_repeats_expect=num_repeats_expect,
#                 persist_params=False,
#             ),
#         }
#     )
#
# with open("num_repeats_rsb_facebook.pickle", "wb") as f:
#     pickle.dump(results_array, f)

# --------------------------------------------------------------------------------------
# %% ------ Comparing (parallel) TIMLinUCB w/ RSB and t_TIM (Facebook - 5 seeds) -------
# --------------------------------------------------------------------------------------
#
# logging.debug("Comparing TIMLinUCB with t_TIM and RSB - Facebook dataset")
# NUM_SEEDS_TO_FIND = 5
#
# # RSB parameters
# OPTIMAL_C_RSB = 1
# OPTIMAL_GAMMA_RSB = 0.4
#
# # TIMLinUCB parameters
# OPTIMAL_EPS_TLU = 0.5
# OPTIMAL_SIGMA_TLU = 1
# OPTIMAL_C_TLU = 0.1
# OPTIMAL_NUM_REPEATS_OIM_TLU = 20
# OPTIMAL_NUM_REPEATS_REW_TLU = 20
#
# setup_array = []
#
# # Setting up TIM_t (offline algorithm)
# setup_array.append(
#     {
#         "algo_name": "tim_t",
#         "function": tim_t_parallel,
#         "args": [DATASET, DATASET_NODES, DATASET_TIMES],
#         "kwargs": {
#             "num_seeds": NUM_SEEDS_TO_FIND,
#             "num_repeats_reward": 25,
#             "epsilon": 0.1,
#         },
#     }
# )
#
# # Setting up RSB with persistent features
# setup_array.append(
#     {
#         "algo_name": "rsb_persist",
#         "function": rsb_parallel,
#         "args": [DATASET, DATASET_NODES, DATASET_TIMES],
#         "kwargs": {
#             "num_seeds": NUM_SEEDS_TO_FIND,
#             "C": OPTIMAL_C_RSB,
#             "gamma": OPTIMAL_GAMMA_RSB,
#             "num_repeats_expect": 25,
#             "persist_params": True,
#         },
#     }
# )
#
# # Setting up RSB without persistent features
# setup_array.append(
#     {
#         "algo_name": "rsb_nopersist",
#         "function": rsb_parallel,
#         "args": [DATASET, DATASET_NODES, DATASET_TIMES],
#         "kwargs": {
#             "num_seeds": NUM_SEEDS_TO_FIND,
#             "C": OPTIMAL_C_RSB,
#             "gamma": OPTIMAL_GAMMA_RSB,
#             "num_repeats_expect": 25,
#             "persist_params": False,
#         },
#     }
# )
#
# # Setting up TIMLinUCB
# setup_array.append(
#     {
#         "algo_name": "timlinucb",
#         "function": timlinucb_parallel_t,
#         "args": [DATASET, DATASET_FEATS, DATASET_TIMES, DATASET_NODES],
#         "kwargs": {
#             "num_seeds": NUM_SEEDS_TO_FIND,
#             "num_repeats_oim": OPTIMAL_NUM_REPEATS_OIM_TLU,
#             "num_repeats_oim_reward": OPTIMAL_NUM_REPEATS_REW_TLU,
#             "sigma": OPTIMAL_SIGMA_TLU,
#             "c": OPTIMAL_C_TLU,
#             "epsilon": OPTIMAL_EPS_TLU,
#         },
#     }
# )
#
#
# with tqdm_joblib(tqdm(desc="Algorithm comparison (5 seeds)", total=len(setup_array))):
#     results_array = joblib.Parallel(n_jobs=len(setup_array))(
#         joblib.delayed(run_algorithm)(setup_dict) for setup_dict in setup_array
#     )
#
#
# with open("comparison_facebook_5seeds.pickle", "wb") as f:
#     pickle.dump(results_array, f)


# --------------------------------------------------------------------------------------
# %% ----------- Comparing TIMLinUCB w/ RSB and t_TIM (Facebook - 5 seeds) -------------
# --------------------------------------------------------------------------------------
#
# logging.debug("Comparing TIMLinUCB with t_TIM and RSB - Facebook dataset")
# NUM_SEEDS_TO_FIND = 5
#
# # RSB parameters
# OPTIMAL_C_RSB = 1
# OPTIMAL_GAMMA_RSB = 0.4
#
# # TIMLinUCB parameters
# OPTIMAL_EPS_TLU = 0.5
# OPTIMAL_SIGMA_TLU = 1
# OPTIMAL_C_TLU = 0.1
# OPTIMAL_NUM_REPEATS_OIM_TLU = 20
# OPTIMAL_NUM_REPEATS_REW_TLU = 20
#
# setup_array = []
# results = {}
#
# # Setting up TIM_t (offline algorithm)
# results["tim_t"] = tim_t(
#     DATASET,
#     DATASET_NODES,
#     DATASET_TIMES,
#     num_seeds=NUM_SEEDS_TO_FIND,
#     num_repeats_reward=25,
#     epsilon=0.1,
# )
#
# # Setting up RSB with persistent features
# results["rsb_persist"] = rsb2(
#     DATASET,
#     DATASET_NODES,
#     DATASET_TIMES,
#     num_seeds=NUM_SEEDS_TO_FIND,
#     C=OPTIMAL_C_RSB,
#     gamma=OPTIMAL_GAMMA_RSB,
#     num_repeats_expect=25,
#     persist_params=True,
# )
#
# # Setting up RSB without persistent features
# results["rsb_nopersist"] = rsb2(
#     DATASET,
#     DATASET_NODES,
#     DATASET_TIMES,
#     num_seeds=NUM_SEEDS_TO_FIND,
#     C=OPTIMAL_C_RSB,
#     gamma=OPTIMAL_GAMMA_RSB,
#     num_repeats_expect=25,
#     persist_params=False,
# )
#
# # Setting up TIMLinUCB
# results["timlinucb"] = timlinucb(
#     DATASET,
#     DATASET_FEATS,
#     DATASET_TIMES,
#     DATASET_NODES,
#     num_seeds=NUM_SEEDS_TO_FIND,
#     num_repeats_oim=OPTIMAL_NUM_REPEATS_OIM_TLU,
#     num_repeats_oim_reward=OPTIMAL_NUM_REPEATS_REW_TLU,
#     sigma=OPTIMAL_SIGMA_TLU,
#     c=OPTIMAL_C_TLU,
#     epsilon=OPTIMAL_EPS_TLU,
# )
#
#
# with open("comparison_facebook_5seeds.pickle", "wb") as f:
#     pickle.dump(results, f)


# --------------------------------------------------------------------------------------
# %% ------------ Comparing TIMLinUCB w/ RSB and t_TIM (Facebook - 20 seeds) -----------
# --------------------------------------------------------------------------------------

# logging.debug("Comparing TIMLinUCB with t_TIM and RSB - Facebook dataset")
# NUM_SEEDS_TO_FIND = 20
#
# # RSB parameters
# OPTIMAL_C_RSB = 1
# OPTIMAL_GAMMA_RSB = 0.4
# NUM_REPEATS_EXPECT_RSB = 20
#
# # TIMLinUCB parameters
# OPTIMAL_EPS_TLU = 0.5
# OPTIMAL_SIGMA_TLU = 1
# OPTIMAL_C_TLU = 0.1
# OPTIMAL_NUM_REPEATS_OIM_TLU = 20
# OPTIMAL_NUM_REPEATS_REW_TLU = 20
#
# # TIM_T parameters
# OPTIMAL_EPS_TIM_T = 0.5
# OPTIMAL_NUM_REPEATS_REW_TIM_T = 20
#
# setup_array = []
#
# # Setting up TIM_t (offline algorithm)
# setup_array.append(
#     {
#         "algo_name": "tim_t",
#         "function": tim_t_parallel,
#         "args": [DATASET, DATASET_NODES, DATASET_TIMES],
#         "kwargs": {
#             "num_seeds": NUM_SEEDS_TO_FIND,
#             "num_repeats_reward": OPTIMAL_NUM_REPEATS_REW_TIM_T,
#             "epsilon": OPTIMAL_EPS_TIM_T,
#         },
#     }
# )
#
# # Setting up RSB with persistent features
# setup_array.append(
#     {
#         "algo_name": "rsb_persist",
#         "function": rsb_parallel,
#         "args": [DATASET, DATASET_NODES, DATASET_TIMES],
#         "kwargs": {
#             "num_seeds": NUM_SEEDS_TO_FIND,
#             "C": OPTIMAL_C_RSB,
#             "gamma": OPTIMAL_GAMMA_RSB,
#             "num_repeats_expect": NUM_REPEATS_EXPECT_RSB,
#             "persist_params": True,
#         },
#     }
# )
#
# # Setting up RSB without persistent features
# setup_array.append(
#     {
#         "algo_name": "rsb_nopersist",
#         "function": rsb_parallel,
#         "args": [DATASET, DATASET_NODES, DATASET_TIMES],
#         "kwargs": {
#             "num_seeds": NUM_SEEDS_TO_FIND,
#             "C": OPTIMAL_C_RSB,
#             "gamma": OPTIMAL_GAMMA_RSB,
#             "num_repeats_expect": NUM_REPEATS_EXPECT_RSB,
#             "persist_params": False,
#         },
#     }
# )
#
# # Setting up TIMLinUCB
# setup_array.append(
#     {
#         "algo_name": "timlinucb",
#         "function": timlinucb_parallel_t,
#         "args": [DATASET, DATASET_FEATS, DATASET_TIMES, DATASET_NODES],
#         "kwargs": {
#             "num_seeds": NUM_SEEDS_TO_FIND,
#             "num_repeats_oim": OPTIMAL_NUM_REPEATS_OIM_TLU,
#             "num_repeats_oim_reward": OPTIMAL_NUM_REPEATS_REW_TLU,
#             "sigma": OPTIMAL_SIGMA_TLU,
#             "c": OPTIMAL_C_TLU,
#             "epsilon": OPTIMAL_EPS_TLU,
#         },
#     }
# )
#
#
# with tqdm_joblib(tqdm(desc="Algorithm comparison (20 seeds)", total=len(setup_array))):
#     results_array = joblib.Parallel(n_jobs=len(setup_array))(
#         joblib.delayed(run_algorithm)(setup_dict) for setup_dict in setup_array
#     )
#
#
# with open("comparison_facebook_20seeds.pickle", "wb") as f:
#     pickle.dump(results_array, f)

# --------------------------------------------------------------------------------------
# %% --------- Comparing TIMLinUCB w/ RSB and t_TIM (Facebook - 20s/100days) -----------
# --------------------------------------------------------------------------------------

# logging.debug("Comparing TIMLinUCB with t_TIM and RSB - Facebook dataset (100 days)")
# NUMBER_OF_DAYS = 100
# df_facebook_times = np.sort(np.unique(df_facebook["day"]))[0:NUMBER_OF_DAYS]
#
# # TIMLinUCB only
# logging.debug("Generating node2vec features...")
# df_facebook_feats = generate_node2vec_fetures(
#     df_facebook, num_features=NUM_FEATURES_NODE2VEC
# )
#
# DATASET = df_facebook
# DATASET_FEATS = df_facebook_feats
# DATASET_TIMES = df_facebook_times
# NUM_SEEDS_TO_FIND = 20
#
# # RSB parameters
# OPTIMAL_C_RSB = 1
# OPTIMAL_GAMMA_RSB = 0.4
# NUM_REPEATS_EXPECT_RSB = 20
#
# # TIMLinUCB parameters
# OPTIMAL_EPS_TLU = 0.5
# OPTIMAL_SIGMA_TLU = 1
# OPTIMAL_C_TLU = 0.1
# OPTIMAL_NUM_REPEATS_OIM_TLU = 20
# OPTIMAL_NUM_REPEATS_REW_TLU = 20
#
# # TIM_T parameters
# OPTIMAL_EPS_TIM_T = 0.5
# OPTIMAL_NUM_REPEATS_REW_TIM_T = 20
#
# setup_array = []
#
# # Setting up TIM_t (offline algorithm)
# setup_array.append(
#     {
#         "algo_name": "tim_t",
#         "function": tim_t_parallel,
#         "args": [DATASET, DATASET_NODES, DATASET_TIMES],
#         "kwargs": {
#             "num_seeds": NUM_SEEDS_TO_FIND,
#             "num_repeats_reward": OPTIMAL_NUM_REPEATS_REW_TIM_T,
#             "epsilon": OPTIMAL_EPS_TIM_T,
#         },
#     }
# )
#
# # Setting up RSB with persistent features
# setup_array.append(
#     {
#         "algo_name": "rsb_persist",
#         "function": rsb_parallel,
#         "args": [DATASET, DATASET_NODES, DATASET_TIMES],
#         "kwargs": {
#             "num_seeds": NUM_SEEDS_TO_FIND,
#             "C": OPTIMAL_C_RSB,
#             "gamma": OPTIMAL_GAMMA_RSB,
#             "num_repeats_expect": NUM_REPEATS_EXPECT_RSB,
#             "persist_params": True,
#         },
#     }
# )
#
# # Setting up RSB without persistent features
# setup_array.append(
#     {
#         "algo_name": "rsb_nopersist",
#         "function": rsb_parallel,
#         "args": [DATASET, DATASET_NODES, DATASET_TIMES],
#         "kwargs": {
#             "num_seeds": NUM_SEEDS_TO_FIND,
#             "C": OPTIMAL_C_RSB,
#             "gamma": OPTIMAL_GAMMA_RSB,
#             "num_repeats_expect": NUM_REPEATS_EXPECT_RSB,
#             "persist_params": False,
#         },
#     }
# )
#
# # Setting up TIMLinUCB
# setup_array.append(
#     {
#         "algo_name": "timlinucb",
#         "function": timlinucb_parallel_t,
#         "args": [DATASET, DATASET_FEATS, DATASET_TIMES, DATASET_NODES],
#         "kwargs": {
#             "num_seeds": NUM_SEEDS_TO_FIND,
#             "num_repeats_oim": OPTIMAL_NUM_REPEATS_OIM_TLU,
#             "num_repeats_oim_reward": OPTIMAL_NUM_REPEATS_REW_TLU,
#             "sigma": OPTIMAL_SIGMA_TLU,
#             "c": OPTIMAL_C_TLU,
#             "epsilon": OPTIMAL_EPS_TLU,
#         },
#     }
# )
#
#
# with tqdm_joblib(tqdm(desc="Algorithm comparison (20 seeds)", total=len(setup_array))):
#     results_array = joblib.Parallel(n_jobs=len(setup_array))(
#         joblib.delayed(run_algorithm)(setup_dict) for setup_dict in setup_array
#     )
#
#
# with open("comparison_facebook_20seeds_100t.pickle", "wb") as f:
#     pickle.dump(results_array, f)


# --------------------------------------------------------------------------------------
# %% ---------- Comparing dynamic and additive approaches for TIMLinUCB ----------------
# --------------------------------------------------------------------------------------

# logging.debug("Running the dynamic version of the algorithm")
# NUMBER_OF_DAYS = 100
# df_facebook_times = np.sort(np.unique(df_facebook["day"]))[0:NUMBER_OF_DAYS]
#
# # TIMLinUCB only
# logging.debug("Generating node2vec features...")
# df_facebook_feats = generate_node2vec_fetures(
#     df_facebook, num_features=NUM_FEATURES_NODE2VEC
# )
#
# DATASET = df_facebook
# DATASET_FEATS = df_facebook_feats
# DATASET_TIMES = df_facebook_times
# NUM_SEEDS_TO_FIND = 20
#
# # TIMLinUCB parameters
# OPTIMAL_EPS_TLU = 0.5
# OPTIMAL_SIGMA_TLU = 1
# OPTIMAL_C_TLU = 0.1
# OPTIMAL_NUM_REPEATS_OIM_TLU = 20
# OPTIMAL_NUM_REPEATS_REW_TLU = 20
#
# df_dynamic = timlinucb(
#     DATASET,
#     DATASET_FEATS,
#     DATASET_TIMES,
#     DATASET_NODES,
#     num_seeds=NUM_SEEDS_TO_FIND,
#     num_repeats_oim=OPTIMAL_NUM_REPEATS_OIM_TLU,
#     num_repeats_oim_reward=OPTIMAL_NUM_REPEATS_REW_TLU,
#     sigma=OPTIMAL_SIGMA_TLU,
#     c=OPTIMAL_C_TLU,
#     epsilon=OPTIMAL_EPS_TLU,
#     style="dynamic",
# )
#
#
# with open("tlu_dynamic_100days_facebook.pickle", "wb") as f:
#     pickle.dump(df_dynamic, f)


# --------------------------------------------------------------------------------------
# %% --------------- Looking at persisting paramerets in TIMLinUCB ---------------------
# --------------------------------------------------------------------------------------

# logging.debug("Looking at the persistent parameters")
# NUMBER_OF_DAYS = 20
# df_facebook_times = np.sort(np.unique(df_facebook["day"]))[0:NUMBER_OF_DAYS]
#
# # TIMLinUCB only
# logging.debug("Generating node2vec features...")
# df_facebook_feats = generate_node2vec_fetures(
#     df_facebook, num_features=NUM_FEATURES_NODE2VEC
# )
#
# DATASET = df_facebook
# DATASET_FEATS = df_facebook_feats
# DATASET_TIMES = df_facebook_times
# NUM_SEEDS_TO_FIND = 20
#
# # TIMLinUCB parameters
# OPTIMAL_EPS_TLU = 0.5
# OPTIMAL_SIGMA_TLU = 1
# OPTIMAL_C_TLU = 0.1
# OPTIMAL_NUM_REPEATS_OIM_TLU = 20
# OPTIMAL_NUM_REPEATS_REW_TLU = 20
#
# df_persist = timlinucb(
#     DATASET,
#     DATASET_FEATS,
#     DATASET_TIMES,
#     DATASET_NODES,
#     num_seeds=NUM_SEEDS_TO_FIND,
#     num_repeats_oim=OPTIMAL_NUM_REPEATS_OIM_TLU,
#     num_repeats_oim_reward=OPTIMAL_NUM_REPEATS_REW_TLU,
#     sigma=OPTIMAL_SIGMA_TLU,
#     c=OPTIMAL_C_TLU,
#     epsilon=OPTIMAL_EPS_TLU,
#     style="additive",
#     persist=True,
# )
#
#
# with open("tlu_20days_persist_facebook.pickle", "wb") as f:
#     pickle.dump(df_persist, f)

# --------------------------------------------------------------------------------------
# %% ----------- Looking at persisting paramerets in TIMLinUCB (100t) ------------------
# --------------------------------------------------------------------------------------

# logging.debug("Looking at the persistent parameters")
# NUMBER_OF_DAYS = 100
# df_facebook_times = np.sort(np.unique(df_facebook["day"]))[0:NUMBER_OF_DAYS]
#
# # TIMLinUCB only
# logging.debug("Generating node2vec features...")
# df_facebook_feats = generate_node2vec_fetures(
#     df_facebook, num_features=NUM_FEATURES_NODE2VEC
# )
#
# DATASET = df_facebook
# DATASET_FEATS = df_facebook_feats
# DATASET_TIMES = df_facebook_times
# NUM_SEEDS_TO_FIND = 20
#
# # TIMLinUCB parameters
# OPTIMAL_EPS_TLU = 0.5
# OPTIMAL_SIGMA_TLU = 1
# OPTIMAL_C_TLU = 0.1
# OPTIMAL_NUM_REPEATS_OIM_TLU = 20
# OPTIMAL_NUM_REPEATS_REW_TLU = 20
#
# df_persist = timlinucb(
#     DATASET,
#     DATASET_FEATS,
#     DATASET_TIMES,
#     DATASET_NODES,
#     num_seeds=NUM_SEEDS_TO_FIND,
#     num_repeats_oim=OPTIMAL_NUM_REPEATS_OIM_TLU,
#     num_repeats_oim_reward=OPTIMAL_NUM_REPEATS_REW_TLU,
#     sigma=OPTIMAL_SIGMA_TLU,
#     c=OPTIMAL_C_TLU,
#     epsilon=OPTIMAL_EPS_TLU,
#     style="additive",
#     persist=True,
# )
#
#
# with open("tlu_100days_persist_facebook.pickle", "wb") as f:
#     pickle.dump(df_persist, f)

# --------------------------------------------------------------------------------------
# %% -------------------- Looking at parallel OIM in TIMLinUCB ------------------------
# --------------------------------------------------------------------------------------

# logging.debug("Looking at the persistent parameters")
# NUMBER_OF_DAYS = 100
# df_facebook_times = np.sort(np.unique(df_facebook["day"]))[0:NUMBER_OF_DAYS]
#
# # TIMLinUCB only
# logging.debug("Generating node2vec features...")
# df_facebook_feats = generate_node2vec_fetures(
#     df_facebook, num_features=NUM_FEATURES_NODE2VEC
# )
#
# DATASET = df_facebook
# DATASET_FEATS = df_facebook_feats
# DATASET_TIMES = df_facebook_times
# NUM_SEEDS_TO_FIND = 20
#
# # TIMLinUCB parameters
# OPTIMAL_EPS_TLU = 0.5
# OPTIMAL_SIGMA_TLU = 1
# OPTIMAL_C_TLU = 0.1
# OPTIMAL_NUM_REPEATS_OIM_TLU = 20
# OPTIMAL_NUM_REPEATS_REW_TLU = 20
#
# df_dynamic = timlinucb_parallel_oim(
#     DATASET,
#     DATASET_FEATS,
#     DATASET_TIMES,
#     DATASET_NODES,
#     num_seeds=NUM_SEEDS_TO_FIND,
#     num_repeats_oim=OPTIMAL_NUM_REPEATS_OIM_TLU,
#     num_repeats_oim_reward=OPTIMAL_NUM_REPEATS_REW_TLU,
#     sigma=OPTIMAL_SIGMA_TLU,
#     c=OPTIMAL_C_TLU,
#     epsilon=OPTIMAL_EPS_TLU,
#     style="additive",
# )
#
#
# with open("tlu_100days_par_facebook.pickle", "wb") as f:
#     pickle.dump(df_dynamic, f)


# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------
# %% -------------------------------- DIGG DATASET -------------------------------------
# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------

DATASET = df_digg
DATASET_FEATS = df_digg_feats
DATASET_TIMES = df_digg_times
DATASET_NODES = df_digg_nodes


# --------------------------------------------------------------------------------------
# %% --------------------- [Parallel] Grid search for best params ----------------------
# --------------------------------------------------------------------------------------

# logging.debug("Starting parallel grid search for best params - TIMLinUCB/Facebook")
# sigma_array = [0.01, 0.1, 0.5, 1, 5, 10]  # was 4
# c_array = [0.01, 0.1, 0.5, 1, 5, 10]  # was 0.1
# epsilon_array = [0.1, 0.5, 1, 5, 10, 50]  # was 0.4 [0.1, 0.5, 1, 5, 10, 50]
# params_array = []
# results_array = []
#
#
# OPTIMAL_NUM_REPEATS_REWARD = 20
# OPTIMAL_NUM_REPEATS_OIM = 5
#
#
# for sigma in sigma_array:
#     for c in c_array:
#         for epsilon in epsilon_array:
#             params_array.append({"c": c, "sigma": sigma, "epsilon": epsilon})
#
#
# def parallel_grid(i, sigma, epsilon, c):
#     return {
#         "sigma": sigma,
#         "c": c,
#         "epsilon": epsilon,
#         "df": timlinucb_parallel_t(
#             DATASET,
#             DATASET_FEATS,
#             DATASET_TIMES,
#             DATASET_NODES,
#             num_seeds=NUM_SEEDS_TO_FIND,
#             num_repeats_oim=OPTIMAL_NUM_REPEATS_OIM,
#             num_repeats_oim_reward=OPTIMAL_NUM_REPEATS_REWARD,
#             sigma=sigma,
#             c=c,
#             epsilon=epsilon,
#             process_id=i,
#         ),
#     }
#
#
# with tqdm_joblib(tqdm(desc="Grid search", total=len(params_array))):
#     results_array = joblib.Parallel(n_jobs=-3)(
#         joblib.delayed(parallel_grid)(
#             i,
#             params_array[i]["sigma"],
#             params_array[i]["epsilon"],
#             params_array[i]["c"],
#         )
#         for i in range(len(params_array))
#     )
#
#
# with open("grid_par_digg.pickle", "wb") as f:
#     pickle.dump(results_array, f)


# --------------------------------------------------------------------------------------
# %% ---------------------- Finding best params of RSB [Digg] --------------------------
# --------------------------------------------------------------------------------------


# logging.debug("Starting grid search for best params - RSB/Facebook")
# c_array = [0, 0.01, 0.1, 0.5, 1, 5, 10, 50]  # was 0.1
# gamma_array = [0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1]  # was 0.4
# NUM_REPEATS_EXPECT_RSB = 20
# results_array = []
#
#
# for gamma in tqdm(gamma_array, desc="Gamma search", leave=True, file=sys.stderr):
#     for c in tqdm(c_array, desc="C search", leave=True, file=sys.stderr):
#         results_array.append(
#             {
#                 "gamma": gamma,
#                 "c": c,
#                 "df": rsb2(
#                     DATASET,
#                     DATASET_NODES,
#                     DATASET_TIMES,
#                     num_seeds=NUM_SEEDS_TO_FIND,
#                     C=c,
#                     gamma=gamma,
#                     num_repeats_expect=NUM_REPEATS_EXPECT_RSB,
#                     persist_params=False,
#                 ),
#             }
#         )
#
# with open("grid_rsb_digg.pickle", "wb") as f:
#     pickle.dump(results_array, f)


# --------------------------------------------------------------------------------------
# %% ----------- Comparing TIMLinUCB w/ RSB and t_TIM (Digg - 20s/100days) -------------
# --------------------------------------------------------------------------------------

# logging.debug("Comparing TIMLinUCB with t_TIM and RSB - Digg dataset (100 days)")
# NUMBER_OF_DAYS = 100
# df_digg_times = np.sort(np.unique(df_digg["day"]))[0:NUMBER_OF_DAYS]
#
# # TIMLinUCB only
# # logging.debug("Generating node2vec features...")
# # df_digg_feats = generate_node2vec_fetures(df_digg, num_features=NUM_FEATURES_NODE2VEC)
#
# DATASET = df_digg
# DATASET_FEATS = df_digg_feats
# DATASET_TIMES = df_digg_times
# NUM_SEEDS_TO_FIND = 20
#
# # RSB parameters
# OPTIMAL_C_RSB = 10
# OPTIMAL_GAMMA_RSB = 0.4
# NUM_REPEATS_EXPECT_RSB = 20
#
# # TIMLinUCB parameters
# OPTIMAL_EPS_TLU = 0.5
# OPTIMAL_SIGMA_TLU = 1
# OPTIMAL_C_TLU = 1
# OPTIMAL_NUM_REPEATS_OIM_TLU = 20
# OPTIMAL_NUM_REPEATS_REW_TLU = 20
#
# # TIM_T parameters
# OPTIMAL_EPS_TIM_T = 0.5
# OPTIMAL_NUM_REPEATS_REW_TIM_T = 20
#
# setup_array = []

# Setting up TIM_t (offline algorithm)
# setup_array.append(
#     {
#         "algo_name": "tim_t",
#         "function": tim_t_parallel,
#         "args": [DATASET, DATASET_NODES, DATASET_TIMES],
#         "kwargs": {
#             "num_seeds": NUM_SEEDS_TO_FIND,
#             "num_repeats_reward": OPTIMAL_NUM_REPEATS_REW_TIM_T,
#             "epsilon": OPTIMAL_EPS_TIM_T,
#         },
#     }
# )

# # Setting up RSB with persistent features
# setup_array.append(
#     {
#         "algo_name": "rsb_persist",
#         "function": rsb2,
#         "args": [DATASET, DATASET_NODES, DATASET_TIMES],
#         "kwargs": {
#             "num_seeds": NUM_SEEDS_TO_FIND,
#             "C": OPTIMAL_C_RSB,
#             "gamma": OPTIMAL_GAMMA_RSB,
#             "num_repeats_expect": NUM_REPEATS_EXPECT_RSB,
#             "persist_params": True,
#             "hide_tqdm": True,
#         },
#     }
# )
#
# # Setting up RSB without persistent features
# setup_array.append(
#     {
#         "algo_name": "rsb_nopersist",
#         "function": rsb2,
#         "args": [DATASET, DATASET_NODES, DATASET_TIMES],
#         "kwargs": {
#             "num_seeds": NUM_SEEDS_TO_FIND,
#             "C": OPTIMAL_C_RSB,
#             "gamma": OPTIMAL_GAMMA_RSB,
#             "num_repeats_expect": NUM_REPEATS_EXPECT_RSB,
#             "persist_params": False,
#             "hide_tqdm": True,
#         },
#     }
# )

# Setting up TIMLinUCB
# setup_array.append(
#     {
#         "algo_name": "timlinucb",
#         "function": timlinucb_parallel_t,
#         "args": [DATASET, DATASET_FEATS, DATASET_TIMES, DATASET_NODES],
#         "kwargs": {
#             "num_seeds": NUM_SEEDS_TO_FIND,
#             "num_repeats_oim": OPTIMAL_NUM_REPEATS_OIM_TLU,
#             "num_repeats_oim_reward": OPTIMAL_NUM_REPEATS_REW_TLU,
#             "sigma": OPTIMAL_SIGMA_TLU,
#             "c": OPTIMAL_C_TLU,
#             "epsilon": OPTIMAL_EPS_TLU,
#         },
#     }
# )

#
# with tqdm_joblib(tqdm(desc="Algorithm comparison (20 seeds)", total=len(setup_array))):
#     results_array = joblib.Parallel(n_jobs=len(setup_array))(
#         joblib.delayed(run_algorithm)(setup_dict) for setup_dict in setup_array
#     )
#
#
# with open("digg_rsb_test.pickle", "wb") as f:
#     pickle.dump(results_array, f)


# --------------------------------------------------------------------------------------
# %% ----------- Comparing TIMLinUCB w/ RSB and t_TIM (Facebook - 5 seeds) -------------
# --------------------------------------------------------------------------------------

logging.debug("Comparing TIMLinUCB with t_TIM and RSB - Digg dataset (100 days)")
NUMBER_OF_DAYS = 100
df_digg_times = np.sort(np.unique(df_digg["day"]))[600 : 600 + NUMBER_OF_DAYS]

# TIMLinUCB only
# logging.debug("Generating node2vec features...")
# df_digg_feats = generate_node2vec_fetures(df_digg, num_features=NUM_FEATURES_NODE2VEC)

DATASET = df_digg
DATASET_FEATS = df_digg_feats
DATASET_TIMES = df_digg_times
NUM_SEEDS_TO_FIND = 20

# RSB parameters
OPTIMAL_C_RSB = 10
OPTIMAL_GAMMA_RSB = 0.4
NUM_REPEATS_EXPECT_RSB = 20

# TIMLinUCB parameters
OPTIMAL_EPS_TLU = 0.5
OPTIMAL_SIGMA_TLU = 1
OPTIMAL_C_TLU = 1
OPTIMAL_NUM_REPEATS_OIM_TLU = 2
OPTIMAL_NUM_REPEATS_REW_TLU = 20

# TIM_T parameters
OPTIMAL_EPS_TIM_T = 0.5
OPTIMAL_NUM_REPEATS_REW_TIM_T = 20


setup_array = []
results = {}

# Setting up TIMLinUCB
results["timlinucb_nopersist"] = timlinucb(
    DATASET,
    DATASET_FEATS,
    DATASET_TIMES,
    DATASET_NODES,
    num_seeds=NUM_SEEDS_TO_FIND,
    num_repeats_oim=OPTIMAL_NUM_REPEATS_OIM_TLU,
    num_repeats_oim_reward=OPTIMAL_NUM_REPEATS_REW_TLU,
    sigma=OPTIMAL_SIGMA_TLU,
    c=OPTIMAL_C_TLU,
    epsilon=OPTIMAL_EPS_TLU,
)

# Setting up TIMLinUCB
results["timlinucb_persist"] = timlinucb(
    DATASET,
    DATASET_FEATS,
    DATASET_TIMES,
    DATASET_NODES,
    num_seeds=NUM_SEEDS_TO_FIND,
    num_repeats_oim=OPTIMAL_NUM_REPEATS_OIM_TLU,
    num_repeats_oim_reward=OPTIMAL_NUM_REPEATS_REW_TLU,
    sigma=OPTIMAL_SIGMA_TLU,
    c=OPTIMAL_C_TLU,
    epsilon=OPTIMAL_EPS_TLU,
)


# Setting up TIM_t (offline algorithm)
results["tim_t"] = tim_t(
    DATASET,
    DATASET_NODES,
    DATASET_TIMES,
    num_seeds=NUM_SEEDS_TO_FIND,
    num_repeats_reward=OPTIMAL_NUM_REPEATS_REW_TIM_T,
    epsilon=OPTIMAL_EPS_TIM_T,
)

# Setting up RSB with persistent features
results["rsb_persist"] = rsb2(
    DATASET,
    DATASET_NODES,
    DATASET_TIMES,
    num_seeds=NUM_SEEDS_TO_FIND,
    C=OPTIMAL_C_RSB,
    gamma=OPTIMAL_GAMMA_RSB,
    num_repeats_expect=NUM_REPEATS_EXPECT_RSB,
    persist_params=True,
)

# Setting up RSB without persistent features
results["rsb_nopersist"] = rsb2(
    DATASET,
    DATASET_NODES,
    DATASET_TIMES,
    num_seeds=NUM_SEEDS_TO_FIND,
    C=OPTIMAL_C_RSB,
    gamma=OPTIMAL_GAMMA_RSB,
    num_repeats_expect=NUM_REPEATS_EXPECT_RSB,
    persist_params=False,
)


with open("comparison_digg_20seeds_600t.pickle", "wb") as f:
    pickle.dump(results, f)


# --------------------------------------------------------------------------------------
# %% ----------- Looking at persisting paramerets in TIMLinUCB (Digg - 100t) -----------
# --------------------------------------------------------------------------------------

# logging.debug("Looking at the persistent parameters")
# NUMBER_OF_DAYS = 100
# df_digg_times = np.sort(np.unique(df_digg["day"]))[0:NUMBER_OF_DAYS]
#
# # TIMLinUCB only
# logging.debug("Generating node2vec features...")
# df_digg_feats = generate_node2vec_fetures(
#     df_digg, num_features=NUM_FEATURES_NODE2VEC
# )
#
# DATASET = df_digg
# DATASET_FEATS = df_digg_feats
# DATASET_TIMES = df_digg_times
# NUM_SEEDS_TO_FIND = 20
#
# # TIMLinUCB parameters
# OPTIMAL_EPS_TLU = 0.5
# OPTIMAL_SIGMA_TLU = 1
# OPTIMAL_C_TLU = 0.1
# OPTIMAL_NUM_REPEATS_OIM_TLU = 20
# OPTIMAL_NUM_REPEATS_REW_TLU = 20
#
# df_persist = timlinucb(
#     DATASET,
#     DATASET_FEATS,
#     DATASET_TIMES,
#     DATASET_NODES,
#     num_seeds=NUM_SEEDS_TO_FIND,
#     num_repeats_oim=OPTIMAL_NUM_REPEATS_OIM_TLU,
#     num_repeats_oim_reward=OPTIMAL_NUM_REPEATS_REW_TLU,
#     sigma=OPTIMAL_SIGMA_TLU,
#     c=OPTIMAL_C_TLU,
#     epsilon=OPTIMAL_EPS_TLU,
#     style="additive",
#     persist=True,
# )
#
#
# with open("tlu_100days_persist_digg.pickle", "wb") as f:
#     pickle.dump(df_persist, f)


# %%
# from rsb import get_reward_arm
# from helpers import get_avg_reward
#
#
# nodes = DATASET_NODES
# num_seeds = 20
# times = np.sort(np.unique(df_digg["day"]))[1:600]
# df_edges = DATASET
# gamma = 0.4
# C = 10
#
# num_nodes = nodes.shape[0]
# df_weights = pd.DataFrame(
#     data=1.0,
#     index=nodes,
#     columns=[f"weight_{k}" for k in range(num_seeds)] + ["temp_weight"],
# )
# df_weights["walked"] = False
# df_weights["expected_gain"] = 0
# results = []
#
# for t in tqdm(times):
#     df_t = df_edges[df_edges["day"] <= t]
#
#     nodes_t = np.sort(np.unique(np.hstack((df_t["source"], df_t["target"]))))
#     num_nodes_t = nodes_t.shape[0]
#
#     df_weights["walked"] = False
#     df_weights_t = df_weights.loc[nodes_t]
#     selected = []
#
#     for cur_seed in range(num_seeds):
#         df_weights_t["temp_weight"] = (
#             gamma / num_nodes_t
#             + (1 - gamma)
#             * df_weights_t[f"weight_{cur_seed}"]
#             / df_weights_t[f"weight_{cur_seed}"].sum()
#         )
#
#         selection_probab = (
#             df_weights_t[~df_weights_t.index.isin(selected)]["temp_weight"]
#             / df_weights_t[~df_weights_t.index.isin(selected)]["temp_weight"].sum()
#         )
#
#         # Draw an arm
#         random_pt = random.uniform(0, df_weights_t[f"weight_{cur_seed}"].sum())
#
#         selected_node = (
#             df_weights_t[~df_weights_t.index.isin(selected)][
#                 f"weight_{cur_seed}"
#             ].cumsum()
#             >= random_pt
#         ).idxmax()
#
#         # Receiving the reward
#         affected_arm = get_reward_arm(df_t, df_weights_t, selected_node)
#         df_weights_t.loc[affected_arm, "walked"] = True
#         marginal_gain = len(affected_arm)
#         df_weights_t["expected_gain"] = 0
#         p_selected = selection_probab.loc[selected_node]
#         df_weights_t.loc[selected_node, "expected_gain"] = marginal_gain / p_selected
#
#         selected.append(selected_node)
#         df_weights_t[f"weight_{cur_seed}"] = df_weights_t[
#             f"weight_{cur_seed}"
#         ] * np.exp((gamma * df_weights_t["expected_gain"]) / (num_nodes * C))
#
#     df_weights.loc[df_weights_t.index] = np.nan
#     df_weights = df_weights.combine_first(df_weights_t)
#     results.append(
#         {"time": t, "reward": get_avg_reward(df_t, selected, 20),}
#     )
#
# # n
#
#
# # %%
# from rsb import get_reward_arm
# from helpers import get_avg_reward
#
#
# nodes = DATASET_NODES
# num_seeds = 20
# times = np.sort(np.unique(df_digg["day"]))[1:600]
# df_edges = DATASET
# gamma = 0.4
# C = 10
#
# num_nodes = nodes.shape[0]
# df_weights = pd.DataFrame(
#     data=1.0,
#     index=nodes,
#     columns=[f"weight_{k}" for k in range(num_seeds)] + ["temp_weight"],
# )
# df_weights["walked"] = False
# df_weights["expected_gain"] = 0
# results2 = []
#
# for t in tqdm(times):
#     df_t = df_edges[df_edges["day"] <= t]
#
#     nodes_t = np.sort(np.unique(np.hstack((df_t["source"], df_t["target"]))))
#     num_nodes_t = nodes_t.shape[0]
#
#     df_weights["walked"] = False
#     df_weights_t = df_weights.loc[nodes_t]
#     selected = []
#
#     for cur_seed in range(num_seeds):
#         df_weights_t["temp_weight"] = (
#             gamma / num_nodes_t
#             + (1 - gamma)
#             * df_weights_t[f"weight_{cur_seed}"]
#             / df_weights_t[f"weight_{cur_seed}"].sum()
#         )
#
#         selection_probab = (
#             df_weights_t[~df_weights_t.index.isin(selected)]["temp_weight"]
#             / df_weights_t[~df_weights_t.index.isin(selected)]["temp_weight"].sum()
#         )
#
#         # Draw an arm
#         random_pt = random.uniform(0, df_weights_t[f"weight_{cur_seed}"].sum())
#
#         selected_node = (
#             df_weights_t[~df_weights_t.index.isin(selected)][
#                 f"weight_{cur_seed}"
#             ].cumsum()
#             >= random_pt
#         ).idxmax()
#
#         # Receiving the reward
#         affected_arm = get_reward_arm(df_t, df_weights_t, selected_node)
#         df_weights_t.loc[affected_arm, "walked"] = True
#         marginal_gain = len(affected_arm)
#         df_weights_t["expected_gain"] = 0
#         p_selected = selection_probab.loc[selected_node]
#         df_weights_t.loc[selected_node, "expected_gain"] = marginal_gain / p_selected
#
#         selected.append(selected_node)
#         df_weights_t[f"weight_{cur_seed}"] = df_weights_t[
#             f"weight_{cur_seed}"
#         ] * np.exp((gamma * df_weights_t["expected_gain"]) / (num_nodes * C))
#
#     # df_weights.loc[df_weights_t.index] = np.nan
#     # df_weights = df_weights.combine_first(df_weights_t)
#     results2.append(
#         {"time": t, "reward": get_avg_reward(df_t, selected, 20),}
#     )
#
# # %%
# import seaborn as sns
#
# df_r = pd.DataFrame(results)
# df_r.columns = df_r.columns.map(
#     lambda x: str(x) + "_" + "persist" if x != "time" else x
# )
#
# df_r2 = pd.DataFrame(results2)
# df_r2.columns = df_r2.columns.map(
#     lambda x: str(x) + "_" + "nopersist" if x != "time" else x
# )
#
#
# df = pd.merge(df_r, df_r2, on="time")
# df.describe()
#
# df_plot = df[["time", "reward_persist", "reward_nopersist"]].melt(
#     "time", var_name="Algorithm", value_name="Reward"
# )
# sns.lineplot(x="time", y="Reward", hue="Algorithm", data=df_plot)
