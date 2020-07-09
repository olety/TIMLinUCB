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
from timlinucb import generate_node2vec_fetures, timlinucb, timlinucb_parallel
from helpers import tim_t
from rsb import rsb2

# --------------------------------------------------------------------------------------
# %% ------------------------------ Initial setup --------------------------------------
# --------------------------------------------------------------------------------------


# Fancy plots in matplotlib and pandas

random.seed(42)
plt.style.use("ggplot")
pd.options.mode.chained_assignment = None  # default='warn'

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


logging.debug("Processing the Facebook dataset...")
logging.debug("Getting the graph...")
df_facebook = pd.read_csv(
    DATASET_FACEBOOK,
    sep=" ",
    # ??? is probably weights, they are always 1
    names=["source", "target", "???", "timestamp"],
    skiprows=2,
)

# Processing the dataframe
logging.debug("Splitting the graph into time steps...")
df_facebook["day"] = pd.to_datetime(df_facebook["timestamp"], unit="s").dt.floor("d")
df_facebook = df_facebook.sort_values(by=["timestamp", "source", "target"])
df_facebook = df_facebook[["source", "target", "day"]]
df_facebook_nodes = np.sort(
    np.unique(np.hstack((df_facebook["source"], df_facebook["target"])))
)

# Getting the true weights
logging.debug('Generating "true" activation probabilities...')
df_facebook["probab"] = np.random.uniform(0, 0.1, size=df_facebook.shape[0])
# df_facebook['probab'] = np.random.uniform(0, 1, size=df_facebook.shape[0])
if NUMBER_OF_DAYS == "ALL":
    df_facebook_times = np.sort(np.unique(df_facebook["day"]))[0:]
else:
    df_facebook_times = np.sort(np.unique(df_facebook["day"]))[0:NUMBER_OF_DAYS]

# TIMLinUCB only
logging.debug("Generating node2vec features...")
df_facebook_feats = generate_node2vec_fetures(
    df_facebook, num_features=NUM_FEATURES_NODE2VEC
)

# --------------------------------------------------------------------------------------
# %% ------------------------- Preparing the Digg friends dataset ----------------------
# --------------------------------------------------------------------------------------
#
# logging.debug("Processing the Digg dataset...")
# logging.debug("Getting the graph...")
# df_digg = pd.read_csv(
#     DATASET_DIGG, sep=" ", names=["source", "target", "timestamp"], skiprows=1,
# )
#
# # Processing the dataframe
# logging.debug("Splitting the graph into time steps...")
# df_digg["day"] = pd.to_datetime(df_digg["timestamp"], unit="s").dt.floor("d")
# df_digg = df_digg.sort_values(by=["timestamp", "source", "target"])
# df_digg = df_digg[["source", "target", "day"]]
# df_digg_nodes = np.sort(np.unique(np.hstack((df_digg["source"], df_digg["target"]))))
#
# # Getting the true weights
# logging.debug('Generating "true" activation probabilities...')
# df_digg["probab"] = np.random.uniform(0, 0.1, size=df_digg.shape[0])
# # df_digg['probab'] = np.random.uniform(0, 1, size=df_digg.shape[0])
# if NUMBER_OF_DAYS == "ALL":
#     df_digg_times = np.sort(np.unique(df_digg["day"]))[0:20]
# else:
#     df_digg_times = np.sort(np.unique(df_digg["day"]))[0:]
#
# # TIMLinUCB only
# logging.debug("Generating node2vec features...")
# df_digg_feats = generate_node2vec_fetures(
#     df_digg, dataset_name="digg", num_features=NUM_FEATURES_NODE2VEC
# )

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


DATASET = df_facebook
DATASET_FEATS = df_facebook_feats
DATASET_TIMES = df_facebook_times
DATASET_NODES = df_facebook_nodes

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
# %% --------------------------- Grid search for best params ---------------------------
# --------------------------------------------------------------------------------------
#
logging.debug("Starting grid search for best params - TIMLinUCB/Facebook")
sigma_array = [0.01, 0.1, 0.5, 1, 5, 10]  # was 4
c_array = [0.01, 0.1, 0.5, 1, 5, 10]  # was 0.1
epsilon_array = [1, 5, 10, 50]  # was 0.4 [0.1, 0.5]
results_array = []

OPTIMAL_NUM_REPEATS_REWARD = 20
OPTIMAL_NUM_REPEATS_OIM = 5

for sigma in tqdm(sigma_array, desc="Sigma search", leave=True, file=sys.stderr):
    for c in tqdm(c_array, desc="C search", leave=True, file=sys.stderr):
        for eps in tqdm(epsilon_array, desc="E search", leave=True, file=sys.stderr):
            results_array.append(
                {
                    "sigma": sigma,
                    "c": c,
                    "epsilon": eps,
                    "df": timlinucb_parallel(
                        DATASET,
                        DATASET_FEATS,
                        DATASET_TIMES,
                        DATASET_NODES,
                        num_seeds=NUM_SEEDS_TO_FIND,
                        num_repeats_oim=OPTIMAL_NUM_REPEATS_OIM,
                        num_repeats_oim_reward=OPTIMAL_NUM_REPEATS_REWARD,
                        sigma=sigma,
                        c=c,
                        epsilon=eps,
                    ),
                }
            )


with open("grid_facebook.pickle", "wb") as f:
    pickle.dump(results_array, f)


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
# %% ---------------- Comparing TIMLinUCB w/ RSB and t_TIM (Facebook) ------------------
# --------------------------------------------------------------------------------------

# logging.debug("Comparing TIMLinUCB with t_TIM and RSB - Facebook dataset")
# NUM_SEEDS_TO_FIND = 5
# OPTIMAL_C_RSB = 1
# OPTIMAL_GAMMA_RSB = 0.4
# OPTIMAL_SIGMA_TLU = 5
# OPTIMAL_C_TLU = 0.1
# OPTIMAL_EPS_TLU = 0.1
# OPTIMAL_NUM_REPEATS_OIM_TLU = 20
# OPTIMAL_NUM_REPEATS_REW_TLU = 20
#
# logging.debug("Running TIM")
# df_tim = tim_t(
#     DATASET,
#     DATASET_NODES,
#     DATASET_TIMES,
#     num_seeds=NUM_SEEDS_TO_FIND,
#     num_repeats_reward=25,
#     epsilon=0.1,
# )
#
# logging.debug("Running RSB (persist)")
# df_rsb2_persist = rsb2(
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
# logging.debug("Running RSB (nopersist)")
# df_rsb2_nopersist = rsb2(
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
# logging.debug("Running TIMLinUCB")
# df_timlinucb = timlinucb(
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
#     pickle.dump([df_tim, df_rsb2_persist, df_rsb2_nopersist, df_timlinucb], f)

# --------------------------------------------------------------------------------------
# %% ---------- Comparing dynamic and additive approaches for TIMLinUCB ----------------
# --------------------------------------------------------------------------------------
#
# df = timlinucb(
#     DATASET,
#     DATASET_FEATS,
#     DATASET_TIMES,
#     DATASET_NODES,
#     num_seeds=NUM_SEEDS_TO_FIND,
#     num_repeats_oim=5,
#     num_repeats_oim_reward=1,
#     sigma=5,
#     c=0.1,
#     epsilon=0.1,
#     style="dynamic",
# )
