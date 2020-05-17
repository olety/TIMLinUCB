from functools import partial
import logging
from scipy.special import expit
import pandas as pd
import numpy as np
import scipy as sc
import scipy.sparse as sparse
import pickle
import re
import sys
import scipy.sparse.linalg
from tqdm import tqdm, trange
from subprocess import Popen, PIPE
import os
import networkx as nx
import itertools
from heapq import heappush, heappop
from copy import deepcopy
import random
import time

# %% Functions


def run_ic_eff(df_graph, seed_nodes):
    """ Runs independent cascade model.
    Input: df_g -- a dataframe representing the graph (with the probabilities)
    S -- initial set of vertices
    tracking -- whether we want to check for active/observed nodes
    Output: T -- resulted influenced set of vertices (including S)
    """
    affected_nodes = deepcopy(seed_nodes)  # copy already selected nodes
    activated_edges = []
    observed_edges = []
    df_graph["activated"] = df_graph["probab"].apply(
        lambda x: random.random() <= x)

    i = 0
    while i < len(affected_nodes):
        # for neighbors of a selected node
        for row in df_graph[df_graph["source"] ==
                            affected_nodes[i]].itertuples():
            observed_edges.append(row.Index)
            if row.activated and row.target not in affected_nodes:
                activated_edges.append(row.Index)
                affected_nodes.append(row.target)
        i += 1

    return np.array(affected_nodes), np.array(activated_edges), np.array(
        observed_edges)


def runIC_approach2(df_edges, df_nodes, seed_nodes):
    affected_nodes = deepcopy(seed_nodes)  # copy already selected nodes

    df_edges["activated"] = df_edges["probab"].apply(
        lambda x: random.random() <= x)
    df_edges["walked"] = False

    i = 0
    for cur_node in affected_nodes:
        for edge in df_nodes.loc[cur_node][0]:
            if df_edges.loc[edge, "target"] not in affected_nodes:
                affected_nodes.append(df_edges.loc[edge, "target"])
                df_edges.loc[edge, "walked"] = df_edges.loc[edge, "activated"]
        i += 1

    return np.array(affected_nodes)


def runIC(df_g, S, tracking=False):
    """ Runs independent cascade model.
    Input: df_g -- a dataframe representing the graph (with the probabilities)
    S -- initial set of vertices
    tracking -- whether we want to check for active/observed nodes
    Output: T -- resulted influenced set of vertices (including S)
    """
    T = deepcopy(S)  # copy already selected nodes
    act = []
    observed = []

    i = 0
    while i < len(T):
        # for neighbors of a selected node
        for row in df_g[df_g["source"] == T[i]].iterrows():
            observed.append(row[0])
            if random.random() <= row[1]["probab"]:
                act.append(row[0])
                T.append(row[1]["target"])
        i += 1
    # print(T)
    if tracking:
        return T, np.array(act), np.array(observed)

    return T


def oracle(df,
           num_nodes,
           num_edges,
           num_inf,
           epsilon,
           temp_dir="temp_dir",
           out_pattern=re.compile("Selected k SeedSet: (.+?) \\n")):
    """ Runs TIM (the oracle function).
    Input: df -- the graph to process
    num_inf -- the k the we are looking for
    epsilon -- hyperparameter
    Output: T -- The k highest influencers
    """
    df.to_csv(os.path.join(temp_dir, "graph_ic.inf"),
              index=False,
              sep=" ",
              header=False)
    # Preparing to run TIM
    with open(os.path.join(temp_dir, "attribute.txt"), "w+") as f:
        f.write(f"n={num_nodes}\nm={num_edges}")

    process = Popen(
        [
            "./tim",
            "-model",
            "IC",
            "-dataset",
            "temp_dir",
            "-k",
            f"{num_inf}",
            "-epsilon",
            f"{epsilon}",
        ],
        stdout=PIPE,
        stderr=PIPE,
    )
    (output, err) = process.communicate()
    _ = process.wait()  # Returns exit code
    out = output.decode("utf-8")
    logging.debug(f"Running TIM, {out}")
    return list(map(int, out_pattern.findall(out)[0].split(" ")))


def get_features_nodes(df_graph,
                       dims=20,
                       epochs=1,
                       node2vec_path="node2vec",
                       dataset_path="datasets",
                       check_existing=True):
    FNAME_IN = os.path.join(dataset_path, "df.edgelist")
    FNAME_OUT = os.path.join(dataset_path, f"df-d{dims}.emb")
    if check_existing and os.path.exists(FNAME_OUT):
        return pd.read_csv(FNAME_OUT,
                           sep=" ",
                           names=(["node"] +
                                  [f"feat_{i}" for i in range(dims)]),
                           skiprows=1)
    df_graph.to_csv(FNAME_IN,
                    index=False,
                    sep=" ",
                    header=False,
                    columns=["source", "target"])
    # Preparing to run node2vec
    process = Popen(
        [
            os.path.join(".", node2vec_path, "node2vec"),
            f"-i:{FNAME_IN}",
            f"-o:{FNAME_OUT}",
            f"-d:{dims}",
            f"-e:{epochs}",
            "-v",
        ],
        stdout=PIPE,
        stderr=PIPE,
    )
    (output, err) = process.communicate()
    _ = process.wait()  # Returns exit code
    out = output.decode("utf-8")
    logging.debug(f"Running node2vec, {out}")

    return pd.read_csv(FNAME_OUT,
                       sep=" ",
                       names=(["node"] + [f"feat_{i}" for i in range(dims)]),
                       skiprows=1)


def oim_node2vec(
    df,
    df_feats,
    num_inf=10,
    sigma=4,
    c=0.1,
    epsilon=0.4,
    num_repeats=30,
    num_repeats_regret=20,
    num_nodes_tim=-1,
    oracle=oracle,
):
    logging.debug("Started Online Influence Maximization...")
    logging.debug("Setting parameters")
    num_feats = df_feats.shape[1]
    num_edges_t = df.shape[0]

    # "True" probabilities - effectively our test set
    true_weights = df["probab"].copy()
    # Using nodes_t[-1] because TIM wants the max node id
    s_true = sorted(
        oracle(df[["source", "target", "probab"]], num_nodes_tim, num_edges_t,
               num_inf, epsilon))

    # b, M_inv - used by IMLinUCB
    b = np.zeros((num_feats, 1))
    m_inv = np.eye(num_feats, num_feats)

    # Returning these
    s_best = []
    reward_best = 0
    u_e_best = []
    regrets = []
    regrets_edges = []

    for iter_oim in tqdm(range(num_repeats),
                         desc=f"OIM iters {num_edges_t} edges",
                         leave=False,
                         file=sys.stderr):
        # ---- Step 1 - Calculating the u_e ----
        theta = (m_inv @ b) / (sigma * sigma)
        # xMx = (df_feats.values @ m_inv @ df_feats.T.values).clip(min=0)

        u_e = []
        for i in range(num_edges_t):
            x_e = df_feats.loc[i].values
            xMx = (x_e @ m_inv @ x_e.T)  # .clip(min=0)
            u_e.append(np.clip(x_e @ theta + c * np.sqrt(xMx), 0, 1))
        u_e = np.array(u_e)

        # ---- Step 2 - Evaluating the performance ----
        # Loss function
        df["probab"] = u_e
        s_oracle = sorted(
            oracle(df[["source", "target", "probab"]], num_nodes_tim,
                   num_edges_t, num_inf, epsilon))

        # Observing edge-level feedback
        df["probab"] = true_weights

        all_true_nodes = []
        all_true_edges = []
        all_true_obs = []
        all_algo_nodes = []
        all_algo_edges = []
        all_algo_obs = []
        for k in range(num_repeats_regret):
            true_act_nodes, true_act_edges, true_obs_edges = run_ic_eff(
                df, s_true)
            algo_act_nodes, algo_act_edges, algo_obs_edges = run_ic_eff(
                df, s_oracle)
            all_true_nodes.append(true_act_nodes)
            all_true_edges.append(true_act_edges)
            all_algo_nodes.append(algo_act_nodes)
            all_algo_edges.append(algo_act_edges)
            all_true_obs.append(true_obs_edges)
            all_algo_obs.append(algo_obs_edges)

        # True seed nodes
        all_true_nodes = np.unique(np.concatenate(all_true_nodes))
        all_true_edges = np.unique(np.concatenate(all_true_edges))
        all_true_obs = np.unique(np.concatenate(all_true_obs))
        # Algo seed nodes
        all_algo_nodes = np.unique(np.concatenate(all_algo_nodes))
        all_algo_edges = np.unique(np.concatenate(all_algo_edges))
        all_algo_obs = np.unique(np.concatenate(all_algo_obs))

        regrets.append(len(all_true_nodes) - len(all_algo_nodes))
        regrets_edges.append(len(all_true_edges) - len(all_algo_edges))

        logging.debug(f"True seeds: {s_true}")
        logging.debug(f"Algo   seeds: {s_oracle}")
        logging.debug(f"True reward: {len(all_true_nodes)}")
        logging.debug(f"Algo   reward: {len(all_algo_nodes)}")
        logging.debug(f"Best algo reward: {reward_best}")
        logging.debug(f"Regrets: {regrets}")
        logging.debug(f"Edge regrets: {regrets_edges}")
        logging.debug(
            f"Observed diff: {len(all_true_obs) - len(all_algo_obs)}")
        logging.debug(f"Algo weights {u_e[80:90]}".replace("\n", ""))
        logging.debug(f"Real weights {true_weights[80:90]}".replace("\n", ""))

        if len(all_algo_nodes) > reward_best:
            reward_best = len(all_algo_nodes)
            s_best = s_oracle
            u_e_best = u_e

        if reward_best > len(all_true_nodes):
            logging.debug(
                "The algorithm has achieved better reward than the true seed node set."
            )
            logging.debug("Stopping learning.")
            logging.debug(f"Best algo seed node set: {s_best}")
            return_dict = {
                "regrets": regrets,
                "regrets_edges": regrets_edges,
                "s_true": s_true,
                "s_best": s_best,
                "u_e_best": u_e_best,
                "reward_best": reward_best
            }
            logging.debug(f"Returning {return_dict}")
            return return_dict

        # ---- Step 3 - Calculating updates ----
        for i in all_algo_obs:
            x_e = np.array([df_feats.loc[i].values])
            m_inv -= (m_inv @ x_e.T @ x_e @ m_inv) / (x_e @ m_inv @ x_e.T +
                                                      sigma * sigma)
            b += x_e.T * int(i in all_algo_edges)

    return_dict = {
        "regrets": regrets,
        "regrets_edges": regrets_edges,
        "s_true": s_true,
        "s_best": s_best,
        "u_e_best": u_e_best,
        "reward_best": reward_best
    }
    logging.debug("The algorithm has finished running.")
    logging.debug(f"Returning: {return_dict}")
    return return_dict


# %% Finding the required directories

# TIM setup
TEMP_DIR = "temp_dir"
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

PATTERN = re.compile("Selected k SeedSet: (.+?) \\n")

# Datasets
DATASET_DIR = os.path.join("..", "Datasets")
if not os.path.exists(DATASET_DIR):
    print("Can't find the dataset directory!")
DATASET_FACEBOOK = os.path.join(DATASET_DIR, "fb-wosn-friends")

# %% setting up logging
VERBOSE = True
LOGGING_FMT = '%(levelname)s | %(asctime)s | line %(lineno)s | %(funcName)s | %(message)s'
LOGGING_DATEFMT = '%H:%M:%S'

logging_conf = partial(logging.basicConfig,
                       format=LOGGING_FMT,
                       datefmt=LOGGING_DATEFMT,
                       stream=sys.stdout)

if VERBOSE:
    logging_conf(level=logging.DEBUG)
else:
    logging_conf(level=logging.INFO)

# %% Running the code
random.seed(42)
logging.debug("Getting the graph...")
df_friend = pd.read_csv(
    os.path.join(DATASET_FACEBOOK, "fb-wosn-friends-clean.edges"),
    sep=" ",
    # ??? is probably weights, they are always 1
    names=["source", "target", "???", "timestamp"],
    skiprows=2,
)

# Processing the dataframe
logging.debug("Splitting the graph into time steps...")
df_friend["day"] = pd.to_datetime(df_friend["timestamp"],
                                  unit="s").dt.floor("d")
df_friend = df_friend.sort_values(by=["timestamp", "source", "target"])
df_friend = df_friend[["source", "target", "day"]]
nodes = np.sort(
    np.unique(np.hstack((df_friend["source"], df_friend["target"]))))
# Getting the true weights
# facebook df_friend['probab'] = np.random.uniform(0, 0.1, size=df_friend.shape[0])
logging.debug("Generating \"true\" activation probabilities...")
df_friend['probab'] = np.random.uniform(0, 1, size=df_friend.shape[0])

# %% OIM - NODE2VEC - Setup
NUM_FEATS = 20
# Getting node embeddings
logging.debug("Getting node embeddings...")
df_emb = get_features_nodes(df_friend,
                            dims=NUM_FEATS,
                            node2vec_path=os.getcwd(),
                            dataset_path=DATASET_FACEBOOK)
df_emb = df_emb.set_index("node").sort_values(by="node")
# Generating edge features
logging.debug(f"Generating {NUM_FEATS} edge features...")
df_feats = []
for row in tqdm(df_friend.itertuples()):
    df_feats.append(df_emb.loc[row.source].values *
                    df_emb.loc[row.target].values)
df_feats = pd.DataFrame(df_feats)

#%%
# Getting a graph of one time slice df_t
logging.debug("Creating df_t and df_feats_t")
t = np.sort(np.unique(df_friend["day"]))[55:56][-1]
df_t = df_friend[df_friend["day"] <= t].sort_values("source").reset_index()
df_feats_t = df_t["index"].apply(lambda x: df_feats.loc[x])

# Running the OIM algorithm
result_oim = oim_node2vec(df_t,
                          df_feats_t,
                          num_repeats_regret=5,
                          num_nodes_tim=nodes[-1])
