from functools import partial
import logging
import pandas as pd
import numpy as np
import scipy as sc
import scipy.sparse as sparse
import pickle
import re
import sys
import scipy.sparse.linalg
from tqdm import tqdm
from subprocess import Popen, PIPE
import os
import networkx as nx
import time
TEMP_DIR = "temp_dir"
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)
ATTRIBUTE_TXT = os.path.join(TEMP_DIR, "attribute.txt")
GRAPH_INF = os.path.join(TEMP_DIR, "graph_ic.inf")
PATTERN = re.compile("Selected k SeedSet: (.+?) \\n")

## %% setting up logging
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

# %% pq

import itertools
from heapq import *
from copy import deepcopy
from random import random


class PriorityQueue(object):
    def __init__(self):
        self.pq = []  # list of entries arranged in a heap
        self.entry_finder = {}  # mapping of tasks to entries
        self.REMOVED = "<removed-task>"  # placeholder for a removed task
        self.counter = itertools.count()  # unique sequence count

    def add_task(self, task, priority=0):
        "Add a new task or update the priority of an existing task"
        if task in self.entry_finder:
            self.remove_task(task)
        count = next(self.counter)
        entry = [priority, count, task]
        self.entry_finder[task] = entry
        heappush(self.pq, entry)

    def remove_task(self, task):
        "Mark an existing task as REMOVED.  Raise KeyError if not found."
        entry = self.entry_finder.pop(task)
        entry[-1] = self.REMOVED

    def pop_item(self):
        "Remove and return the lowest priority task. Raise KeyError if empty."
        while self.pq:
            priority, count, task = heappop(self.pq)
            if task is not self.REMOVED:
                del self.entry_finder[task]
                return task, priority
        raise KeyError("pop from an empty priority queue")

    def __str__(self):
        return str([entry for entry in self.pq if entry[2] != self.REMOVED])


# %% Functions


# def oracle(graph, num_inf, edge_prob, epsilon=0.1):
# return range(num_inf), [0.5] * graph
def runIC(df_g, S, tracking=False):
    """ Runs independent cascade model.
    Input: G -- networkx graph object
    S -- initial set of vertices
    p -- propagation probability
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
            if random() <= 1 - (1 - row[1]["probab"]):
                act.append(row[0])
                T.append(row[1]["target"])
        i += 1
    # print(T)
    if tracking:
        return T, np.array(act), np.array(observed)

    return T


# a,b,c = runIC(df_t, s_oracle, True)

# df_t["act"] = random() <= 1 - (1 - df_t["probab"])

# df_g[(df_g["source"] == T[i]) & (df_g["target"] == v)]
# s_oracle = oracle_greedy(df_t, nodes_t, num_inf)


def oracle_greedy(df_g, nodes, k, num_repeats=20):
    S = []  # set of selected nodes
    # add node to S if achieves maximum propagation for current chosen + this node
    for i in range(k):
        s = PriorityQueue()  # priority queue
        for v in nodes:
            if v not in S:
                s.add_task(v, 0)  # initialize spread value
                for j in range(num_repeats):  # run R times Random Cascade
                    # print("Running random cascade...")
                    [priority, count, task] = s.entry_finder[v]
                    s.add_task(v, priority - float(len(runIC(df_g, S + [v]))) /
                               num_repeats)  # add normalized spread value
        task, priority = s.pop_item()
        S.append(task)
    return S


def oracle(df_t, num_inf, epsilon):
    df_t.to_csv(GRAPH_INF, index=False, sep=" ", header=False)
    # Preparing to run TIM
    with open(ATTRIBUTE_TXT, "w+") as f:
        f.write(ATTRIBUTE_STR.format(df_t.shape[0]))

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
    exit_code = process.wait()
    out = output.decode("utf-8")
    logging.debug(f"Running TIM, {out}")
    return list(map(int, PATTERN.findall(out)[0].split(" ")))


def proj(x):
    # Using the sigmoid function to project x onto [0,1]
    # sigm(x) = 1 / (1 + np.exp(-x))
    return 1 / (1 + np.exp(-x))


def im_rec(graph,
           num_inf,
           oracle,
           x,
           sigma,
           c,
           b_last,
           m_last,
           num_repeats,
           epsilon=0.1):
    # m_t = Mt^-1
    m = m_last
    b = b_last
    # x_i: feature matrix, rows = feature vectors of all observed edges in i steps
    # y_i: binary column vector encoding edge realizations in t steps
    for i in range(num_repeats):
        # Step 1
        theta = m * b / (sigma**2)
        xMx = np.transpose(x) * m * x
        upper_bound = proj(x * theta + c * np.sqrt(xMx))
        # Step 2
        source_nodes = oracle(graph, num_inf, upper_bound)
        # Step 3
        m -= (m * x * np.transpose(x) * m) / (xMx + sigma**2)
        b += x * y


def im(graph, num_inf, oracle, x_e, sigma, c, num_repeats, epsilon=0.1):
    pass


# %% Running the code

df_friend = pd.read_csv(
    "fb-wosn-friends/fb-wosn-friends-clean.edges",
    sep=" ",
    names=["source", "target", "???", "timestamp"],
    skiprows=2,
)

df_friend["day"] = pd.to_datetime(df_friend["timestamp"],
                                  unit="s").dt.floor("d")
df_friend = df_friend.sort_values(by=["timestamp", "source", "target"])
df_friend = df_friend[["source", "target", "day"]]
nodes = np.sort(
    np.unique(np.hstack((df_friend["source"], df_friend["target"]))))
df_friend.shape
# Getting the true weights
df_friend['probab'] = np.random.uniform(0, 0.1, size=df_friend.shape[0])
ATTRIBUTE_STR = f"n={nodes[-1]}\n" "m={}"
# %% Testing sparse matrices

# %% TIMLinUCB
# Datastructure from group by:
# list of [df_name, df]
num_inf = 20
num_repeats = 5
num_repeats_regret = 5
epsilon = 0.4
sigma = 4
c = 0.1
results = []
PRESERVE_PARAMS = False
FOLDER_NAME = f"eps{epsilon}-pparams{PRESERVE_PARAMS}"
if not os.path.exists(FOLDER_NAME):
    os.makedirs(FOLDER_NAME)
PICKLE_NAME = f"results_eps{epsilon}.pkl"
PRESERVE_PARAMS = False

# for t in tqdm(range(len(list(df_friend.groupby("day")))),
#               desc="Iterating through the dyn net"):
# kkk = 0
# for i in np.sort(np.unique(df_friend["day"])):
#     print(df_friend[df_friend["day"] <= i])
#     kkk += 1
#     if not kkk % 3:
#         break
# pd.concat([list(df_friend.groupby("day"))][0:3])
# %% main loop
df_t = df_friend[df_friend["day"] <= np.sort(np.unique(df_friend["day"]))
                 [0]].drop_duplicates().sort_values("source").reset_index()
# df_t[df_t["source"] == 20440]
# df_t[df_t["target"] == 20440]
for t in tqdm(np.sort(np.unique(df_friend["day"]))[50:60],
              desc="Iterating through the dyn net",
              file=sys.stderr):
    t_start = time.time()

    logging.debug("Entered the time loop")
    # df_t[df_t["index"].isin(df_t1["index"])].index
    # m_inv[]
    df_t = df_friend[df_friend["day"] <= t].drop_duplicates().sort_values(
        "source").reset_index()
    nodes_t = np.sort(np.unique(np.hstack((df_t["source"], df_t["target"]))))
    num_nodes_t = len(nodes_t)
    num_edges_t = df_t.shape[0]
    true_weights = df_t["probab"].copy()

    # "Ground truth" graph
    logging.debug("Getting the true seed nodes")
    logging.debug(df_t)
    s_true = oracle(df_t[["source", "target", "probab"]], num_inf, epsilon)
    logging.debug("Got the true seed nodes")

    # Starting the algo
    logging.debug("------------------------------------------")
    logging.debug(f"Testing sigma = {sigma}, c = {c}")

    regrets = []
    regrets_edges = []
    logging.debug("Setting m_inv, x and b")
    m_inv = sparse.eye(num_edges_t, format="csr")
    x = sparse.eye(num_edges_t, format="csr")

    if PRESERVE_PARAMS:
        try:
            # Using the t-1 results to our advantage
            logging.debug("Using the preserved params from the last run...")
            # past_ind = df_t["index"].isin(df_t1["index"])
            past_ind_rev = df_t1.index[df_t1["index"].isin(df_t["index"])]
            # b1.data
            past_ind1 = df_t.index[df_t["index"].isin(df_t1["index"])]
            m_inv.data[past_ind1] = m_inv1.data
            row = past_ind1[b1.nonzero()[0]].values
            data = b1[past_ind_rev].data
            col = np.zeros(data.shape[0])

            b = sc.sparse.csr_matrix((data, (row, col)),
                                     shape=(num_edges_t, 1))
            # b.data[past_ind] = b1.data
        except Exception as e:
            logging.debug(e)
            # logging.debug("Initial run, skipping m_inv1 b_1")
            b = sc.sparse.csr_matrix((num_edges_t, 1))
    else:
        b = sc.sparse.csr_matrix((num_edges_t, 1))

    for i in tqdm(range(num_repeats),
                  desc=f"OIM iters {num_edges_t} edges",
                  leave=False,
                  file=sys.stderr):
        logging.debug("Entered the inside loop")
        # Step 1 - Calculate the upper confidence bound Ut
        theta = (m_inv @ b) / (sigma * sigma)
        x_th = x * theta
        xMx = sparse.csr_matrix((x.T @ m_inv @ x).diagonal()).T
        # Sigmoid to make sure u_e is [0..1]
        u_e = 1 / (2 + (x_th + c * np.sqrt(xMx)).expm1().toarray())
        u_e = np.nan_to_num(u_e)
        # Step 2 - Get the edge-level semi-bandit feedback
        # source_nodes, y = oracle(graph, num_inf, upper_bound)
        # TIM algorithm
        # Temporary solution: run TIM
        # Create temp files in the folder temp_dir:
        # attribute.txt: n = <num nodes> \n m = <num_edges>
        # graph_ic.inf node1, node2, act probab
        # u_e.shape
        # oracle(df_t, u_e)

        # s_oracle = oracle_greedy(df_t, nodes, num_inf)
        df_t["probab"] = u_e
        # logging.debug(df_t["probab"])
        s_oracle = oracle(df_t[["source", "target", "probab"]], num_inf,
                          epsilon)

        # Observing edge-level feedback
        df_t["probab"] = true_weights

        # logging.debug(df_t["probab"])
        true_inf_nodes, true_act, true_obs = runIC(df_t, s_true, True)
        true_inf_edges = len(true_act)
        ue_inf_nodes, ue_act, ue_obs = runIC(df_t, s_oracle, True)
        ue_inf_edges = len(ue_act)
        regrets.append(len(true_inf_nodes) - len(ue_inf_nodes))
        regrets_edges.append(true_inf_edges - ue_inf_edges)

        logging.debug(f"True nodes: {true_inf_nodes}")
        logging.debug(f"Ue   nodes: {ue_inf_nodes}")
        logging.debug(f"Regrets: {regrets}")
        logging.debug(f"Edge regrets: {regrets_edges}")
        # Step 3 - Update M_inv and B
        # Getting the observed edges' weights
        x_t = x[ue_obs, :]
        # Getting the observed edges' realizations and updating b
        y_t = sparse.csr_matrix(np.isin(ue_obs, ue_act).astype(int)).T
        b += x_t.T * y_t
        # Updating m_inv
        # sparse.csr_matrix((m_inv @ edge.T @ edge @ m_inv) / ((edge @ m_inv @ edge.T).data + sigma**2)).nonzero()
        for edge in x[ue_obs, :]:
            m_inv[edge.data,
                  edge.data] -= (m_inv @ edge.T @ edge @ m_inv).data / (
                      (edge @ m_inv @ edge.T).data + sigma**2)
        m_inv.eliminate_zeros()

    logging.debug("Checking that the weights are true {}".format(
        np.all(df_t["probab"] == true_weights)))

    regret_t = np.average([
        len(runIC(df_t, s_true, False)) - len(runIC(df_t, s_oracle, False))
        for _ in range(num_repeats_regret)
    ])

    logging.debug(f"Regret at time t is {regret_t}")

    if PRESERVE_PARAMS:
        # Preserving the data for the next timestamp
        df_t1 = df_t
        m_inv1 = m_inv
        b1 = b

    results.append({
        "regrets": regrets,
        "regrets_edges": regrets_edges,
        "regret_t": regret_t,
        "best_set_oim": s_oracle,
        "best_set_truth": s_true,
        "u_e_last": u_e,
        "time": t,
        "exec_time": time.time() - t_start
    })
    with open(os.path.join(FOLDER_NAME, f"{t}.pkl"), "wb") as f:
        pickle.dump({"m_inv": m_inv, "df": df_t, "b": b}, f)

# Saving results
with open(os.path.join(FOLDER_NAME, PICKLE_NAME), "wb") as f:
    pickle.dump(results, f)

# %%
#
# x = np.matrix([[3, 0, 0], [0, 5, 0], [0, 0, 7]])  # np.eye(3)  #n
# m = np.matrix([[3, 0, 0], [0, 5, 0], [0, 0, 7]])
# print(f"x = \n{x}")
# print(f"m = \n{m}")
# x.T * x
# m * x * x.T * m
# # # %%
# # # %%timeit
# # #
# # for i in range(3):
# #     print(m * x[i].T * x[i] * m)
# # # # %%
# # # %%timeit
# # # This only works for the tabular case - i.e. X0 = I and only the main diagonal of M^-1
# # # is updated
# # np.diag(m*x*x.T*m)
# b = 1 + 1
# f"this is the value of the variable b: {b}"
# 8 * 56
# import pandas as pd
#
# fb_df = pd.read_csv(
#     "fb-wosn-friends/fb-wosn-friends.edges",
#     sep=" ",
#     names=["node1", "node2", "???", "timestamp"],
#     skiprows=2,
# )
# fb_df

# %% Testing metaparams
# Datastructure from group by:
# list of [df_name, df]
num_inf = 10
num_repeats = 150
num_repeats_regret = 5
epsilon = 0.4
sigma_arr = [4]  # 5, 8, 10, 100]
c_arr = [0.1]  #[0.01, 0.1, 1, 5, 10, 100]
results = []
FOLDER_NAME = f"eps{epsilon}-csigmatest4"
if not os.path.exists(FOLDER_NAME):
    os.makedirs(FOLDER_NAME)
PRESERVE_PARAMS = False

# for t in tqdm(range(len(list(df_friend.groupby("day")))),
#               desc="Iterating through the dyn net"):
# kkk = 0
# for i in np.sort(np.unique(df_friend["day"])):
#     print(df_friend[df_friend["day"] <= i])
#     kkk += 1
#     if not kkk % 3:
#         break
# pd.concat([list(df_friend.groupby("day"))][0:3])
# %% main loop
df_t = df_friend[df_friend["day"] <= np.sort(np.unique(df_friend["day"]))
                 [0]].drop_duplicates().sort_values("source").reset_index()
# df_t[df_t["source"] == 20440]
# df_t[df_t["target"] == 20440]
for sigma in sigma_arr:
    for c in c_arr:
        PICKLE_NAME = f"results_eps{epsilon}_sigma{sigma}_c{c}.pkl"
        for t in tqdm(np.sort(np.unique(df_friend["day"]))[50:51],
                      desc="Iterating through the dyn net",
                      file=sys.stdout):
            t_start = time.time()

            logging.debug("Entered the time loop")
            # df_t[df_t["index"].isin(df_t1["index"])].index
            # m_inv[]
            df_t = df_friend[df_friend["day"] <= t].drop_duplicates(
            ).sort_values("source").reset_index()
            nodes_t = np.sort(
                np.unique(np.hstack((df_t["source"], df_t["target"]))))
            num_nodes_t = len(nodes_t)
            num_edges_t = df_t.shape[0]
            true_weights = df_t["probab"].copy()

            # "Ground truth" graph
            logging.debug("Getting the true seed nodes")
            logging.debug(df_t)
            s_true = oracle(df_t[["source", "target", "probab"]], num_inf,
                            epsilon)
            logging.debug("Got the true seed nodes")

            # Starting the algo
            logging.debug("------------------------------------------")
            logging.debug(f"Testing sigma = {sigma}, c = {c}")

            regrets = []
            regrets_edges = []
            logging.debug("Setting m_inv, x and b")
            m_inv = sparse.eye(num_edges_t, format="csr")
            x = sparse.eye(num_edges_t, format="csr")

            if PRESERVE_PARAMS:
                try:
                    # Using the t-1 results to our advantage
                    logging.debug(
                        "Using the preserved params from the last run...")
                    # past_ind = df_t["index"].isin(df_t1["index"])
                    past_ind_rev = df_t1.index[df_t1["index"].isin(
                        df_t["index"])]
                    # b1.data
                    past_ind1 = df_t.index[df_t["index"].isin(df_t1["index"])]
                    m_inv.data[past_ind1] = m_inv1.data
                    row = past_ind1[b1.nonzero()[0]].values
                    data = b1[past_ind_rev].data
                    col = np.zeros(data.shape[0])

                    b = sc.sparse.csr_matrix((data, (row, col)),
                                             shape=(num_edges_t, 1))
                    # b.data[past_ind] = b1.data
                except Exception as e:
                    logging.debug(e)
                    # logging.debug("Initial run, skipping m_inv1 b_1")
                    b = sc.sparse.csr_matrix((num_edges_t, 1))
            else:
                b = sc.sparse.csr_matrix((num_edges_t, 1))

            for i in tqdm(range(num_repeats),
                          desc=f"OIM iters {num_edges_t} edges",
                          leave=False,
                          file=sys.stderr):
                logging.debug("Entered the inside loop")
                # Step 1 - Calculate the upper confidence bound Ut
                theta = (m_inv @ b) / (sigma * sigma)
                x_th = x * theta
                # x_th.data
                xMx = sparse.csr_matrix((x.T @ m_inv @ x).diagonal()).T
                # Sigmoid to make sure u_e is [0..1]
                # u_e = 1 / (2 + (-1 *
                #                 (x_th + c * np.sqrt(xMx))).expm1().toarray())
                u_e = np.clip((x_th + c * np.sqrt(xMx)).toarray(), 0, 1)
                # u_e[86]
                # true_weights[86]

                # u_e = np.nan_to_num(u_e)
                # Step 2 - Get the edge-level semi-bandit feedback
                # source_nodes, y = oracle(graph, num_inf, upper_bound)
                # TIM algorithm
                # Temporary solution: run TIM
                # Create temp files in the folder temp_dir:
                # attribute.txt: n = <num nodes> \n m = <num_edges>
                # graph_ic.inf node1, node2, act probab
                # u_e.shape
                # oracle(df_t, u_e)

                # s_oracle = oracle_greedy(df_t, nodes, num_inf)
                df_t["probab"] = u_e
                # logging.debug(df_t["probab"])
                s_oracle = oracle(df_t[["source", "target", "probab"]],
                                  num_inf, epsilon)

                # Observing edge-level feedback
                df_t["probab"] = true_weights

                # logging.debug(df_t["probab"])
                true_inf_nodes, true_act, true_obs = runIC(df_t, s_true, True)
                true_inf_edges = len(true_act)
                ue_inf_nodes, ue_act, ue_obs = runIC(df_t, s_oracle, True)
                ue_inf_edges = len(ue_act)
                regrets.append(len(true_inf_nodes) - len(ue_inf_nodes))
                regrets_edges.append(true_inf_edges - ue_inf_edges)

                logging.debug(f"True nodes: {true_inf_nodes}")
                logging.debug(f"Ue   nodes: {ue_inf_nodes}")
                logging.debug(f"Regrets: {regrets}")
                logging.debug(f"Edge regrets: {regrets_edges}")

                logging.debug(f"Alg % {u_e[80:90]}".replace("\n", ""))
                logging.debug(f"Real % {true_weights[80:90]}".replace(
                    "\n", ""))
                # Step 3 - Update M_inv and B
                # Getting the observed edges' weights
                x_t = x[ue_obs, :]
                # Getting the observed edges' realizations and updating b
                y_t = sparse.csr_matrix(np.isin(ue_obs, ue_act).astype(int)).T
                b += x_t.T * y_t
                # Updating m_inv
                # sparse.csr_matrix((m_inv @ edge.T @ edge @ m_inv) / ((edge @ m_inv @ edge.T).data + sigma**2)).nonzero()
                for edge in x[ue_obs, :]:
                    m_inv[edge.indices[0], edge.indices[0]] -= (
                        m_inv @ edge.T @ edge @ m_inv).data / (
                            (edge @ m_inv @ edge.T).data + sigma**2)
                m_inv.eliminate_zeros()
            logging.debug("Checking that the weights are true {}".format(
                np.all(df_t["probab"] == true_weights)))

            regret_t = np.average([
                len(runIC(df_t, s_true, False)) -
                len(runIC(df_t, s_oracle, False))
                for _ in range(num_repeats_regret)
            ])

            logging.debug(f"Regret at time t is {regret_t}")

            if PRESERVE_PARAMS:
                # Preserving the data for the next timestamp
                df_t1 = df_t
                m_inv1 = m_inv
                b1 = b

            results.append({
                "sigma": sigma,
                "c": c,
                "regrets": regrets,
                "regrets_edges": regrets_edges,
                "regret_t": regret_t,
                "best_set_oim": s_oracle,
                "best_set_truth": s_true,
                "u_e_last": u_e,
                "time": t,
                "exec_time": time.time() - t_start
            })

            logging.debug(results)
            with open(os.path.join(FOLDER_NAME, f"{t}.pkl"), "wb") as f:
                pickle.dump({"m_inv": m_inv, "df": df_t, "b": b}, f)

        # Saving results
        with open(os.path.join(FOLDER_NAME, PICKLE_NAME), "wb") as f:
            pickle.dump(results, f)
