#! /usr/local/bin/python3

from functools import partial
import logging
import pandas as pd
import numpy as np
import sys
from tqdm import tqdm
from subprocess import Popen, PIPE
import os
import matplotlib.pyplot as plt
import random
from helpers import run_ic_eff, tim


# --------------------------------------------------------------------------------------
# %% ------------------------------ Initial setup --------------------------------------
# --------------------------------------------------------------------------------------

# Fancy plots in matplotlib and pandas
random.seed(42)
plt.style.use("ggplot")

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
# %% -------------------- Generating edge features (preprocessing) ---------------------
# --------------------------------------------------------------------------------------


def get_features_nodes(
    df_graph,
    dims=20,
    epochs=1,
    node2vec_path="node2vec",
    tempdir_name="temp_dir",
    dataset_name="facebook",
    check_existing=True,
):
    if not os.path.exists(tempdir_name):
        os.makedirs(tempdir_name)

    FNAME_IN = os.path.join(tempdir_name, f"{dataset_name}.edgelist")
    FNAME_OUT = os.path.join(tempdir_name, f"{dataset_name}-d{dims}.emb")

    # Checking if we already ran the function before
    if check_existing and os.path.exists(FNAME_OUT):
        return pd.read_csv(
            FNAME_OUT,
            sep=" ",
            names=(["node"] + [f"feat_{i}" for i in range(dims)]),
            skiprows=1,
        )
    # Saving the edgelist to run node2vec on
    df_graph.to_csv(
        FNAME_IN, index=False, sep=" ", header=False, columns=["source", "target"]
    )
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

    return pd.read_csv(
        FNAME_OUT,
        sep=" ",
        names=(["node"] + [f"feat_{i}" for i in range(dims)]),
        skiprows=1,
    )


def generate_node2vec_fetures(
    df,
    node2vec_path=os.getcwd(),
    tempdir_name="temp_dir",
    dataset_name="facebook",
    num_features=20,
    check_existing=True,
):
    # Checking if we did this before
    FNAME_SAVE = os.path.join(tempdir_name, f"{dataset_name}-d{num_features}-edges.emb")
    if check_existing and os.path.exists(FNAME_SAVE):
        return pd.read_csv(FNAME_SAVE, index_col=0)

    # Getting node embeddings
    logging.debug("Getting node embeddings...")
    df_emb = get_features_nodes(
        df,
        dims=num_features,
        node2vec_path=node2vec_path,
        tempdir_name=tempdir_name,
        dataset_name=dataset_name,
        check_existing=check_existing,
    )
    df_emb = df_emb.set_index("node").sort_values(by="node")

    # Generating edge features
    logging.debug(f"Generating {num_features} edge features...")
    df_feats = []
    for row in tqdm(df.itertuples(), total=df.shape[0]):
        df_feats.append(df_emb.loc[row.source].values * df_emb.loc[row.target].values)
    df_feats = pd.DataFrame(df_feats)

    # Saving the results
    logging.debug(f"Saving the edge features to {FNAME_SAVE}")
    df_feats.to_csv(FNAME_SAVE)

    return df_feats


# --------------------------------------------------------------------------------------
# %% ---------------------------------  Online IM --------------------------------------
# --------------------------------------------------------------------------------------


def oim_node2vec_test(
    df,
    df_feats,
    num_inf=10,
    sigma=4,
    c=0.1,
    epsilon=0.4,
    num_repeats=30,
    num_repeats_regret_algo=20,
    num_repeats_regret_true=20,
    num_nodes_tim=-1,
    oracle=tim,
):
    logging.debug("Started Online Influence Maximization...")
    logging.debug("Setting parameters")
    num_feats = df_feats.shape[1]
    num_edges_t = df.shape[0]

    # "True" probabilities - effectively our test set
    true_weights = df["probab"].copy()
    # Using nodes_t[-1] because TIM wants the max node id
    s_true = sorted(
        oracle(
            df[["source", "target", "probab"]],
            num_nodes_tim,
            num_edges_t,
            num_inf,
            epsilon,
        )
    )
    # Gathering the stats for the "true" seed set
    all_true_nodes = []
    all_true_edges = []
    all_true_obs = []
    for k in range(num_repeats_regret_true):
        true_act_nodes, true_act_edges, true_obs_edges = run_ic_eff(df, s_true)
        all_true_nodes.append(true_act_nodes)
        all_true_edges.append(true_act_edges)
        all_true_obs.append(true_obs_edges)

    # Means for nodes and activated edges
    mean_true_nodes = np.mean([len(i) for i in all_true_nodes])
    mean_true_edges = np.mean([len(i) for i in all_true_edges])
    mean_true_obs = np.mean([len(i) for i in all_true_obs])

    # b, M_inv - used by IMLinUCB
    b = np.zeros((num_feats, 1))
    m_inv = np.eye(num_feats, num_feats)

    # Returning these
    s_best = []
    reward_best = 0
    u_e_best = []
    regrets = []
    regrets_edges = []

    for iter_oim in tqdm(
        range(num_repeats),
        desc=f"OIM iters {num_edges_t} edges",
        leave=False,
        file=sys.stderr,
    ):
        # ---- Step 1 - Calculating the u_e ----
        theta = (m_inv @ b) / (sigma * sigma)
        # xMx = (df_feats.values @ m_inv @ df_feats.T.values).clip(min=0)

        u_e = []
        for i in range(num_edges_t):
            x_e = df_feats.loc[i].values
            xMx = x_e @ m_inv @ x_e.T  # .clip(min=0)
            u_e.append(np.clip(x_e @ theta + c * np.sqrt(xMx), 0, 1))
            # u_e.append(expit(x_e @ theta + c * np.sqrt(xMx)))

        u_e = np.array(u_e)

        # ---- Step 2 - Evaluating the performance ----
        # Loss function
        df["probab"] = u_e
        s_oracle = sorted(
            oracle(
                df[["source", "target", "probab"]],
                num_nodes_tim,
                num_edges_t,
                num_inf,
                epsilon,
            )
        )

        # Observing edge-level feedback
        df["probab"] = true_weights

        all_algo_nodes = []
        all_algo_edges = []
        all_algo_obs = []
        for k in range(num_repeats_regret_algo):
            algo_act_nodes, algo_act_edges, algo_obs_edges = run_ic_eff(df, s_oracle)
            all_algo_nodes.append(algo_act_nodes)
            all_algo_edges.append(algo_act_edges)
            all_algo_obs.append(algo_obs_edges)

        # Mean node counts
        mean_algo_nodes = np.mean([len(i) for i in all_algo_nodes])
        # Mean activated edge counts
        mean_algo_edges = np.mean([len(i) for i in all_algo_edges])
        mean_algo_obs = np.mean([len(i) for i in all_algo_obs])

        # Used for updating M and b later
        all_algo_edges = np.unique(np.concatenate(all_algo_edges))
        all_algo_obs = np.unique(np.concatenate(all_algo_obs))

        regrets.append(mean_true_nodes - mean_algo_nodes)
        regrets_edges.append(mean_true_edges - mean_algo_edges)

        logging.debug(f"True seeds: {s_true}")
        logging.debug(f"Algo   seeds: {s_oracle}")
        logging.debug(
            "Diff between true and algo seeds: "
            f"{len(np.setdiff1d(s_true, s_oracle))}"
        )
        logging.debug(f"True reward: {mean_true_nodes}")
        logging.debug(f"Algo   reward: {mean_algo_nodes}")
        logging.debug(f"Best algo reward: {reward_best}")
        logging.debug(f"Regrets: {regrets}")
        logging.debug(f"Edge regrets: {regrets_edges}")
        logging.debug(f"Observed diff: {mean_true_obs - mean_algo_obs}")
        logging.debug(f"Algo weights {u_e[80:90]}".replace("\n", ""))
        logging.debug(f"Real weights {true_weights[80:90]}".replace("\n", ""))

        if mean_algo_nodes > reward_best:
            reward_best = mean_algo_nodes
            s_best = s_oracle
            u_e_best = u_e

        if mean_algo_nodes > mean_true_nodes:
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
                "reward_best": reward_best,
            }
            logging.debug(f"Returning {return_dict}")
            return return_dict

        # ---- Step 3 - Calculating updates ----
        for i in all_algo_obs:
            x_e = np.array([df_feats.loc[i].values])
            m_inv -= (m_inv @ x_e.T @ x_e @ m_inv) / (
                x_e @ m_inv @ x_e.T + sigma * sigma
            )
            b += x_e.T * int(i in all_algo_edges)

    return_dict = {
        "regrets": regrets,
        "regrets_edges": regrets_edges,
        "s_true": s_true,
        "s_best": s_best,
        "u_e_best": u_e_best,
        "reward_best": reward_best,
    }
    logging.debug("The algorithm has finished running.")
    logging.debug(f"Returning: {return_dict}")
    return return_dict


def oim_node2vec_simple(
    df,
    df_feats,
    num_inf=10,
    sigma=4,
    c=0.1,
    epsilon=0.4,
    num_repeats=15,
    num_nodes_tim=-1,
    oracle=tim,
):
    logging.debug("Started Online Influence Maximization...")
    logging.debug("Setting parameters")
    num_feats = df_feats.shape[1]
    num_edges_t = df.shape[0]

    # "True" probabilities - effectively our test set
    true_weights = df["probab"].copy()

    # b, M_inv - used by IMLinUCB
    b = np.zeros((num_feats, 1))
    m_inv = np.eye(num_feats, num_feats)

    # Returning these
    s_best = []
    reward_best = 0
    u_e_best = []
    rewards = []
    rewards_edges = []

    for iter_oim in tqdm(
        range(num_repeats),
        desc=f"OIM iters {num_edges_t} edges",
        leave=False,
        file=sys.stderr,
    ):
        # ---- Step 1 - Calculating the u_e ----
        theta = (m_inv @ b) / (sigma * sigma)
        # xMx = (df_feats.values @ m_inv @ df_feats.T.values).clip(min=0)

        u_e = []
        for i in range(num_edges_t):
            x_e = df_feats.loc[i].values
            xMx = x_e @ m_inv @ x_e.T  # .clip(min=0)
            u_e.append(np.clip(x_e @ theta + c * np.sqrt(xMx), 0, 1))
            # u_e.append(expit(x_e @ theta + c * np.sqrt(xMx)))

        u_e = np.array(u_e)

        # ---- Step 2 - Evaluating the performance ----
        # Loss function
        df["probab"] = u_e
        s_oracle = sorted(
            oracle(
                df[["source", "target", "probab"]],
                num_nodes_tim,
                num_edges_t,
                num_inf,
                epsilon,
            )
        )

        # Observing edge-level feedback
        df["probab"] = true_weights

        algo_act_nodes, algo_act_edges, algo_obs_edges = run_ic_eff(df, s_oracle)

        algo_num_nodes = len(algo_act_nodes)
        algo_num_edges = len(algo_act_edges)

        rewards.append(algo_num_nodes)
        rewards_edges.append(algo_num_edges)

        logging.debug(f"Algo   seeds: {s_oracle}")
        logging.debug(f"Algo   reward: {algo_num_nodes}")
        logging.debug(f"Best algo reward: {reward_best}")
        logging.debug(f"Rewards: {rewards}")
        logging.debug(f"Edge rewards: {rewards_edges}")
        logging.debug(f"Algo weights {u_e[80:90]}".replace("\n", ""))
        logging.debug(f"Real weights {true_weights[80:90]}".replace("\n", ""))

        if algo_num_nodes > reward_best:
            reward_best = algo_num_nodes
            s_best = s_oracle
            u_e_best = u_e

        # ---- Step 3 - Calculating updates ----
        for i in algo_obs_edges:
            x_e = np.array([df_feats.loc[i].values])
            m_inv -= (m_inv @ x_e.T @ x_e @ m_inv) / (
                x_e @ m_inv @ x_e.T + sigma * sigma
            )
            b += x_e.T * int(i in algo_act_edges)

    return_dict = {
        "rewards": rewards,
        "rewards_edges": rewards_edges,
        "s_best": s_best,
        "u_e_best": u_e_best,
        "reward_best": reward_best,
    }
    logging.debug("The algorithm has finished running.")
    logging.debug(f"Returning: {return_dict}")
    return return_dict


def oim_node2vec(
    df,
    df_feats,
    nodes,
    num_inf=10,
    sigma=4,
    c=0.1,
    epsilon=0.4,
    num_repeats=30,
    num_repeats_reward=20,
    oracle=tim,
):
    logging.debug("Started Online Influence Maximization...")
    logging.debug("Setting parameters")
    num_feats = df_feats.shape[1]
    num_edges_t = df.shape[0]
    num_nodes_tim = nodes[-1]
    # "True" probabilities - effectively our test set
    true_weights = df["probab"].copy()

    # b, M_inv - used by IMLinUCB
    b = np.zeros((num_feats, 1))
    m_inv = np.eye(num_feats, num_feats)

    # Returning these
    s_best = []
    reward_best = 0
    u_e_best = []

    for iter_oim in tqdm(
        range(num_repeats),
        desc=f"OIM iters {num_edges_t} edges",
        leave=False,
        file=sys.stderr,
    ):
        # ---- Step 1 - Calculating the u_e ----
        theta = (m_inv @ b) / (sigma * sigma)
        # xMx = (df_feats.values @ m_inv @ df_feats.T.values).clip(min=0)

        u_e = []
        for i in range(num_edges_t):
            x_e = df_feats.loc[i].values
            xMx = x_e @ m_inv @ x_e.T  # .clip(min=0)
            u_e.append(np.clip(x_e @ theta + c * np.sqrt(xMx), 0, 1))
            # u_e.append(expit(x_e @ theta + c * np.sqrt(xMx)))

        u_e = np.array(u_e)

        # ---- Step 2 - Evaluating the performance ----
        # Loss function
        df["probab"] = u_e
        s_oracle = sorted(
            oracle(
                df[["source", "target", "probab"]],
                num_nodes_tim,
                num_edges_t,
                num_inf,
                epsilon,
            )
        )

        # Observing edge-level feedback
        df["probab"] = true_weights

        all_algo_nodes = []
        all_algo_edges = []
        all_algo_obs = []
        for k in range(num_repeats_reward):
            algo_act_nodes, algo_act_edges, algo_obs_edges = run_ic_eff(df, s_oracle)
            all_algo_nodes.append(algo_act_nodes)
            all_algo_edges.append(algo_act_edges)
            all_algo_obs.append(algo_obs_edges)

        # Mean node counts
        mean_algo_nodes = np.mean([len(i) for i in all_algo_nodes])

        # Used for updating M and b later
        all_algo_edges = np.unique(np.concatenate(all_algo_edges))
        all_algo_obs = np.unique(np.concatenate(all_algo_obs))

        logging.debug(f"Algo   seeds: {s_oracle}")
        logging.debug(f"Algo   reward: {mean_algo_nodes}")
        logging.debug(f"Best algo reward: {reward_best}")
        logging.debug(f"Algo weights {u_e[80:90]}".replace("\n", ""))
        logging.debug(f"Real weights {true_weights[80:90]}".replace("\n", ""))

        if mean_algo_nodes > reward_best:
            reward_best = mean_algo_nodes
            s_best = s_oracle
            u_e_best = u_e

        # ---- Step 3 - Calculating updates ----
        for i in all_algo_obs:
            x_e = np.array([df_feats.loc[i].values])
            m_inv -= (m_inv @ x_e.T @ x_e @ m_inv) / (
                x_e @ m_inv @ x_e.T + sigma * sigma
            )
            b += x_e.T * int(i in all_algo_edges)

    return_dict = {"s_best": s_best, "u_e_best": u_e_best, "reward_best": reward_best}
    logging.debug("The algorithm has finished running.")
    logging.debug(f"Returning: {return_dict}")
    return return_dict


# --------------------------------------------------------------------------------------
# %% ------------------------------ Temporal Online IM ---------------------------------
# --------------------------------------------------------------------------------------


def timlinucb(
    df_edges,
    df_feats,
    times,
    nodes,
    num_seeds=5,
    sigma=4,
    c=0.1,
    epsilon=0.4,
    num_repeats_oim=10,
    num_repeats_oim_reward=10,
):
    results = []
    for t in tqdm(times):
        df_t = df_edges[df_edges["day"] <= t].sort_values("source").reset_index()
        df_feats_t = df_t["index"].apply(lambda x: df_feats.loc[x])
        result_oim = oim_node2vec(
            df_t,
            df_feats_t,
            nodes,
            num_inf=num_seeds,
            sigma=sigma,
            c=c,
            epsilon=epsilon,
            num_repeats=num_repeats_oim,
            num_repeats_reward=num_repeats_oim_reward,
        )
        result_oim["time"] = t
        results.append(result_oim)
    return pd.DataFrame(results)
