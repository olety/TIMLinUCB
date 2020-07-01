#! /usr/local/bin/python3

import random
from tqdm import tqdm
import pandas as pd
import numpy as np
import logging
from matplotlib import pyplot as plt
from functools import partial
import sys
from helpers import get_avg_reward


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
# %% ------------------------------------- RSB -----------------------------------------
# --------------------------------------------------------------------------------------


def get_reward_arm(df_graph, df_weights, new_seed):
    """ Runs independent cascade model.
    Input: df_g -- a dataframe representing the graph (with the probabilities)
    S -- initial set of vertices
    tracking -- whether we want to check for active/observed nodes
    Output: T -- resulted influenced set of vertices (including S)
    """
    prev_affected = df_weights[df_weights["walked"] == 1].index.tolist()
    new_affected = [new_seed]
    df_graph["activated"] = df_graph["probab"].apply(lambda x: random.random() <= x)

    i = 0
    while i < len(new_affected):
        # for neighbors of a selected node
        for row in df_graph[df_graph["source"] == new_affected[i]].itertuples():
            if (
                row.activated
                and row.target not in prev_affected
                and row.target not in new_affected
            ):
                new_affected.append(row.target)
        i += 1

    return np.array(new_affected)


def rsb(df_edges, nodes, times, num_seeds=10, C=1, gamma=0.2, num_repeats_expect=25):
    num_nodes = nodes.shape[0]
    df_weights = pd.DataFrame(
        data=1, index=nodes, columns=[f"weight_{k}" for k in range(num_seeds)]
    )
    df_weights["walked"] = False
    results = []
    for t in tqdm(times):
        # print(t)
        df_t = df_edges[df_edges["day"] <= t]
        df_weights["walked"] = False
        selected = []
        for cur_seed in range(num_seeds):
            df_weights["temp_weight"] = (
                gamma / num_nodes
                + (1 - gamma)
                * df_weights[f"weight_{cur_seed}"]
                / df_weights[f"weight_{cur_seed}"].sum()
            )

            selection_probab = (
                df_weights[~df_weights.index.isin(selected)]["temp_weight"]
                / df_weights[~df_weights.index.isin(selected)]["temp_weight"].sum()
            )
            # Draw an arm
            random_pt = random.uniform(0, df_weights[f"weight_{cur_seed}"].sum())
            selected_node = (
                df_weights[f"weight_{cur_seed}"].cumsum() >= random_pt
            ).idxmax()
            # Receiving the reward
            affected_arm = get_reward_arm(df_t, df_weights, selected_node)
            df_weights.loc[affected_arm, "walked"] = True
            marginal_gain = len(affected_arm)
            df_weights["expected_gain"] = 0
            df_weights.loc[selected_node, "expected_gain"] = (
                marginal_gain / selection_probab[selected_node]
            )

            selected.append(selected_node)
            df_weights[f"weight_{cur_seed}"] = df_weights[
                f"weight_{cur_seed}"
            ] * np.exp((gamma * df_weights["expected_gain"]) / (num_nodes * C))

        results.append(
            {
                "time": t,
                "reward": get_avg_reward(df_t, selected, num_repeats_expect),
                "selected": selected,
            }
        )
    return pd.DataFrame(results)


# --------------------------------------------------------------------------------------
# %% ------------------------------------- RSB 2 ---------------------------------------
# --------------------------------------------------------------------------------------


def rsb2(
    df_edges,
    nodes,
    times,
    num_seeds=10,
    C=1,
    gamma=0.2,
    num_repeats_expect=25,
    persist_params=True,
):
    num_nodes = nodes.shape[0]
    df_weights = pd.DataFrame(
        data=1,
        index=nodes,
        columns=[f"weight_{k}" for k in range(num_seeds)] + ["temp_weight"],
    )
    df_weights["walked"] = False
    df_weights["expected_gain"] = 0
    results = []
    for t in tqdm(times):
        df_t = df_edges[df_edges["day"] <= t]
        nodes_t = np.sort(np.unique(np.hstack((df_t["source"], df_t["target"]))))
        num_nodes_t = nodes_t.shape[0]
        df_weights["walked"] = False
        df_weights_t = df_weights.loc[nodes_t]
        selected = []

        for cur_seed in range(num_seeds):
            df_weights_t["temp_weight"] = (
                gamma / num_nodes_t
                + (1 - gamma)
                * df_weights_t[f"weight_{cur_seed}"]
                / df_weights_t[f"weight_{cur_seed}"].sum()
            )

            selection_probab = (
                df_weights_t[~df_weights_t.index.isin(selected)]["temp_weight"]
                / df_weights_t[~df_weights_t.index.isin(selected)]["temp_weight"].sum()
            )
            # Draw an arm
            random_pt = random.uniform(0, df_weights_t[f"weight_{cur_seed}"].sum())

            selected_node = (
                df_weights_t[~df_weights_t.index.isin(selected)][
                    f"weight_{cur_seed}"
                ].cumsum()
                >= random_pt
            ).idxmax()
            # Receiving the reward
            affected_arm = get_reward_arm(df_t, df_weights_t, selected_node)
            df_weights_t.loc[affected_arm, "walked"] = True
            marginal_gain = len(affected_arm)
            df_weights_t["expected_gain"] = 0
            p_selected = selection_probab.loc[selected_node]
            df_weights_t.loc[selected_node, "expected_gain"] = (
                marginal_gain / p_selected
            )

            selected.append(selected_node)
            df_weights_t[f"weight_{cur_seed}"] = df_weights_t[
                f"weight_{cur_seed}"
            ] * np.exp((gamma * df_weights_t["expected_gain"]) / (num_nodes * C))

        if persist_params:
            df_weights.loc[df_weights_t.index] = np.nan
            df_weights = df_weights.combine_first(df_weights_t)

        results.append(
            {
                "time": t,
                "reward": get_avg_reward(df_t, selected, num_repeats_expect),
                "selected": selected,
            }
        )
    return pd.DataFrame(results)
