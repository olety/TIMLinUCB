#! /usr/local/bin/python3

import random
from tqdm import tqdm
import pandas as pd
import numpy as np
import logging
from matplotlib import pyplot as plt
from helpers import get_stats_reward


# --------------------------------------------------------------------------------------
# %% ------------------------------ Initial setup --------------------------------------
# --------------------------------------------------------------------------------------

# Fancy plots in matplotlib and pandas
random.seed(42)
plt.style.use("ggplot")
pd.options.mode.chained_assignment = None  # default='warn'

# Setting up logging
VERBOSE = False
LOGGING_FMT = (
    "%(levelname)s | %(asctime)s | line %(lineno)s | %(funcName)s | %(message)s"
)
LOGGING_DATEFMT = "%H:%M:%S"

logger_rsb = logging.getLogger("logger_rsb")

syslog = logging.StreamHandler()
formatter = logging.Formatter(fmt=LOGGING_FMT, datefmt=LOGGING_DATEFMT)
syslog.setFormatter(formatter)
logger_rsb.addHandler(syslog)

if VERBOSE:
    logger_rsb.setLevel(logging.DEBUG)
else:
    logger_rsb.setLevel(logging.INFO)


# --------------------------------------------------------------------------------------
# %% ------------------------------------- RSB -----------------------------------------
# --------------------------------------------------------------------------------------


def get_reward_arm(df_graph, df_weights, new_seed):
    """ Run the IC model and get the reward of adding a seed node

    Parameters
    ----------
    df_graph : pandas.DataFrame
        The graph we run the RSB on, in the form of a DataFrame. A row represents one
        edge in the graph, with columns being named "source", "target", "probab".
        "probab" column is the "true" activation probability used for the simulation.
    df_weights : pandas.DataFrame
        A dataframe of the node weights used by RSB.
    new_seed : int
        An id of the new seed node that we are going to test.

    Returns
    -------
    results : numpy.Array
        An array for nodes affected by adding the new_seed to the seed nodes

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


# --------------------------------------------------------------------------------------
# %% -------------------------------------- RSB ----------------------------------------
# --------------------------------------------------------------------------------------


def rsb(
    df_edges,
    nodes,
    times,
    num_seeds=10,
    C=1.0,
    gamma=0.2,
    num_repeats_expect=25,
    persist_params=True,
    style="additive",
    hide_tqdm=False,
):
    """ Run the RSB algorithm on a graph

    Runs the RSB algorithm [1].

    Parameters
    ----------
    df_edges : pandas.DataFrame
        The graph we run the TOIM on, in the form of a DataFrame. A row represents one
        edge in the graph, with columns being named "source", "target", "probab",
        and "day". "probab" column is the "true" activation probability and "day" should
        correspond to the days specified in times.
    nodes : pandas.Series
        A series containing all unique nodes in df.
    times : pandas.Series, list
        A series or a list of the times that we are going to iterate through. Useful
        if you don't want to iterate through every day in the network.
    num_seeds : int, optional
        Number of seed nodes to find. Default: 10
    C: float, optional
        A hyperparameter used by the RSB algorithm. Refer to the RSB paper for
        more details. [1] Default: 1.0
    gamma : float, optional
        A hyperparameter used by the RSB algorithm. Refer to the RSB paper for
        more details. [1] Default: 0.2
    num_repeats_expect : int, optional
        Default: 25
    persist_params : boolean, optional
        Determines if we want to persist the OIM parameters. Default: False
    style : str, optional
        Determines whether we take into account all edges up to t ("additive") or just
        the ones that were formed at t ("dynamic"). Default: "additive"
    hide_tqdm : boolean, optional
        A paremeters used if you want to hide all tqdm progress bars. It's useful if
        you want to paralellize the algorithm. Default: False

    Returns
    -------
    results : DataFrame
        A dataframe with the following columns
        - time_t, the time step at which everything else was obtained
        - reward, the average reward obtained by running IC with s_best
        - selected, the list of the selected seed nodes

    References
    ----------
    .. [1] Bao, Yixin, et al.
        "Online influence maximization in non-stationary social networks."
        2016 IEEE/ACM 24th International Symposium on Quality of Service (IWQoS). IEEE, 2016

    """
    num_nodes = nodes.shape[0]
    df_weights = pd.DataFrame(
        data=1,
        index=nodes,
        columns=[f"weight_{k}" for k in range(num_seeds)] + ["temp_weight"],
    )
    df_weights["walked"] = False
    df_weights["expected_gain"] = 0
    results = []

    # We want to hide TQDM if processing RSB in parallel
    if hide_tqdm:
        times_iter = times
    else:
        times_iter = tqdm(times, desc=f"RSB iters", leave=True)

    for t in times_iter:

        if style == "additive":
            df_t = df_edges[df_edges["day"] <= t]
        elif style == "dynamic":
            df_t = df_edges[df_edges["day"] == t]

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
        reward, std = get_stats_reward(df_t, selected, num_repeats_expect)
        results.append(
            {"time": t, "reward": reward, "std": std, "selected": selected,}
        )
    return pd.DataFrame(results)
