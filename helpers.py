#! /usr/local/bin/python3

import random
from tqdm import tqdm
import pandas as pd
import numpy as np
import re
import os
from copy import deepcopy
import joblib
import contextlib
import shutil
import stat
import traceback
from subprocess import Popen, PIPE

# --------------------------------------------------------------------------------------
# Parallel processing-related functions
# --------------------------------------------------------------------------------------


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    # Taken from https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution/41815007
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    def tqdm_print_progress(self):
        if self.n_completed_tasks > tqdm_object.n:
            n_completed = self.n_completed_tasks - tqdm_object.n
            tqdm_object.update(n=n_completed)

    original_print_progress = joblib.parallel.Parallel.print_progress
    joblib.parallel.Parallel.print_progress = tqdm_print_progress

    try:
        yield tqdm_object
    finally:
        joblib.parallel.Parallel.print_progress = original_print_progress
        tqdm_object.close()


def run_algorithm(setup_dict):
    """ Run an algorithm in parallel

    Run an algorithm in parallel. This is supposed to be used in conjunction with
    joblib.Parallel to run different IM algorithms at the same time.


    Parameters
    ----------
    setup_dict : dict
        A dictionary containing the following keys:
        * "function" : a function that we have to run
        * "algo_name" : name of the algorithm represented by the function
        * "args" : arguments for that functions
        * "kwargs" : keyword arguments for that functions

    Returns
    -------
    result_dict : dict
        A dictionary containing the following:
        * "results" : results generated by the function
        * "algo_name" : name of the algorithm represented by the function
        * "kwargs" : keyword argument from the function

    Examples
    --------
    >>> setup_array = []
    ... setup_array.append(
    ...    {
    ...        "algo_name": "timlinucb",
    ...        "function": timlinucb_parallel_t,
    ...        "args": [DATASET, DATASET_FEATS, DATASET_TIMES, DATASET_NODES],
    ...        "kwargs": {
    ...            "num_seeds": NUM_SEEDS_TO_FIND,
    ...            "num_repeats_oim": OPTIMAL_NUM_REPEATS_OIM_TLU,
    ...            "num_repeats_oim_reward": OPTIMAL_NUM_REPEATS_REW_TLU,
    ...            "sigma": OPTIMAL_SIGMA_TLU,
    ...            "c": OPTIMAL_C_TLU,
    ...            "epsilon": OPTIMAL_EPS_TLU,
    ...        },
    ...    }
    ... )
    ... results_array = joblib.Parallel(n_jobs=len(setup_array))(
    ...     joblib.delayed(run_algorithm)(setup_dict) for setup_dict in setup_array
    ... )


    """
    try:
        result_dict = {
            "result": setup_dict.get("function")(
                *setup_dict.get("args"), **setup_dict.get("kwargs")
            ),
            "algo_name": setup_dict.get("algo_name"),
            "kwargs": setup_dict.get("kwargs"),
        }
    except Exception as e:
        print(e)
        print(setup_dict)
        traceback.print_exc()
        return {}
    return result_dict


def _run_timlinucb_parallel(setup_dict):
    """ Run IMLinUCB in parallel

    This is a helper function used by timlinucb_parallel_oim from timlinucb.py to
    run multiple IMLinUCB instances at the same time.

    Parameters
    ----------
    setup_dict : dict
        A dictionary containing the following keys:
        * "function" : a function that we have to run
        * "time" : time t of the current OIM execution
        * "args" : arguments for that functions
        * "kwargs" : keyword arguments for that functions

    Returns
    -------
    result_dict : dict
        A dictionary containing the following:
        * "results" : results generated by IMLinUCB
        * "time" : time t of the current OIM execution


    """
    result = setup_dict["function"](*setup_dict["args"], **setup_dict["kwargs"])
    result["time"] = setup_dict["time"]
    return result


# --------------------------------------------------------------------------------------
# IC-related functions
# --------------------------------------------------------------------------------------


def get_avg_reward(df, seeds, num_repeats):
    """ Simulate the influence propagation using the IC model

    Parameters
    ----------
    df : pandas.DataFrame
        The graph we run the IC on, in the form of a DataFrame. A row represents one
        edge in the graph, with columns being named "source", "target", "probab".
        "probab" column contains the activation probability.
    seeds : list, pandas.Series
        A list of the nodes to start propagating from.
    num_repeats : int
        Specifies how many times we want to simulate the propagation with IC.

    Returns
    -------
    avg_reward : float
        Number showing how many nodes were influenced on average
    """
    reward = []
    for i in range(num_repeats):
        reward.append(run_ic_nodes(df, seeds).shape[0])
    return np.average(reward)


def get_stats_reward(df, seeds, num_repeats):
    """ Simulate the influence propagation using the IC model

    Parameters
    ----------
    df : pandas.DataFrame
        The graph we run the IC on, in the form of a DataFrame. A row represents one
        edge in the graph, with columns being named "source", "target", "probab".
        "probab" column contains the activation probability.
    seeds : list, pandas.Series
        A list of the nodes to start propagating from.
    num_repeats : int
        Specifies how many times we want to simulate the propagation with IC.

    Returns
    -------
    avg_reward : float
        Number showing how many nodes were influenced on average
    std_reward : float
        Standard deviation of avg_reward
    """
    reward = []
    for i in range(num_repeats):
        reward.append(run_ic_nodes(df, seeds).shape[0])
    return np.average(reward), np.std(reward)


def run_ic_eff(df_graph, seed_nodes):
    """ Simulate the influence propagation using the IC model

    Parameters
    ----------
    df_graph : pandas.DataFrame
        The graph we run the IC on, in the form of a DataFrame. A row represents one
        edge in the graph, with columns being named "source", "target", "probab".
        "probab" column contains the activation probability.
    seed_nodes : list, pandas.Series
        A list of the nodes to start propagating from.

    Returns
    -------
    results : tuple
        A tuple of the following numpy arrays
        - Affected nodes
        - Activated edges
        - Observed edges

    """
    affected_nodes = deepcopy(seed_nodes)  # copy already selected nodes
    activated_edges = []
    observed_edges = []
    df_graph["activated"] = df_graph["probab"].apply(lambda x: random.random() <= x)

    i = 0
    while i < len(affected_nodes):
        # for neighbors of a selected node
        for row in df_graph[df_graph["source"] == affected_nodes[i]].itertuples():
            observed_edges.append(row.Index)
            if row.activated and row.target not in affected_nodes:
                activated_edges.append(row.Index)
                affected_nodes.append(row.target)
        i += 1

    return np.array(affected_nodes), np.array(activated_edges), np.array(observed_edges)


def run_ic_nodes(df_graph, seed_nodes):
    """ Simulate the influence propagation using the IC model

    Parameters
    ----------
    df_graph : pandas.DataFrame
        The graph we run the IC on, in the form of a DataFrame. A row represents one
        edge in the graph, with columns being named "source", "target", "probab".
        "probab" column contains the activation probability.
    seed_nodes : list, pandas.Series
        A list of the nodes to start propagating from.

    Returns
    -------
    affected_nodes : numpy.array
        Nodes influenced by propagating the seed nodes.
    """
    affected_nodes = deepcopy(seed_nodes)  # copy already selected nodes
    df_graph["activated"] = df_graph["probab"].apply(lambda x: random.random() <= x)

    i = 0
    while i < len(affected_nodes):
        # for neighbors of a selected node
        for row in df_graph[df_graph["source"] == affected_nodes[i]].itertuples():
            if row.activated and row.target not in affected_nodes:
                affected_nodes.append(row.target)
        i += 1

    return np.array(affected_nodes)


# --------------------------------------------------------------------------------------
# TIM-related functions
# --------------------------------------------------------------------------------------


def tim(
    df,
    num_nodes,
    num_edges,
    num_inf,
    epsilon,
    temp_dir="temp_dir",
    out_pattern=re.compile("Selected k SeedSet: (.+?) \\n"),
):
    """ Runs TIM (the oracle function).
    Input: df -- the graph to process
    num_inf -- the k the we are looking for
    epsilon -- hyperparameter
    Output: T -- The k highest influencers
    """
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    df.to_csv(
        os.path.join(temp_dir, "graph_ic.inf"), index=False, sep=" ", header=False
    )
    # Preparing to run TIM
    with open(os.path.join(temp_dir, "attribute.txt"), "w+") as f:
        f.write(f"n={num_nodes}\nm={num_edges}")

    process = Popen(
        [
            "./tim",
            "-model",
            "IC",
            "-dataset",
            temp_dir,
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
    # logging.debug(f"Running TIM, {out}")
    return list(map(int, out_pattern.findall(out)[0].split(" ")))


def tim_parallel(
    df,
    num_nodes,
    num_edges,
    num_inf,
    epsilon,
    tim_file="./tim",
    temp_dir="temp_dir",
    out_pattern=re.compile("Selected k SeedSet: (.+?) \\n"),
):
    """ Runs TIM (the oracle function).
    Input: df -- the graph to process
    num_inf -- the k the we are looking for
    epsilon -- hyperparameter
    Output: T -- The k highest influencers
    """
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    df.to_csv(
        os.path.join(temp_dir, "graph_ic.inf"), index=False, sep=" ", header=False
    )
    # Preparing to run TIM
    with open(os.path.join(temp_dir, "attribute.txt"), "w+") as f:
        f.write(f"n={num_nodes}\nm={num_edges}")

    process = Popen(
        [
            f"./{tim_file}",
            "-model",
            "IC",
            "-dataset",
            f"{temp_dir}",
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
    # logging.debug(f"Running TIM, {out}")
    return list(map(int, out_pattern.findall(out)[0].split(" ")))


def tim_t(df_edges, nodes, times, num_seeds=5, num_repeats_reward=20, epsilon=0.4):
    # TIM wants the max node ID ()
    num_nodes = nodes[-1] + 1
    results = []
    for t in tqdm(times):
        df_t = df_edges[df_edges["day"] <= t]
        num_edges_t = df_t.shape[0]
        selected_seeds = tim(
            df_t[["source", "target", "probab"]],
            num_nodes,
            num_edges_t,
            num_seeds,
            epsilon,
        )
        results.append(
            {
                "time": t,
                "reward": get_avg_reward(df_t, selected_seeds, num_repeats_reward),
                "selected": selected_seeds,
            }
        )
    return pd.DataFrame(results)


def tim_t_parallel(
    df_edges,
    nodes,
    times,
    num_seeds=5,
    num_repeats_reward=20,
    epsilon=0.4,
    process_id=1,
):
    tim_name = "tim_t_" + str(process_id)
    temp_dir_name = tim_name + "_dir"
    shutil.copyfile("tim", tim_name)
    # Making the new tim file executable
    st = os.stat(tim_name)
    os.chmod(tim_name, st.st_mode | stat.S_IEXEC)

    # TIM wants the max node ID, starting from 0
    num_nodes = nodes[-1] + 1
    results = []
    for t in times:
        df_t = df_edges[df_edges["day"] <= t]
        num_edges_t = df_t.shape[0]
        selected_seeds = tim_parallel(
            df_t[["source", "target", "probab"]],
            num_nodes,
            num_edges_t,
            num_seeds,
            epsilon,
            tim_name,
            temp_dir_name,
        )
        results.append(
            {
                "time": t,
                "reward": get_avg_reward(df_t, selected_seeds, num_repeats_reward),
                "selected": selected_seeds,
            }
        )
    return pd.DataFrame(results)
