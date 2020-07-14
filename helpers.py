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


def get_avg_reward(df, seeds, num_repeats):
    reward = []
    for i in range(num_repeats):
        reward.append(run_ic_nodes(df, seeds).shape[0])
    return np.average(reward)


def get_stats_reward(df, seeds, num_repeats):
    reward = []
    for i in range(num_repeats):
        reward.append(run_ic_nodes(df, seeds).shape[0])
    return np.average(reward), np.std(reward)


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
    """ Runs independent cascade model.
    Input: df_g -- a dataframe representing the graph (with the probabilities)
    S -- initial set of vertices
    tracking -- whether we want to check for active/observed nodes
    Output: T -- resulted influenced set of vertices (including S)
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
    result = setup_dict["function"](*setup_dict["args"], **setup_dict["kwargs"])
    result["time"] = setup_dict["time"]
    return result
