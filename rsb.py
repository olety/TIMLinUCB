import random
from tqdm import tqdm
from copy import deepcopy
from subprocess import Popen, PIPE
import pandas as pd
import numpy as np
import logging
from matplotlib import pyplot as plt
from functools import partial
import os
import sys
import re
# %% ---- Initial setup ----

# Fancy plots in matplotlib and pandas
random.seed(42)
plt.style.use('ggplot')
pd.options.mode.chained_assignment = None  # default='warn'

# Setting up logging
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

# Datasets
DATASET_DIR = os.path.join("..", "Datasets")
if not os.path.exists(DATASET_DIR):
    print("Can't find the dataset directory!")
DATASET_FACEBOOK = os.path.join(DATASET_DIR, "fb-wosn-friends")


# %%
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


def run_ic_nodes(df_graph, seed_nodes):
    """ Runs independent cascade model.
    Input: df_g -- a dataframe representing the graph (with the probabilities)
    S -- initial set of vertices
    tracking -- whether we want to check for active/observed nodes
    Output: T -- resulted influenced set of vertices (including S)
    """
    affected_nodes = deepcopy(seed_nodes)  # copy already selected nodes
    df_graph["activated"] = df_graph["probab"].apply(
        lambda x: random.random() <= x)

    i = 0
    while i < len(affected_nodes):
        # for neighbors of a selected node
        for row in df_graph[df_graph["source"] ==
                            affected_nodes[i]].itertuples():
            if row.activated and row.target not in affected_nodes:
                affected_nodes.append(row.target)
        i += 1

    return np.array(affected_nodes)


def get_reward_arm(df_graph, df_weights, new_seed):
    """ Runs independent cascade model.
    Input: df_g -- a dataframe representing the graph (with the probabilities)
    S -- initial set of vertices
    tracking -- whether we want to check for active/observed nodes
    Output: T -- resulted influenced set of vertices (including S)
    """
    prev_affected = df_weights[df_weights["walked"] == 1].index.tolist()
    new_affected = [new_seed]
    df_graph["activated"] = df_graph["probab"].apply(
        lambda x: random.random() <= x)

    i = 0
    while i < len(new_affected):
        # for neighbors of a selected node
        for row in df_graph[df_graph["source"] ==
                            new_affected[i]].itertuples():
            if row.activated and row.target not in prev_affected and row.target not in new_affected:
                new_affected.append(row.target)
        i += 1

    return np.array(new_affected)


def tim(df,
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
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

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
    # logging.debug(f"Running TIM, {out}")
    return list(map(int, out_pattern.findall(out)[0].split(" ")))


# %%


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


def tim_t(df_edges,
          nodes,
          times,
          num_seeds=5,
          num_repeats_reward=20,
          epsilon=0.4):
    # TIM wants the max node ID
    num_nodes = nodes[-1]
    results = []
    for t in tqdm(times):
        df_t = df_edges[df_edges["day"] <= t]
        num_edges_t = df_t.shape[0]
        selected_seeds = tim(df_t[["source", "target", "probab"]], num_nodes,
                             num_edges_t, num_seeds, epsilon)
        results.append({
            "time":
            t,
            "reward":
            get_avg_reward(df_t, selected_seeds, num_repeats_reward),
            "selected":
            selected_seeds
        })
    return pd.DataFrame(results)


# %% ---- RSB ----


def rsb(df_edges,
        nodes,
        times,
        num_seeds=10,
        C=1,
        gamma=0.2,
        num_repeats_expect=25):
    num_nodes = nodes.shape[0]
    df_weights = pd.DataFrame(
        data=1, index=nodes, columns=[f"weight_{k}" for k in range(num_seeds)])
    df_weights["walked"] = False
    results = []
    for t in tqdm(times):
        # print(t)
        df_t = df_edges[df_edges["day"] <= t]
        df_weights["walked"] = False
        selected = []
        for cur_seed in range(num_seeds):
            df_weights["temp_weight"] = gamma/num_nodes \
                                        + (1-gamma) * df_weights[f"weight_{cur_seed}"] \
                                        / df_weights[f"weight_{cur_seed}"].sum()

            selection_probab = df_weights[~df_weights.index.isin(selected)]["temp_weight"] \
                                / df_weights[~df_weights.index.isin(selected)]["temp_weight"].sum()
            # Draw an arm
            random_pt = random.uniform(0,
                                       df_weights[f"weight_{cur_seed}"].sum())
            selected_node = (df_weights[f"weight_{cur_seed}"].cumsum() >=
                             random_pt).idxmax()
            # Receiving the reward
            affected_arm = get_reward_arm(df_t, df_weights, selected_node)
            df_weights.loc[affected_arm, "walked"] = True
            marginal_gain = len(affected_arm)
            df_weights["expected_gain"] = 0
            df_weights.loc[selected_node,
                           "expected_gain"] = marginal_gain / selection_probab[
                               selected_node]

            selected.append(selected_node)
            df_weights[f"weight_{cur_seed}"] = df_weights[
                f"weight_{cur_seed}"] * np.exp(
                    (gamma * df_weights["expected_gain"]) / (num_nodes * C))

        results.append({
            "time":
            t,
            "reward":
            get_avg_reward(df_t, selected, num_repeats_expect),
            "selected":
            selected
        })
    return pd.DataFrame(results)


# %% IMLinUCB


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


def generate_node2vec_fetures(df_friend, num_features=20):
    # Getting node embeddings
    logging.debug("Getting node embeddings...")
    df_emb = get_features_nodes(df_friend,
                                dims=num_features,
                                node2vec_path=os.getcwd(),
                                dataset_path=DATASET_FACEBOOK)
    df_emb = df_emb.set_index("node").sort_values(by="node")
    # Generating edge features
    logging.debug(f"Generating {num_features} edge features...")
    df_feats = []
    for row in tqdm(df_friend.itertuples()):
        df_feats.append(df_emb.loc[row.source].values *
                        df_emb.loc[row.target].values)
    df_feats = pd.DataFrame(df_feats)
    return df_feats


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
            # u_e.append(expit(x_e @ theta + c * np.sqrt(xMx)))

        u_e = np.array(u_e)

        # ---- Step 2 - Evaluating the performance ----
        # Loss function
        df["probab"] = u_e
        s_oracle = sorted(
            oracle(df[["source", "target", "probab"]], num_nodes_tim,
                   num_edges_t, num_inf, epsilon))

        # Observing edge-level feedback
        df["probab"] = true_weights

        all_algo_nodes = []
        all_algo_edges = []
        all_algo_obs = []
        for k in range(num_repeats_reward):
            algo_act_nodes, algo_act_edges, algo_obs_edges = run_ic_eff(
                df, s_oracle)
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
            m_inv -= (m_inv @ x_e.T @ x_e @ m_inv) / (x_e @ m_inv @ x_e.T +
                                                      sigma * sigma)
            b += x_e.T * int(i in all_algo_edges)

    return_dict = {
        "s_best": s_best,
        "u_e_best": u_e_best,
        "reward_best": reward_best
    }
    logging.debug("The algorithm has finished running.")
    logging.debug(f"Returning: {return_dict}")
    return return_dict


# %%  TIMLinUCB
def timlinucb(df_edges,
              df_feats,
              times,
              nodes,
              num_seeds=5,
              sigma=4,
              c=0.1,
              epsilon=0.4,
              num_repeats_oim=10,
              num_repeats_oim_reward=10):
    results = []
    for t in tqdm(times):
        df_t = df_friend[df_friend["day"] <= t].sort_values(
            "source").reset_index()
        df_feats_t = df_t["index"].apply(lambda x: df_feats.loc[x])
        result_oim = oim_node2vec(df_t,
                                  df_feats_t,
                                  nodes,
                                  num_inf=num_seeds,
                                  sigma=sigma,
                                  c=c,
                                  epsilon=epsilon,
                                  num_repeats=num_repeats_oim,
                                  num_repeats_reward=num_repeats_oim_reward)
        result_oim["time"] = t
        results.append(result_oim)
    return pd.DataFrame(results)


# %% RSB 2


def rsb2(df_edges,
         nodes,
         times,
         num_seeds=10,
         C=1,
         gamma=0.2,
         num_repeats_expect=25,
         persist_params=True):
    num_nodes = nodes.shape[0]
    df_weights = pd.DataFrame(
        data=1,
        index=nodes,
        columns=[f"weight_{k}" for k in range(num_seeds)] + ["temp_weight"])
    df_weights["walked"] = False
    df_weights["expected_gain"] = 0
    results = []
    for t in tqdm(times):
        df_t = df_edges[df_edges["day"] <= t]
        nodes_t = np.sort(
            np.unique(np.hstack((df_t["source"], df_t["target"]))))
        num_nodes_t = nodes_t.shape[0]
        df_weights["walked"] = False
        df_weights_t = df_weights.loc[nodes_t]
        selected = []

        for cur_seed in range(num_seeds):
            df_weights_t["temp_weight"] = gamma/num_nodes_t \
                                        + (1-gamma) * df_weights_t[f"weight_{cur_seed}"] \
                                        / df_weights_t[f"weight_{cur_seed}"].sum()

            selection_probab = df_weights_t[~df_weights_t.index.isin(selected)]["temp_weight"] \
                                / df_weights_t[~df_weights_t.index.isin(selected)]["temp_weight"].sum()
            # Draw an arm
            random_pt = random.uniform(
                0, df_weights_t[f"weight_{cur_seed}"].sum())

            selected_node = (df_weights_t[~df_weights_t.index.isin(selected)]
                             [f"weight_{cur_seed}"].cumsum() >=
                             random_pt).idxmax()
            # Receiving the reward
            affected_arm = get_reward_arm(df_t, df_weights_t, selected_node)
            df_weights_t.loc[affected_arm, "walked"] = True
            marginal_gain = len(affected_arm)
            df_weights_t["expected_gain"] = 0
            p_selected = selection_probab.loc[selected_node]
            df_weights_t.loc[selected_node,
                             "expected_gain"] = marginal_gain / p_selected

            selected.append(selected_node)
            df_weights_t[f"weight_{cur_seed}"] = df_weights_t[
                f"weight_{cur_seed}"] * np.exp(
                    (gamma * df_weights_t["expected_gain"]) / (num_nodes * C))

        if persist_params:
            df_weights.loc[df_weights_t.index] = np.nan
            df_weights = df_weights.combine_first(df_weights_t)

        results.append({
            "time":
            t,
            "reward":
            get_avg_reward(df_t, selected, num_repeats_expect),
            "selected":
            selected
        })
    return pd.DataFrame(results)


# %% ---- Running the code ----

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
logging.debug("Generating \"true\" activation probabilities...")
df_friend['probab'] = np.random.uniform(0, 0.1, size=df_friend.shape[0])
# df_friend['probab'] = np.random.uniform(0, 1, size=df_friend.shape[0])
times = np.sort(np.unique(df_friend["day"]))[0:20]

# %% Preparing for TIMLinUCB
df_feats = generate_node2vec_fetures(df_friend, num_features=20)

# %%
df_timlinucb = timlinucb(df_friend,
                         df_feats,
                         times,
                         nodes,
                         num_seeds=5,
                         num_repeats_oim=10,
                         num_repeats_oim_reward=1,
                         sigma=4,
                         c=0.1,
                         epsilon=0.4)

# %%

logging.debug("Running TIM")
df_tim = tim_t(df_friend,
               nodes,
               times,
               num_seeds=5,
               num_repeats_reward=25,
               epsilon=0.1)

# %%

logging.debug("Running RSB")
df_rsb = rsb(df_friend,
             nodes,
             times,
             num_seeds=5,
             C=1,
             gamma=0.2,
             num_repeats_expect=25)
df_rsb2_persist = rsb2(df_friend,
                       nodes,
                       times,
                       num_seeds=5,
                       C=1,
                       gamma=0.2,
                       num_repeats_expect=25,
                       persist_params=True)

df_rsb2_nopersist = rsb2(df_friend,
                         nodes,
                         times,
                         num_seeds=5,
                         C=1,
                         gamma=0.2,
                         num_repeats_expect=25,
                         persist_params=False)
# %%
results = []
num_repeats_test = 10
for t in tqdm(times):
    df_t = df_friend[df_friend["day"] <= t]
    results_t = {"time": t}
    results_t["timlinucb_mean"], results_t["timlinucb_std"] = get_stats_reward(
        df_t, df_timlinucb.loc[df_timlinucb["time"] == t, "s_best"].item(),
        num_repeats_test)
    results_t["rsb_mean"], results_t["rsb_std"] = get_stats_reward(
        df_t, df_rsb.loc[df_rsb["time"] == t, "selected"].item(),
        num_repeats_test)
    results_t["tim_mean"], results_t["tim_std"] = get_stats_reward(
        df_t, df_tim.loc[df_tim["time"] == t, "selected"].item(),
        num_repeats_test)
    results_t["rsb_persist_mean"], results_t[
        "rsb_persist_std"] = get_stats_reward(
            df_t, df_rsb2_persist.loc[df_rsb2_persist["time"] == t,
                                      "selected"].item(), num_repeats_test)
    results_t["rsb_nopersist_mean"], results_t[
        "rsb_nopersist_std"] = get_stats_reward(
            df_t, df_rsb2_nopersist.loc[df_rsb2_nopersist["time"] == t,
                                        "selected"].item(), num_repeats_test)
    results.append(results_t)
results_df = pd.DataFrame(results)
# %%
ax = results_df.plot(x="time", y="rsb_mean", yerr="rsb_std")
results_df.plot(x="time", y="rsb_persist_mean", yerr="rsb_persist_std", ax=ax)
results_df.plot(x="time",
                y="rsb_nopersist_mean",
                yerr="rsb_nopersist_std",
                ax=ax)
results_df.plot(x="time", y="tim_mean", yerr="tim_std", ax=ax)
results_df.plot(x="time", y="timlinucb_mean", yerr="timlinucb_std", ax=ax)
ax.set(xlabel="Snapshot time",
       ylabel="Reward (nodes)",
       title="Reward comparison (5 seeds)")
ax.legend(["RSB", "RSB2", "RSB2 (No persist)", "TIM", "TIMLinUCB"])
plt.savefig("plot_5seeds.png", dpi=250)

# TODO:
# Calculate average regret for timlinucb and rsb, compare them
# The result is the % imp to write in the thesis
# The smaller regret_frac is the better

regret_frac_tlu = (
    results_df["timlinucb_mean"].sum() -
    results_df["tim_mean"].sum()) / results_df["tim_mean"].sum()
regret_frac_rsb = (
    results_df["rsb_nopersist_mean"].sum() -
    results_df["tim_mean"].sum()) / results_df["tim_mean"].sum()

# %% Experiment 2 - 10 seeds, times 1-20
NUM_SEEDS = 10
logging.debug(
    "Running TIMLinUCB (15 oim reps, 1 reward rep, sigma 4, c 0.1, eps 0.4)")
df_timlinucb_15rep = timlinucb(df_friend,
                               df_feats,
                               times,
                               nodes,
                               num_seeds=NUM_SEEDS,
                               num_repeats_oim=15,
                               num_repeats_oim_reward=1,
                               sigma=4,
                               c=0.1,
                               epsilon=0.4)
logging.debug(
    "Running TIMLinUCB (5 oim reps, 1 reward rep, sigma 4, c 0.1, eps 0.25)")
df_timlinucb_5rep = timlinucb(df_friend,
                              df_feats,
                              times,
                              nodes,
                              num_seeds=NUM_SEEDS,
                              num_repeats_oim=5,
                              num_repeats_oim_reward=1,
                              sigma=4,
                              c=0.1,
                              epsilon=0.2)
logging.debug(
    "Running TIMLinUCB (10 oim reps, 1 reward rep, sigma 4, c 0.1, eps 0.4)")
df_timlinucb_5repeps04 = timlinucb(df_friend,
                                   df_feats,
                                   times,
                                   nodes,
                                   num_seeds=NUM_SEEDS,
                                   num_repeats_oim=5,
                                   num_repeats_oim_reward=1,
                                   sigma=4,
                                   c=0.1,
                                   epsilon=0.4)
logging.debug(
    "Running TIMLinUCB (10 oim reps, 1 reward rep, sigma 4, c 0.1, eps 0.4)")
df_timlinucb_10rep = timlinucb(df_friend,
                               df_feats,
                               times,
                               nodes,
                               num_seeds=NUM_SEEDS,
                               num_repeats_oim=10,
                               num_repeats_oim_reward=1,
                               sigma=4,
                               c=0.1,
                               epsilon=0.4)
logging.debug(
    "Running TIMLinUCB (10 oim reps, 5 reward rep, sigma 4, c 0.1, eps 0.4)")
df_timlinucb_105 = timlinucb(df_friend,
                             df_feats,
                             times,
                             nodes,
                             num_seeds=NUM_SEEDS,
                             num_repeats_oim=10,
                             num_repeats_oim_reward=5,
                             sigma=4,
                             c=0.1,
                             epsilon=0.4)
# %%
logging.debug("Running RSB")
df_rsb_10s = rsb(df_friend,
                 nodes,
                 times,
                 num_seeds=NUM_SEEDS,
                 C=1,
                 gamma=0.2,
                 num_repeats_expect=1)
logging.debug("Running RSB2 (persist)")
df_rsb2_10s_persist = rsb2(df_friend,
                           nodes,
                           times,
                           num_seeds=NUM_SEEDS,
                           C=1,
                           gamma=0.2,
                           num_repeats_expect=1,
                           persist_params=True)
logging.debug("Running RSB2 (no persist)")
df_rsb2_10s_nopersist = rsb2(df_friend,
                             nodes,
                             times,
                             num_seeds=NUM_SEEDS,
                             C=1,
                             gamma=0.2,
                             num_repeats_expect=1,
                             persist_params=False)
logging.debug("Running RSB2 (no persist, gamma = 0.8)")
df_rsb2_10s_nopersist_g08 = rsb2(df_friend,
                                 nodes,
                                 times,
                                 num_seeds=NUM_SEEDS,
                                 C=1,
                                 gamma=0.8,
                                 num_repeats_expect=1,
                                 persist_params=False)
logging.debug("Running RSB2 (no persist, gamma = 0.5)")
df_rsb2_10s_nopersist_g05 = rsb2(df_friend,
                                 nodes,
                                 times,
                                 num_seeds=NUM_SEEDS,
                                 C=1,
                                 gamma=0.5,
                                 num_repeats_expect=1,
                                 persist_params=False)
# %% TIM
logging.debug("Running TIM")
df_tim_10s = tim_t(df_friend,
                   nodes,
                   times,
                   num_seeds=NUM_SEEDS,
                   num_repeats_reward=1,
                   epsilon=0.1)
logging.debug("Running TIM")
df_tim_10se05 = tim_t(df_friend,
                      nodes,
                      times,
                      num_seeds=NUM_SEEDS,
                      num_repeats_reward=1,
                      epsilon=0.5)
logging.debug("Running TIM")
df_tim_10se07 = tim_t(df_friend,
                      nodes,
                      times,
                      num_seeds=NUM_SEEDS,
                      num_repeats_reward=1,
                      epsilon=0.7)

# %%
results = []
num_repeats_test = 10
res_dict = {}
for t in tqdm(times):
    df_t = df_friend[df_friend["day"] <= t]
    results_t = {"time": t}
    results_t["df_rsb_10s"] = get_stats_reward(
        df_t, df_rsb_10s.loc[df_rsb_10s["time"] == t, "selected"].item(),
        num_repeats_test)
    results_t["df_rsb2_10s_persist"] = get_stats_reward(
        df_t, df_rsb2_10s_persist.loc[df_rsb2_10s_persist["time"] == t,
                                      "selected"].item(), num_repeats_test)
    results_t["df_rsb2_10s_nopersist"] = get_stats_reward(
        df_t, df_rsb_10s.loc[df_rsb_10s["time"] == t, "selected"].item(),
        num_repeats_test)
    results_t["df_rsb2_10s_nopersist"] = get_stats_reward(
        df_t, df_rsb2_10s_nopersist.loc[df_rsb2_10s_nopersist["time"] == t,
                                        "selected"].item(), num_repeats_test)
    results_t["df_rsb2_10s_nopersist_g08"] = get_stats_reward(
        df_t,
        df_rsb2_10s_nopersist_g08.loc[df_rsb2_10s_nopersist_g08["time"] == t,
                                      "selected"].item(), num_repeats_test)
    results_t["df_rsb2_10s_nopersist_g05"] = get_stats_reward(
        df_t, df_rsb_10s.loc[df_rsb_10s["time"] == t, "selected"].item(),
        num_repeats_test)
    results_t["df_tim_10s"] = get_stats_reward(
        df_t, df_tim_10s.loc[df_tim_10s["time"] == t, "selected"].item(),
        num_repeats_test)
    results_t["df_tim_10se05"] = get_stats_reward(
        df_t, df_tim_10se05.loc[df_tim_10se05["time"] == t, "selected"].item(),
        num_repeats_test)
    results_t["df_tim_10se07"] = get_stats_reward(
        df_t, df_tim_10se07.loc[df_tim_10se07["time"] == t, "selected"].item(),
        num_repeats_test)
    results_t["df_timlinucb_15rep"] = get_stats_reward(
        df_t, df_timlinucb_15rep.loc[df_timlinucb_15rep["time"] == t,
                                     "s_best"].item(), num_repeats_test)
    results_t["df_timlinucb_10rep"] = get_stats_reward(
        df_t, df_timlinucb_10rep.loc[df_timlinucb_10rep["time"] == t,
                                     "s_best"].item(), num_repeats_test)
    results_t["df_timlinucb_5rep"] = get_stats_reward(
        df_t, df_timlinucb_5rep.loc[df_timlinucb_5rep["time"] == t,
                                    "s_best"].item(), num_repeats_test)
    results_t["df_timlinucb_5repeps04"] = get_stats_reward(
        df_t, df_timlinucb_5repeps04.loc[df_timlinucb_5repeps04["time"] == t,
                                         "s_best"].item(), num_repeats_test)
    results_t["df_timlinucb_105"] = get_stats_reward(
        df_t, df_timlinucb_105.loc[df_timlinucb_105["time"] == t,
                                   "s_best"].item(), num_repeats_test)
    results.append(results_t)
results_df = pd.DataFrame(results)

# %%
res_sum = {}
for col, val in results_df.iteritems():
    if col == "time":
        continue
    res_sum[col] = sum([x[0] for x in val])
res_sum
pd.DataFrame(res_sum, index=[0])
rsb_res = 429.9  # df_rsb2_10s_nopersist
tlu_res = 2571.2  # df_timlinucb_15rep
tim_res = 2878.1

tlu_res / tim_res - rsb_res / tim_res

# %%
results_df
fig = plt.figure(1, figsize=[10, 10])
ax = plt.gca()

for col, val in results_df.iteritems():
    if col == "time":
        continue
    plt.plot(results_df["time"], [x[0] for x in val], label=col)

ax.set(xlabel="Snapshot time",
       ylabel="Reward (nodes)",
       title="Reward comparison (5 seeds)")
ax.legend()
# ax.legend(["RSB", "RSB2", "RSB2 (No persist)", "TIM", "TIMLinUCB"])
plt.savefig("plot_10seeds_20t.png", dpi=250)
# plt.show()

# %% 100 time_it
times = np.sort(np.unique(df_friend["day"]))[0:100]
logging_conf(level=logging.INFO)

# %%
logging.debug("Running RSB2 (persist)")
df_rsb2_persist_100 = rsb2(df_friend,
                           nodes,
                           times,
                           num_seeds=NUM_SEEDS,
                           C=1,
                           gamma=0.2,
                           num_repeats_expect=1,
                           persist_params=True)
logging.debug("Running RSB2 (no persist)")
df_rsb2_nopersist_100 = rsb2(df_friend,
                             nodes,
                             times,
                             num_seeds=NUM_SEEDS,
                             C=1,
                             gamma=0.2,
                             num_repeats_expect=1,
                             persist_params=False)
# %%
logging.debug("Running TIM")
df_tim_10s_100 = tim_t(df_friend,
                       nodes,
                       times,
                       num_seeds=NUM_SEEDS,
                       num_repeats_reward=1,
                       epsilon=0.1)

# %%
logging.debug(
    "Running TIMLinUCB (5 oim reps, 1 reward rep, sigma 4, c 0.1, eps 0.25)")
df_timlinucb_5rep_100 = timlinucb(df_friend,
                                  df_feats,
                                  times,
                                  nodes,
                                  num_seeds=NUM_SEEDS,
                                  num_repeats_oim=5,
                                  num_repeats_oim_reward=1,
                                  sigma=4,
                                  c=0.1,
                                  epsilon=0.2)
# %%

# %%
results = []
num_repeats_test = 10
res_dict = {}

for t in tqdm(times):
    df_t = df_friend[df_friend["day"] <= t]
    results_t = {"time": t}
    results_t["df_rsb2_persist_100"] = get_stats_reward(
        df_t, df_rsb2_persist_100.loc[df_rsb2_persist_100["time"] == t,
                                      "selected"].item(), num_repeats_test)
    results_t["df_rsb2_nopersist_100"] = get_stats_reward(
        df_t, df_rsb2_nopersist_100.loc[df_rsb2_nopersist_100["time"] == t,
                                        "selected"].item(), num_repeats_test)
    results_t["df_timlinucb_5rep_100"] = get_stats_reward(
        df_t, df_timlinucb_5rep_100.loc[df_timlinucb_5rep_100["time"] == t,
                                        "selected"].item(), num_repeats_test)
    results_t["df_tim_10s_100"] = get_stats_reward(
        df_t, df_tim_10s_100.loc[df_tim_10s_100["time"] == t,
                                 "selected"].item(), num_repeats_test)
    results.append(results_t)
results_df = pd.DataFrame(results)
