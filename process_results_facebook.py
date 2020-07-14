import pickle
import itertools
import pandas as pd
from functools import reduce
import matplotlib.pyplot as plt
import seaborn as sns


# Loading the files

with open("grid_par_facebook.pickle", "rb") as f:
    df = pickle.load(f)

for res in df:
    res["mean_reward"] = res["df"]["reward_best"].mean()

max(df, key=lambda x: x["mean_reward"])

# Converting to DataFrames

df = pd.DataFrame(df)
df["rewards"] = df.apply(lambda x: [x["df"]["reward_best"]], axis=1)

df_test = df[["sigma", "c", "epsilon", "mean_reward", "rewards"]]
df_test = df_test.sort_values("mean_reward", ascending=False)

# %% Making a heatmap for sigma/c, new map for every epsilon

fig = plt.figure(figsize=(20, 20))
subplots = fig.subplots(3, 2)

subplots_cont = itertools.chain([x for i in subplots for x in i])
for epsilon in sorted(df_test["epsilon"].unique()):
    axis = next(subplots_cont)
    axis.set_title(f"Epsilon = {epsilon}")
    df_heat = df_test[df_test["epsilon"] == epsilon][["sigma", "c", "mean_reward"]]
    sns.heatmap(df_heat.pivot("sigma", "c", "mean_reward"), ax=axis, vmin=5, vmax=9)


# %% Processing the num_repeats results

with open("num_reps_facebook.pickle", "rb") as f:
    df_num_repeats = pickle.load(f)

df_num_repeats

for res in df_num_repeats:
    res["mean_reward"] = res["df"]["reward_best"].mean()

df_num_repeats = pd.DataFrame(df_num_repeats)


sns.lineplot(df_num_repeats["num_repeats"], df_num_repeats["mean_reward"])


# %% Processing num_repeats_reward results

with open("num_reps_reward_facebook.pickle", "rb") as f:
    df_num_repeats_rew = pickle.load(f)

for res in df_num_repeats_rew:
    res["mean_reward"] = res["df"]["reward_best"].mean()

df_num_repeats_rew = pd.DataFrame(df_num_repeats_rew)


sns.lineplot(
    df_num_repeats_rew["num_repeats_reward"], df_num_repeats_rew["mean_reward"]
)


# %% Processing the RSB results

with open("grid_rsb.pickle", "rb") as f:
    df_rsb = pickle.load(f)

for res in df_rsb:
    res["mean_reward"] = res["df"]["reward"].mean()

max(df_rsb, key=lambda x: x["mean_reward"])
df_rsb = pd.DataFrame(df_rsb)
df_rsb["rewards"] = df_rsb.apply(lambda x: [x["df"]["reward"]], axis=1)
df_rsb = df_rsb.sort_values("mean_reward", ascending=False)

df_heat = df_rsb[["gamma", "c", "mean_reward"]]
sns.heatmap(df_heat.pivot("gamma", "c", "mean_reward"))  # , vmin=5, vmax=11)


# %% Processing num_repeats_reward results for RSB

with open("num_repeats_rsb_facebook.pickle", "rb") as f:
    df_num_repeats_rsb = pickle.load(f)

for res in df_num_repeats_rsb:
    res["mean_reward"] = res["df"]["reward"].mean()

df_num_repeats_rsb = pd.DataFrame(df_num_repeats_rsb)

df_num_repeats_rsb_plot = df_num_repeats_rsb[
    ["num_repeats_expect", "mean_reward"]
].melt("num_repeats_expect", var_name="cols", value_name="values")
sns.lineplot(x="num_repeats_expect", y="values", data=df_num_repeats_rsb_plot)

# %% Processing the comparison for 5 seeds (FB dataset)


with open("comparison_facebook_5seeds.pickle", "rb") as f:
    df_comp = pickle.load(f)

df_comp["tim_t"]
df_comp["rsb_persist"]
df_comp["rsb_nppersist"]
df_comp["timlinucb"]

for key, df in df_comp.items():
    df.columns = df.columns.map(lambda x: str(x) + "_" + key if x != "time" else x)

df_comp = reduce(lambda left, right: pd.merge(left, right, on="time"), df_comp.values())

df_comp_plot = df_comp[
    [
        "time",
        "reward_tim_t",
        "reward_rsb_persist",
        "reward_rsb_nppersist",
        "reward_best_timlinucb",
    ]
].melt("time", var_name="Algorithm", value_name="Reward")
sns.lineplot(x="time", y="Reward", hue="Algorithm", data=df_comp_plot)

# %% Processing the comprison for 20 seeds, 100 time steps


with open("comparison_facebook_20seeds_100t.pickle", "rb") as f:
    df_comp_100t = pickle.load(f)

df_comp_100t

df_comp_100t

for df in df_comp_100t:
    algo_name = df["algo_name"][0]
    df.columns = df.columns.map(
        lambda x: str(x) + "_" + algo_name if x != "time" else x
    )

df_comp_100t = reduce(
    lambda left, right: pd.merge(left, right, on="time"), df_comp_100t
)

df_comp_100t_plot = df_comp_100t[
    [
        "time",
        "reward_tim_t",
        "reward_rsb_persist",
        "reward_rsb_nopersist",
        "reward_best_timlinucb",
    ]
].melt("time", var_name="Algorithm", value_name="Reward")
sns.lineplot(x="time", y="Reward", hue="Algorithm", data=df_comp_100t_plot)

# %% Checking persistent vs non-persistent parameters (20 time steps)

with open("comparison_facebook_20seeds.pickle", "rb") as f:
    df_comp = pickle.load(f)

with open("tlu_20days_persist_facebook.pickle", "rb") as f:
    df_tlu_persist = pickle.load(f)


df_tlu_nopersist = df_comp[3][["reward_best", "time", "s_best"]]
df_tlu_nopersist.rename(
    columns={"reward_best": "reward_nopersist", "s_best": "s_nopersist"}, inplace=True
)

df_tlu_persist = df_tlu_persist[["reward_best", "time", "s_best"]]
df_tlu_persist.rename(
    columns={"reward_best": "reward_persist", "s_best": "s_persist"}, inplace=True
)

df_persist_nopersist = pd.merge(df_tlu_nopersist, df_tlu_persist, on="time")

df_persist_nopersist_plot = df_persist_nopersist[
    ["time", "reward_persist", "reward_nopersist",]
].melt("time", var_name="Algorithm", value_name="Reward")

sns.lineplot(x="time", y="Reward", hue="Algorithm", data=df_persist_nopersist_plot)


# %% Checking persistent vs non-persistent parameters (100 time steps)

with open("comparison_facebook_20seeds_100t.pickle", "rb") as f:
    df_comp = pickle.load(f)

with open("tlu_100days_persist_facebook.pickle", "rb") as f:
    df_tlu_persist = pickle.load(f)


df_tlu_nopersist = df_comp[3][["reward_best", "time", "s_best"]]
df_tlu_nopersist.rename(
    columns={"reward_best": "reward_nopersist", "s_best": "s_nopersist"}, inplace=True
)

df_tlu_persist = df_tlu_persist[["reward_best", "time", "s_best"]]
df_tlu_persist.rename(
    columns={"reward_best": "reward_persist", "s_best": "s_persist"}, inplace=True
)

df_persist_nopersist = pd.merge(df_tlu_nopersist, df_tlu_persist, on="time")

df_persist_nopersist_plot = df_persist_nopersist[
    ["time", "reward_persist", "reward_nopersist",]
].melt("time", var_name="Algorithm", value_name="Reward")

sns.lineplot(x="time", y="Reward", hue="Algorithm", data=df_persist_nopersist_plot)


# --------------------------------------------------------------------------------------
# %% -------------------------------- DIGG DATASET -------------------------------------
# --------------------------------------------------------------------------------------


with open("grid_rsb_digg.pickle", "rb") as f:
    df_rsb = pickle.load(f)

for res in df_rsb:
    res["mean_reward"] = res["df"]["reward"].mean()

max(df_rsb, key=lambda x: x["mean_reward"])
df_rsb = pd.DataFrame(df_rsb)
df_rsb["rewards"] = df_rsb.apply(lambda x: [x["df"]["reward"]], axis=1)
df_rsb = df_rsb.sort_values("mean_reward", ascending=False)

df_heat = df_rsb[["gamma", "c", "mean_reward"]]
sns.heatmap(df_heat.pivot("gamma", "c", "mean_reward"))  # , vmin=5, vmax=11)


# %% TLU for digg


with open("grid_par_digg.pickle", "rb") as f:
    df = pickle.load(f)

for res in df:
    res["mean_reward"] = res["df"]["reward_best"].mean()

max(df, key=lambda x: x["mean_reward"])

# Converting to DataFrames

df = pd.DataFrame(df)
df["rewards"] = df.apply(lambda x: [x["df"]["reward_best"]], axis=1)

df_test = df[["sigma", "c", "epsilon", "mean_reward", "rewards"]]
df_test = df_test.sort_values("mean_reward", ascending=False)

fig = plt.figure(figsize=(20, 20))
subplots = fig.subplots(3, 2)

subplots_cont = itertools.chain([x for i in subplots for x in i])
for epsilon in sorted(df_test["epsilon"].unique()):
    axis = next(subplots_cont)
    axis.set_title(f"Epsilon = {epsilon}")
    df_heat = df_test[df_test["epsilon"] == epsilon][["sigma", "c", "mean_reward"]]
    sns.heatmap(df_heat.pivot("sigma", "c", "mean_reward"), ax=axis)


# %% Processing the comparison for 20 seeds/100t (Digg dataset)


with open("comparison_digg_20seeds_100t.pickle", "rb") as f:
    df_comp = pickle.load(f)

df_comp["tim_t"]
df_comp["rsb_persist"]
df_comp["rsb_nppersist"]
df_comp["timlinucb"]

for key, df in df_comp.items():
    df.columns = df.columns.map(lambda x: str(x) + "_" + key if x != "time" else x)

df_comp = reduce(lambda left, right: pd.merge(left, right, on="time"), df_comp.values())

df_comp_plot = df_comp[
    [
        "time",
        "reward_tim_t",
        "reward_rsb_persist",
        "reward_rsb_nppersist",
        "reward_best_timlinucb",
    ]
].melt("time", var_name="Algorithm", value_name="Reward")
sns.lineplot(x="time", y="Reward", hue="Algorithm", data=df_comp_plot)
