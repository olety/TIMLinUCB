import pickle
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Loading the files

with open("grid_facebook_smalleps.pickle", "rb") as f:
    df_smalleps = pickle.load(f)

with open("grid_facebook_bigeps.pickle", "rb") as f:
    df_bigeps = pickle.load(f)

with open("grid_facebook.pickle", "rb") as f:
    df = pickle.load(f)

# Merging the arrays

df_all = df + df_smalleps + df_bigeps

for res in df_all:
    res["mean_reward"] = res["df"]["reward_best"].mean()

max(df_all, key=lambda x: x["mean_reward"])

# Converting to DataFrames

df_all = pd.DataFrame(df_all)
df_all["rewards"] = df_all.apply(lambda x: [x["df"]["reward_best"]], axis=1)

df_test = df_all[["sigma", "c", "epsilon", "mean_reward", "rewards"]]
df_test = df_test.sort_values("mean_reward", ascending=False)


# Making a heatmap for sigma/c, new map for every epsilon

fig = plt.figure(figsize=(36, 15))
subplots = fig.subplots(3, 3)

subplots_cont = itertools.chain([x for i in subplots for x in i])
for epsilon in sorted(df_test["epsilon"].unique()):
    axis = next(subplots_cont)
    axis.set_title(f"Epsilon = {epsilon}")
    df_heat = df_test[df_test["epsilon"] == epsilon][["sigma", "c", "mean_reward"]]
    sns.heatmap(df_heat.pivot("sigma", "c", "mean_reward"), ax=axis, vmin=5, vmax=11)


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
