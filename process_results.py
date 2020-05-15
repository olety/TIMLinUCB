import pickle

with open("results_notime.pkl", "rb") as f:
    res_notime = pickle.load(f)

with open("results_time.pkl", "rb") as f:
    res_time = pickle.load(f)

regrets_notime = [i["regrets"][0] for i in res_notime]
regrets_notime

regrets_time = [i["regrets"][0] for i in res_time]
regrets_time

len(set(res[0]["best_set_truth"]) - set(res[0]["best_set_oim"])) / len(
    res[0]["best_set_truth"])

#%%
with open("results_notime_5_2.pkl", "rb") as f:
    res_notime = pickle.load(f)

with open("results_time_5_2.pkl", "rb") as f:
    res_time = pickle.load(f)

regrets_notime = [i["regret_t"] for i in res_notime]
regrets_time = [i["regret_t"] for i in res_time]
print(f"results_time = {regrets_time}\nresults_notime = {regrets_notime}")
import numpy as np
np.array(regrets_notime) - np.array(regrets_time)

#%%

with open("eps0.5-csigmatest/results_eps0.5_sigma0.01_c0.01.pkl", "rb") as f:
    test = pickle.load(f)

test

#%%
import os
with open(
        os.path.join("eps0.4-2pparamsTrue", "results_eps0.4_sigma4_c0.1.pkl"),
        "rb") as f:
    res_time = pickle.load(f)

with open(
        os.path.join("eps0.4-2pparamsFalse", "results_eps0.4_sigma4_c0.1.pkl"),
        "rb") as f:
    res_notime = pickle.load(f)

regrets_notime = [i["regret_t"] for i in res_notime]
regrets_time = [i["regret_t"] for i in res_time]
