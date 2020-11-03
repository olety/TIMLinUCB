# TIMLinUCB

[![DOI](https://zenodo.org/badge/256289048.svg)](https://zenodo.org/badge/latestdoi/256289048)
[![Documentation Status](https://readthedocs.org/projects/timlinucb/badge/?version=latest)](https://timlinucb.readthedocs.io/en/latest/?badge=latest)

TIMLinUCB - an algorithm for Online Influence Maximization in Temporal Networks.

## What?

An effective way of finding the most influential nodes in a temporal network without knowing everything about it (aka in an online way).

## Why?

The current (2020) literature is focused on analyzing static networks as well as temporal networks in cases where we know the "activation probability" of a node X transferring some information to the node Y. The only algorithm designed specifically for the case of Online IM in Temporal Networks is RSB. This work aims to change this by introducing TIMLinUCB - a temporal adaptation of a state-of-the-art Online IM algorithm for Static networks.

![](pictures/comparison_table.png)

## TIMLinUCB

<p align="center">
<img src="pictures/toim.png" alt="Online Influence Maximization in Temporal Networks algorithm" width="200" /><br/>
<sup>TIMLinUCB's structure (simplified)</sup>
</p>


## IMLinUCB

<p align="center">
<img src="pictures/oim.png" alt="Online Influence Maximization algorithm" width="200" /><br/>
<sup>IMLinUCB's structure (simplified)</sup>
</p>


## References

[[arXiv]](https://arxiv.org/abs/1605.06593)[IMLinUCB] Wen, Zheng, et al. "Online influence maximization under independent cascade model with semi-bandit feedback." Advances in neural information processing systems. 2017.

[[arXiv]](https://arxiv.org/abs/1404.0900)[TIM] Tang, Youze, Xiaokui Xiao, and Yanchen Shi. "Influence maximization: Near-optimal time complexity meets practical efficiency." Proceedings of the 2014 ACM SIGMOD international conference on Management of data. 2014.

[[arXiv]](https://arxiv.org/abs/1607.00653)[node2vec] Grover, Aditya, and Jure Leskovec. "node2vec: Scalable feature learning for networks." Proceedings of the 22nd ACM SIGKDD international conference on Knowledge discovery and data mining. 2016.

[[arXiv]](https://arxiv.org/abs/1604.07638)[RSB] Bao, Yixin, et al. "Online influence maximization in non-stationary social networks." 2016 IEEE/ACM 24th International Symposium on Quality of Service (IWQoS). IEEE, 2016.
