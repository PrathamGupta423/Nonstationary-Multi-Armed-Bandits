This Repository is partly inspired by experiments done in [https://github.com/edouardfouche/G-NS-MAB](https://github.com/edouardfouche/G-NS-MAB) [Arxiv Version: https://arxiv.org/pdf/2107.11419] and developed as part of project assignment in E1 240 (Theory of Multi-armed Bandits) in AUG-DEC 2025.

The repository includes following folders:
* `Utils`: This includes helping function which are to be used in experiments. This mostly includes bandits algorithms like UCB, KL-UCB. The exhaustive list is shared later in this document.
* `Plots`: This includes results obtained from the experiments.
* `ADS_Experiments`: Since, ADS doesn't have a proof yet, we believe it would be wise to have a separate set of experiments just to test how good it can be.

The repository includes functions which implements:
1.  Thompson Sampling
2.  UCB
3.  KL-UCB
4.  RExp-3
5.  D-UCB
6.  SW-TS
7.  SW-UCB#
8.  GLR-KLUCB
9.  ADS-TS
10. ADR-KL-UCB

We consider following variants of Non-stationary environment to test the algorithm:
1.  Static
2.  Non static abrupt
3.  Non static abrupt long
4.  Non static abrupt global
5.  Non static gradual global
