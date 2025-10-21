This Repository is partly inspired by experiments done in https://github.com/edouardfouche/G-NS-MAB [Arxiv Version: https://arxiv.org/pdf/2107.11419] and developed as part of project assignment in E1 240 (Theory of Multi-armed Bandits) in AUG-DEC 2025.
The repository includes following folders:
\begin{itemize}
    \item Utils: This includes helping function which are to be used in experiments. This mostly includes bandits algorithms like UCB,KL-UCB. The exhaustive list is shared later in this document.
    \item Plots: This includes results obtained from the experiments.
    \item ADS\_Experiments: Since, ADS doesn't have a proof yet, we believe it would be wise to have a separate set of experiments just to test how good it can be.
\end{itemize}
The repository includes functions which implements:
\begin{enumerate}
    \item Thompson Sampling
    \item UCB
    \item KL-UCB
    \item RExp-3
    \item D-UCB
    \item SW-TS
    \item SW-UCB \# 
    \item GLR-KLUCB
    \item ADS-TS
    \item ADR-KL-UCB
\end{enumerate}

We consider following varinats of Non-stationary enviornment to test the algorithm:
\begin{enumerate}
    \item Static
    \item Non static abrupt 
    \item Non static abrupt long
    \item Non static abrupt global
    \item Non static gradual global
\end{enumerate}
