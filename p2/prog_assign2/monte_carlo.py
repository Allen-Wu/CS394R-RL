from typing import Iterable, Tuple

import numpy as np
from env import EnvSpec
from policy import Policy

def off_policy_mc_prediction_ordinary_importance_sampling(
    env_spec:EnvSpec,
    trajs:Iterable[Iterable[Tuple[int,int,int,int]]],
    bpi:Policy,
    pi:Policy,
    initQ:np.array
) -> np.array:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        bpi: behavior policy used to generate trajectories
        pi: evaluation target policy
        initQ: initial Q values; np array shape of [nS,nA]
    ret:
        Q: $q_pi$ function; numpy array shape of [nS,nA]
    """

    #####################
    # TODO: Implement Off Policy Monte-Carlo prediction algorithm using ordinary importance
    # sampling (Hint: Sutton Book p. 109, every-visit implementation is fine)
    #####################
    Q = initQ
    gamma = env_spec.gamma
    # TODO: Verify this is correct or not
    # n[nS][nA] records the number of visiting times
    n = np.zeros((env_spec.nS, env_spec.nA))
    # Loop over each trajectory
    for traj in trajs:
        # Return
        G = 0
        W = 1
        # Iterate over the list in reversed order
        for step in reversed(traj):
            if W == 0:
                break
            G = gamma * G + step[2]
            n[step[0]][step[1]] += 1
            Q[step[0]][step[1]] += float(1/(n[step[0]][step[1]]) * (G - Q[step[0]][step[1]]))
            W *= float(pi.action_prob(step[0], step[1]) / bpi.action_prob(step[0], step[1]))

    return Q

def off_policy_mc_prediction_weighted_importance_sampling(
    env_spec:EnvSpec,
    trajs:Iterable[Iterable[Tuple[int,int,int,int]]],
    bpi:Policy,
    pi:Policy,
    initQ:np.array
) -> np.array:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using behavior policy bpi
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        bpi: behavior policy used to generate trajectories
        pi: evaluation target policy
        initQ: initial Q values; np array shape of [nS,nA]
    ret:
        Q: $q_pi$ function; numpy array shape of [nS,nA]
    """

    #####################
    # TODO: Implement Off Policy Monte-Carlo prediction algorithm using weighted importance
    # sampling (Hint: Sutton Book p. 110, every-visit implementation is fine)
    #####################
    Q = initQ
    C = np.zeros((env_spec.nS, env_spec.nA))
    gamma = env_spec.gamma
    # Loop over each trajectory
    for traj in trajs:
        # Return
        G = 0
        W = 1
        # Iterate over the list in reversed order
        for step in reversed(traj):
            if W == 0:
                break
            G = gamma * G + step[2]
            C[step[0]][step[1]] += W
            Q[step[0]][step[1]] += float(W / (C[step[0]][step[1]]) * (G - Q[step[0]][step[1]]))
            W *= float(pi.action_prob(step[0], step[1]) / bpi.action_prob(step[0], step[1]))
    
    return Q
