from typing import Iterable, Tuple

import numpy as np
from env import EnvSpec
from policy import Policy

def on_policy_n_step_td(
    env_spec:EnvSpec,
    trajs:Iterable[Iterable[Tuple[int,int,int,int]]],
    n:int,
    alpha:float,
    initV:np.array
) -> Tuple[np.array]:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        n: how many steps?
        alpha: learning rate
        initV: initial V values; np array shape of [nS]
    ret:
        V: $v_pi$ function; numpy array shape of [nS]
    """

    #####################
    # TODO: Implement On Policy n-Step TD algorithm
    # sampling (Hint: Sutton Book p. 144)
    #####################
    V = initV
    gamma = env_spec.gamma
    # Loop for each episode
    for traj in trajs:
        T = len(traj) # T = t + 1, S_{t+1} is the terminal state
        t = 0
        while True:
            tau = t - n + 1
            if tau >= 0:
                end_t = min(tau + n, T)
                G = 0
                for i in range(tau + 1, end_t + 1):
                    G += (gamma ** (i - tau - 1)) * traj[i - 1][2]
                if tau + n < T:
                    G += (gamma ** n) * V[traj[tau + n][0]]
                V[traj[tau][0]] += alpha * (G - V[traj[tau][0]])
            if tau + 1 == T:
                break
            t += 1

    return V

# Derived class: Optimal policy
class OptimalPolicy(Policy):

    def __init__(self, optActionProb, optPolicy):
        # numpy array of shape [nS, nA]
        self.optActionProb = optActionProb
        # numpy array of shape [nS]
        self.optPolicy = optPolicy

    def action_prob(self, state, action):
        return self.optActionProb[state][action]

    def action(self, state):
        return self.optPolicy[state]



def off_policy_n_step_sarsa(
    env_spec:EnvSpec,
    trajs:Iterable[Iterable[Tuple[int,int,int,int]]],
    bpi:Policy,
    n:int,
    alpha:float,
    initQ:np.array
) -> Tuple[np.array,Policy]:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        bpi: behavior policy used to generate trajectories
        n: how many steps?
        alpha: learning rate
        initQ: initial Q values; np array shape of [nS,nA]
    ret:
        Q: $q_star$ function; numpy array shape of [nS,nA]
        policy: $pi_star$; instance of policy class
    """

    #####################
    # TODO: Implement Off Policy n-Step SARSA algorithm
    # sampling (Hint: Sutton Book p. 149)
    #####################
    Q = initQ
    gamma = env_spec.gamma

    # Init the optimal policy with greedy approach
    optActionProb = np.zeros((env_spec.nS, env_spec.nA))
    optPolicy = np.zeros(env_spec.nS)
    # For each state
    for s_i in range(env_spec.nS):
        # Calculate the Q value of each (state, action) pair
        q_val = []
        for a_i in range(env_spec.nA):
            # Q value
            q_val.append(Q[s_i][a_i])
        # Find argmax_a
        optimal_action = q_val.index(max(q_val))
        optActionProb[s_i][optimal_action] = 1.0
        optPolicy[s_i] = optimal_action
    
    pi = OptimalPolicy(optActionProb, optPolicy)

    # Loop for each episode
    for traj in trajs:
        T = len(traj) # T = t + 1, S_{t+1} is the terminal state
        t = 0
        while True:
            tau = t - n + 1
            if tau >= 0:
                # Calculate rho_{tau+1 : tau+n}
                rho = 1
                end_t = min(tau + n, T - 1)
                for i in range(tau + 1, end_t + 1):
                    s_i = traj[i][0]
                    a_i = traj[i][1]
                    rho *= float(pi.action_prob(s_i, a_i) / bpi.action_prob(s_i, a_i))
                G = 0
                end_t = min(tau + n, T)
                for i in range(tau + 1, end_t + 1):
                    G += (gamma ** (i - tau - 1)) * traj[i - 1][2]
                if tau + n < T:
                    G += (gamma ** n) * Q[traj[tau + n][0]][traj[tau + n][1]]
                Q[traj[tau][0]][traj[tau][1]] += alpha * rho * (G - Q[traj[tau][0]][traj[tau][1]])
                # Update pi
                q_val = []
                s_i = traj[tau][0]
                for a_i in range(env_spec.nA):
                    # Q value
                    q_val.append(Q[s_i][a_i])
                # Find argmax_a
                optimal_action = q_val.index(max(q_val))
                pi.optActionProb[s_i][optimal_action] = 1.0
                pi.optPolicy[s_i] = optimal_action
            if tau + 1 == T:
                break
            t += 1

    return Q, pi
