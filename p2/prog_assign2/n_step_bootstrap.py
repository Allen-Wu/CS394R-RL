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
                    G += (gamma ** n) * V[tau + n]
                V[tau] += alpha * (G - V[tau])
            if tau + 1 == T:
                break

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

    # Init the optimal policy
    optActionProb = np.zeros((env.spec.nS, env.spec.nA))
    optPolicy = np.zeros(env.spec.nS)
    # Output the optimal policy pi
    # For each state
    for s_i in range(env.spec.nS):
        # Calculate the Q value of each (state, action) pair
        q_val = []
        for a_i in range(env.spec.nA):
            # Q value
            temp_sum = 0
            for next_s in range(env.spec.nS):
                temp_sum += trans_func[s_i][a_i][next_s] * (reward_func[s_i][a_i][next_s] + gamma * V[next_s])
            q_val.append(temp_sum)
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
                    rho *= ()





                end_t = min(tau + n, T)
                G = 0
                for i in range(tau + 1, end_t + 1):
                    G += (gamma ** (i - tau - 1)) * traj[i - 1][2]
                if tau + n < T:
                    G += (gamma ** n) * V[tau + n]
                V[tau] += alpha * (G - V[tau])
            if tau + 1 == T:
                break

    return Q, pi
