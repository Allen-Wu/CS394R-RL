from typing import Tuple

import numpy as np
from env import EnvWithModel
from policy import Policy

def value_prediction(env:EnvWithModel, pi:Policy, initV:np.array, theta:float) -> Tuple[np.array,np.array]:
    """
    inp:
        env: environment with model information, i.e. you know transition dynamics and reward function
        pi: policy
        initV: initial V(s); numpy array shape of [nS,]
        theta: exit criteria
    return:
        V: $v_\pi$ function; numpy array shape of [nS]
        Q: $q_\pi$ function; numpy array shape of [nS,nA]
    """

    #####################
    # TODO: Implement Value Prediction Algorithm (Hint: Sutton Book p.75)
    #####################
    trans_func = env.TD
    reward_func = env.R
    gamma = env.spec.gamma
    V = initV
    Q = np.zeros((env.spec.nS, env.spec.nA))

    while True:
        delta = 0
        # Loop over each state
        for s_i in range(env.spec.nS):
            old_val = V[s_i]
            # Update
            new_val = 0
            # Calculate new state value
            # For all possible actions
            for a_i in range(env.spec.nA):
                temp_sum = 0
                for next_s in range(env.spec.nS):
                    temp_sum += trans_func[s_i][a_i][next_s] * (reward_func[s_i][a_i][next_s] + gamma * V[next_s])
                Q[s_i][a_i] = temp_sum
                # q(s, a) = sigma p * (r + gamma * v(s'))
                # Scale the sum with pi(a|s)
                temp_sum *= pi.action_prob(s_i, a_i)
                new_val += temp_sum
            V[s_i] = new_val
            delta = max(delta, abs(new_val - old_val))
        if delta < theta:
            break

    return V, Q


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


def value_iteration(env:EnvWithModel, initV:np.array, theta:float) -> Tuple[np.array,Policy]:
    """
    inp:
        env: environment with model information, i.e. you know transition dynamics and reward function
        initV: initial V(s); numpy array shape of [nS,]
        theta: exit criteria
    return:
        value: optimal value function; numpy array shape of [nS]
        policy: optimal deterministic policy; instance of Policy class
    """

    #####################
    # TODO: Implement Value Iteration Algorithm (Hint: Sutton Book p.83)
    #####################
    trans_func = env.TD
    reward_func = env.R
    gamma = env.spec.gamma
    V = initV
    while True:
        delta = 0
        # Loop over each state
        for s_i in range(env.spec.nS):
            old_val = V[s_i]
            # Update
            action_val = []
            # Calculate new state value
            # For all possible actions
            for a_i in range(env.spec.nA):
                temp_sum = 0
                for next_s in range(env.spec.nS):
                    temp_sum += trans_func[s_i][a_i][next_s] * (reward_func[s_i][a_i][next_s] + gamma * V[next_s])
                # NOT scale the sum with pi(a|s)
                action_val.append(temp_sum)
            V[s_i] = max(action_val)
            delta = max(delta, abs(V[s_i] - old_val))
        if delta < theta:
            break
    
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
    return V, pi
