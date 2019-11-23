import numpy as np
from policy import Policy
import math

class ValueFunctionWithApproximation(object):
    def __call__(self,s) -> float:
        """
        return the value of given state; \hat{v}(s)

        input:
            state
        output:
            value of the given state
        """
        raise NotImplementedError()

    def update(self,alpha,G,s_tau):
        """
        Implement the update rule;
        w <- w + \alpha[G- \hat{v}(s_tau;w)] \nabla\hat{v}(s_tau;w)

        input:
            alpha: learning rate
            G: TD-target
            s_tau: target state for updating (yet, update will affect the other states)
        ouptut:
            None
        """
        raise NotImplementedError()

def semi_gradient_n_step_td(
    env, #open-ai environment
    gamma:float,
    pi:Policy,
    n:int,
    alpha:float,
    V:ValueFunctionWithApproximation,
    num_episode:int,
):
    """
    implement n-step semi gradient TD for estimating v

    input:
        env: target environment
        gamma: discounting factor
        pi: target evaluation policy
        n: n-step
        alpha: learning rate
        V: value function
        num_episode: #episodes to iterate
    output:
        None
    """
    #TODO: implement this function
    
    # Loop over each episode
    for episode in range(num_episode):
        # Reset the environment
        state = env.reset()
        T = math.inf
        t = 0
        # Reward map for each time stamp
        r_map = {}
        # State map for each time stamp
        s_map = {0: state}
        done = False # Whether reach terminal
        while True:
            if t < T:
                # Take an action
                a = pi.action(state)
                state, r, done, info = env.step(a)
                r_map[(t + 1)] = r
                s_map[(t + 1)] = state
                if done:
                    T = t + 1
            tau = t - n + 1
            if tau >= 0:
                G = 0
                end_t = min(tau + n, T)
                for i in range(tau + 1, end_t + 1):
                    G += (gamma ** (i - tau - 1)) * r_map[i]
                if tau + n < T:
                    G += (gamma ** n) * V(s_map[tau + n])
                # Update
                V.update(alpha, G, s_map[tau])
            if (tau + 1) == T:
                break
            t += 1
