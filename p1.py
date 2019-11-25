import random
import numpy as np
from matplotlib import pyplot as plt
import sys

# Sample average formula: Q_{n+1} = Q_n + 1/n * [R_n - Q_n]
# alpha = 0.1 update formula: Q_{n+1} = Q_n + alpha * [R_n - Q_n]

def ten_arm_bandit(sample_avg):
    alpha = 0.1
    epsilon = 0.1
    num_iter = 10000
    num_run = 300

    # Average reward at each iteration/step
    # Should be normalized by num_run
    # avg_step_reward = [0.0 for x in range(num_iter)]
    avg_step_reward = np.zeros(num_iter)
    # Average ratio of picking up optimal action at each iteration/step
    # Should be normalized by num_run
    # avg_opt_ratio = [0.0 for x in range(num_iter)]
    avg_opt_ratio = np.zeros(num_iter)

    for i in range(num_run):
        # Start each simulation
        true_values = [0.0 for x in range(10)]

        # Number of occurances for each action
        num_action = [0.0 for x in range(10)]
        q_estimate = [0.0 for x in range(10)]
        for j in range(num_iter):
            # Pick the action
            action_idx = -1
            # Perform epsilon greedy action selection
            if random.random() <= epsilon:
                # Randomly pick an action
                action_idx = random.randint(0, 9)
            else:
                max_est = max(q_estimate)
                max_est_idx = [i for i, x in enumerate(q_estimate) if x == max_est]
                action_idx = random.choice(max_est_idx)
            # Perform action
            # Get reward, with noise around the true reward
            r = np.random.normal(true_values[action_idx], 1)
            avg_step_reward[j] += r
            if true_values[action_idx] == max(true_values):
                # If choosing the optimal action
                avg_opt_ratio[j] += 1
            # Update estimate
            num_action[action_idx] += 1
            if sample_avg:
                # Sample average
                q_estimate[action_idx] = q_estimate[action_idx] + \
                                        1/num_action[action_idx] * \
                                        (r - q_estimate[action_idx])
            else:
                # const step size
                q_estimate[action_idx] = q_estimate[action_idx] + \
                                         alpha * (r - q_estimate[action_idx])
            # Update true values
            for i in range(len(true_values)):
                s = np.random.normal(0, 0.01)
                true_values[i] += s
                # print(s)
        # print("{0:.5f}".format(true_values[0]))
    avg_step_reward[:] = [x / num_run for x in avg_step_reward]
    avg_opt_ratio[:] = [x / num_run for x in avg_opt_ratio]

    return avg_step_reward, avg_opt_ratio


# fig,axes = plt.subplots(2,1)
# axes[0].plot(range(1, num_iter + 1), avg_step_reward)
# axes[1].plot(range(1, num_iter + 1), avg_opt_ratio)
# plt.show()

# np.savetxt('out.txt', np.array(avg_step_reward))
# np.savetxt('out.txt', np.array(avg_opt_ratio))

# print("{0:.5f}".format(true_values[0]))
# print(avg_step_reward)

output_file = sys.argv[1]
sample_avg_reward, sample_avg_opt = ten_arm_bandit(True)
const_step_reward, const_step_opt = ten_arm_bandit(False)
np.savetxt(output_file, np.array([sample_avg_reward, sample_avg_opt, const_step_reward, const_step_opt]))
# np.savetxt(output_file, np.array(const_step_reward))
# np.savetxt(output_file, np.array(const_step_opt))
