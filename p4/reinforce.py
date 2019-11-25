from typing import Iterable
import numpy as np
import tensorflow as tf

class LogError(tf.compat.v1.keras.losses.Loss):
    def call(self, y_prob, gamma_t, delta, temp):
        return -1 * gamma_t * delta * np.log(y_prob)

class PiApproximationWithNN():
    def __init__(self,
                 state_dims,
                 num_actions,
                 alpha):
        """
        state_dims: the number of dimensions of state space
        action_dims: the number of possible actions
        alpha: learning rate
        """
        # TODO: implement here

        # Tips for TF users: You will need a function that collects the probability of action taken
        # actions; i.e. you need something like
        #
            # pi(.|s_t) = tf.constant([[.3,.6,.1], [.4,.4,.2]])
            # a_t = tf.constant([1, 2])
            # pi(a_t|s_t) =  [.6,.2]
        #
        # To implement this, you need a tf.gather_nd operation. You can use implement this by,
        #
            # tf.gather_nd(pi,tf.stack([tf.range(tf.shape(a_t)[0]),a_t],axis=1)),
        # assuming len(pi) == len(a_t) == batch_size

        # Build up the network
        self.num_actions = num_actions
        self.state_dims = state_dims
        with tf.name_scope("model_inputs"):
            # raw state representation
            self.X = tf.compat.v1.placeholder(tf.float32, (None, self.state_dims), name="states")
        # self.Y = tf.placeholder('float', [None, num_actions]) # Single output of state value
        self.weights = {
            'h1': tf.Variable(tf.random_normal([state_dims, 32])),
            'h2': tf.Variable(tf.random_normal([32, 32])),
            'out': tf.Variable(tf.random_normal([32, num_actions]))
        }
        # No bias
        self.biases = {
            'b1': tf.Variable(tf.constant(0.0, shape=[32])),
            'b2': tf.Variable(tf.constant(0.0, shape=[32])),
            'out': tf.Variable(tf.constant(0.0, shape=[num_actions]))
        }
        with tf.name_scope("predict_actions"):
            with tf.compat.v1.variable_scope("policy_network"):
                self.Y_hat = self.policy_neural_net(self.X)
            self.action_scores = tf.identity(self.Y_hat, name="action_scores")
            self.predicted_actions = tf.compat.v1.random.categorical(self.action_scores, 1)
        # self.action_loss = []
        # for a in range(num_actions):
        #     self.action_loss.append(-1 * tf.math.log(self.Y_hat[0, a]))
        # self.action_prob = tf.placeholder('float', [1])
        with tf.name_scope("compute_gradient"):
            self.a = tf.placeholder(tf.int32, shape=(None,))
            self.discount = tf.placeholder(tf.float32, shape=(None,))
            # self.gamma_t = tf.placeholder(tf.float32, shape=(None,))

            # self.loss_op = tf.reduce_mean(-1 * self.delta * self.gamma_t * tf.math.log(tf.gather_nd(self.Y_hat, tf.stack([tf.range(tf.shape(self.a)[0]), self.a],axis=1))))

            # self.loss_op = tf.reduce_mean(tf.stop_gradient(self.discount) * tf.nn.sparse_softmax_cross_entropy_with_logits(
            #                               logits=self.Y_hat, labels=self.a), name='loss_pi')

            # beta1=0.9, beta2=0.999
            self.optimizer = tf.train.AdamOptimizer(learning_rate=alpha, 
                                                    beta1=0.9,
                                                    beta2=0.999) # Adam optimizer
                
            self.temp = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.Y_hat,
                                                                       labels=self.a)
            self.loss_op = tf.reduce_mean(self.temp)

            # compute gradients
            self.gradients = self.optimizer.compute_gradients(self.loss_op)

            # compute policy gradients
            for i, (grad, var) in enumerate(self.gradients):
                if grad is not None:
                    self.gradients[i] = (grad * self.discount, var)

        self.train_op = self.optimizer.apply_gradients(self.gradients)
        
        # self.train_op = self.optimizer.minimize(self.loss_op) # minimize losss

        self.init = tf.global_variables_initializer()
        # Init the network
        self.sess = tf.Session()
        self.sess.run(self.init)

    def policy_neural_net(self, x):
        # Hidden layer 1
        # layer_1 = tf.matmul(x, self.weights['h1'])
        layer_1 = tf.add(tf.matmul(x, self.weights['h1']), self.biases['b1'])
        layer_1 = tf.nn.relu(layer_1) # Add activation layer
        # Hidden layer 2
        # layer_2 = tf.matmul(layer_1, self.weights['h2'])
        layer_2 = tf.add(tf.matmul(layer_1, self.weights['h2']), self.biases['b2'])
        layer_2 = tf.nn.relu(layer_2) # Add activation layer
        out_layer = tf.add(tf.matmul(layer_2, self.weights['out']), self.biases['out'])

        # out_layer = tf.nn.softmax(out_layer)
        return out_layer
        

    def __call__(self,s) -> int:
        # TODO: implement this method
        # s = np.reshape(s, (1, self.state_dims))
        pred_state_val = self.sess.run(self.predicted_actions, feed_dict={self.X: [s]})
        # TODO: Could be problematic
        # return np.argmax(pred_state_val[0, :])
        return pred_state_val[0, 0]

    def update(self, s, a, gamma_t, delta):
        """
        s: state S_t
        a: action A_t
        gamma_t: gamma^t
        delta: G-v(S_t,w)
        """
        # TODO: implement this method
        # s = np.reshape(s, (1, self.state_dims))
        # a = np.reshape(a, (1, 1))
        # gamma_t = np.reshape(gamma_t, (1, 1))
        # delta = np.reshape(delta, (1, 1))
        # loss_op = LogError(self.Y_hat[0, a], gamma_t, delta)
        # train_op = self.optimizer.minimize(loss_op) # minimize losss
        self.sess.run(self.train_op, feed_dict={self.X: [s],
                                                self.a: [a],
                                                self.discount: [gamma_t * delta]})


class Baseline(object):
    """
    The dumbest baseline; a constant for every state
    """
    def __init__(self,b):
        self.b = b

    def __call__(self,s) -> float:
        return self.b

    def update(self,s,G):
        pass

class VApproximationWithNN(Baseline):
    def __init__(self,
                 state_dims,
                 alpha):
        """
        state_dims: the number of dimensions of state space
        alpha: learning rate
        """
        # TODO: implement here
        self.state_dims = state_dims
        self.X = tf.placeholder('float', [None, state_dims])
        self.Y = tf.placeholder('float', [None, 1]) # Single output of state value
        self.weights = {
            'h1': tf.Variable(tf.random_normal([state_dims, 32])),
            'h2': tf.Variable(tf.random_normal([32, 32])),
            'out': tf.Variable(tf.random_normal([32, 1]))
        }
        # No bias
        self.biases = {
            'b1': tf.Variable(tf.random_normal([32])),
            'b2': tf.Variable(tf.random_normal([32])),
            'out': tf.Variable(tf.random_normal([1]))
        }
        self.Y_hat = self.value_neural_net(self.X)
        self.loss_op = 0.5 * tf.losses.mean_squared_error(self.Y, self.Y_hat) #loss function
        # beta1=0.9, beta2=0.999
        self.optimizer = tf.train.AdamOptimizer(learning_rate=alpha, 
                                                beta1=0.9,
                                                beta2=0.999) # Adam optimizer
        self.train_op = self.optimizer.minimize(self.loss_op) # minimize losss
        self.init = tf.global_variables_initializer()
        # Init the network
        self.sess = tf.Session()
        self.sess.run(self.init)

    def value_neural_net(self, x):
        # Hidden layer 1
        # layer_1 = tf.matmul(x, self.weights['h1'])
        layer_1 = tf.add(tf.matmul(x, self.weights['h1']), self.biases['b1'])
        layer_1 = tf.nn.relu(layer_1) # Add activation layer
        # Hidden layer 2
        # layer_2 = tf.matmul(layer_1, self.weights['h2'])
        layer_2 = tf.add(tf.matmul(layer_1, self.weights['h2']), self.biases['b2'])
        layer_2 = tf.nn.relu(layer_2) # Add activation layer
        out_layer = tf.matmul(layer_2, self.weights['out']) + self.biases['out']
        return out_layer

    def __call__(self,s) -> float:
        # TODO: implement this method
        s = np.reshape(s, (1, self.state_dims))
        pred_state_val = self.sess.run(self.Y_hat, feed_dict={self.X: s})
        return pred_state_val[0, 0]

    def update(self,s,G):
        # TODO: implement this method
        s = np.reshape(s, (1, self.state_dims))
        G = np.reshape(G, (1, 1))
        self.sess.run(self.train_op, feed_dict={self.X: s, self.Y: G})
        return None


def REINFORCE(
    env, #open-ai environment
    gamma:float,
    num_episodes:int,
    pi:PiApproximationWithNN,
    V:Baseline) -> Iterable[float]:
    """
    implement REINFORCE algorithm with and without baseline.

    input:
        env: target environment; openai gym
        gamma: discount factor
        num_episode: #episodes to iterate
        pi: policy
        V: baseline
    output:
        a list that includes the G_0 for every episodes.
    """
    # TODO: implement this method
    res = []
    for episode in range(num_episodes):
        # Loop over each episode
        # Generate an episode
        state = env.reset()
        next_a = pi(state)
        done = False
        s_map = {}
        r_map = {}
        a_map = {}
        t = 0
        s_map[t] = state
        a_map[t] = next_a
        while not done:
            state, r, done, info = env.step(next_a)
            t += 1
            s_map[t] = state
            r_map[t] = r
            next_a = pi(state)
            a_map[t] = next_a
        T = t
        for t in range(T):
            G = 0
            for k in range(t + 1, T + 1):
                G += (gamma ** (k - t - 1)) * r_map[k]
            if t == 0:
                # Return G0
                res.append(G)
            delta = G - V(s_map[t])
            # Update value network
            V.update(s_map[t], G)
            # Update policy network
            pi.update(s_map[t], a_map[t], (gamma ** t), delta)

    return res