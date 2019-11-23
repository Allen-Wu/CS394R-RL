import numpy as np
from algo import ValueFunctionWithApproximation

import tensorflow as tf

class ValueFunctionWithNN(ValueFunctionWithApproximation):
    def __init__(self,
                 state_dims):
        """
        state_dims: the number of dimensions of state space
        """
        # TODO: implement this method
        # Build up the network
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
        self.Y_hat = self.neural_net(self.X)
        self.loss_op = 0.5 * tf.losses.mean_squared_error(self.Y, self.Y_hat) #loss function
        # beta1=0.9, beta2=0.999
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001, 
                                                beta1=0.9,
                                                beta2=0.999) # Adam optimizer
        self.train_op = self.optimizer.minimize(self.loss_op) # minimize losss
        self.init = tf.global_variables_initializer()
        # Init the network
        self.sess = tf.Session()
        self.sess.run(self.init)

    def neural_net(self, x):
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

    def __call__(self,s):
        # TODO: implement this method
        # Prediction
        s = np.reshape(s, (1, self.state_dims))
        pred_state_val = self.sess.run(self.Y_hat, feed_dict={self.X: s})
        return pred_state_val[0, 0]
        # with tf.Session() as sess:
        #     pred_state_val = sess.run(self.Y_hat, feed_dict={self.X: s})
        #     sess.close()
        #     return pred_state_val
        # return 0.

    def update(self,alpha,G,s_tau):
        # TODO: implement this method
        # Perform SGD
        # with tf.Session() as sess:
        #     sess.run(self.train_op, feed_dict={self.X: s_tau, self.Y: G})
        #     sess.close()
        s_tau = np.reshape(s_tau, (1, self.state_dims))
        G = np.reshape(G, (1, 1))
        self.sess.run(self.train_op, feed_dict={self.X: s_tau, self.Y: G})
        return None

