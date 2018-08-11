import tensorflow as tf 
import numpy as np 
from . import Agent 
from collections import deque

class ExponentialStochatsicAgent(Agent): 
    ''' Inspiration for this agent from the top solution to CartPole 
    https://github.com/udacity/deep-reinforcement-learning/blob/master/hill-climbing/Hill_Climbing.ipynb
    
    Let's try to reproduce results within htis framework
    '''

    def init(self, input_size, output_size, noise_scale = 1e-2, noise_scale_up = 1e-4, noise_scale_down = 2, gamma=1.0, **kwargs): 
        self.input_size = input_size 
        self.output_size = output_size 
        self.noise_scale = noise_scale
        self.gamma = gamma
        self.__init_graph()
        self.session = tf.Session() 
        self.session.run(self.weights.initializer)
        self.best_R = -np.Inf 
        self.best_w = self.session.run(self.weights) 
        self.rewards = [] 
        self.noise_scale_up = noise_scale_up
        self.noise_scale_down = noise_scale_down
    def __init_graph(self): 
        with tf.variable_scope("model"): 
            # initialize weights 
            self.weights = tf.get_variable("weights", shape=(self.input_size, self.output_size), 
                initializer=tf.random_uniform_initializer(maxval=self.noise_scale))
            with tf.variable_scope("inputs"): 
                inputs = tf.placeholder(tf.float32, shape=(None, self.input_size), name="inputs") 
            prod = tf.matmul(inputs, self.weights, name="multiply")
            exp = tf.exp(prod, name="exponentiate") 
            self.forward = exp / tf.reduce_sum(exp, name="reduction")
        with tf.variable_scope("update"): 
            new_weights = tf.placeholder(tf.float32, shape = self.weights.get_shape(), name="new_weights")
            self.update = tf.assign(self.weights, new_weights, name="update") 
    # define ops 
    def do(self, observation): 
        probs = self.session.run(self.forward, feed_dict={"model/inputs/inputs:0": observation.reshape(1, -1)})
        action = np.argmax(probs.reshape((-1,)))
        return action 
    
    # define rewards 
    def reward(self, reward, done): 
        self.rewards.append(reward) 
        if done: 
            self.train() 

    def train(self): 
        discounts = [self.gamma**i for i in range(len(self.rewards) + 1)]
        R = sum([a*b for a, b in zip(discounts, self.rewards)])

        # If better weights were found: 
        if R > self.best_R: 
            self.best_R = R 
            self.best_w = self.session.run(self.weights) 
            self.noise_scale = max(self.noise_scale_up, self.noise_scale / 2) 
        else: 
            self.noise_scale = min(self.noise_scale_down, self.noise_scale * 2) 
        
        # update 
        new_weights = self.best_w + self.noise_scale * np.random.rand(*self.best_w.shape) 
        self.session.run(self.update, feed_dict = {"update/new_weights:0": new_weights}) 
        self.rewards = [] 
