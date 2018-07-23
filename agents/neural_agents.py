from . import Agent 
import tensorflow as tf 
import numpy as np
import random 
from multiprocessing import Pool 

class NeuralAgent(Agent): 
    def init(self, epsilon = 1, epsilon_decay = 0.9, **kwargs): 
        ''' 
        Creates a basic neural agent. 
        
        Keyword Arguments
        -----------------
        epsilon:            float       (1) 
            The probability of choosing a random action 
        
        epsilon_decay:      float       (0.9) 
            Epsilon decay rate. Epsilon is updated by the following rule: 
            if random.random() < epsilon: 
                epsilon *= epsilon_decay 
        
        size:               int         size of input space 
        out_size:           int         size of action space 
        c_step_update:      int         (100) how many steps should run between target and primary updates
        ''' 

        # basic neural agent 
        # only works with flat input 
        self.session = tf.Session() 
        self.input_size = kwargs['size'] 
        self.output_size = kwargs['out_size'] 
        self.epsilon = epsilon  
        self.epsilon_decay = epsilon_decay
        self._train_set = [] 
        if 'c_step_update' in kwargs: 
            self._c = kwargs['c_step_update']
        else: 
            self._c = 10
        # dense layer 
        scopes = ['primary', 'target'] 
        for scope in scopes: 
            with tf.variable_scope(scope): 
                inputs = tf.placeholder(tf.float32, shape=(None, self.input_size), name="inputs") 
                with tf.variable_scope("net"): 
                    d1 = tf.layers.dense(
                        inputs=inputs, 
                        units = 16, 
                        activation = tf.nn.sigmoid, 
                        use_bias=True, 
                        name="hidden_1"
                    )        
                    d2 = tf.layers.dense(
                        inputs = d1, 
                        units = 4, 
                        activation = tf.nn.sigmoid, 
                        use_bias = True, 
                        name="hidden_2", 
                    )
                    out = tf.identity(tf.layers.dense(
                        inputs = d2, 
                        units = self.output_size, 
                        activation = tf.exp, 
                        use_bias = True, 
                        name = "output"
                    ), name="op_output")
                    self.summarize(out) 
        with tf.variable_scope('learning'): 
            # inputs necessary: 
            with tf.variable_scope('inputs'): 
                reward = tf.placeholder(dtype=tf.float32, shape=(None,), name="rewards") 
                actions = tf.placeholder(dtype = tf.float32, shape=(None, self.output_size), name="actions") 
                gamma = tf.placeholder(dtype = tf.float32, shape=(), name="gamma") 
                done_flag = tf.placeholder(dtype = tf.float32, shape = (None,), name="done_flags") 
            
            # y = reward + gamma * max_a (Q(s', a'))
            with tf.control_dependencies([tf.get_default_graph().get_tensor_by_name("{}/net/op_output:0".format(scope)) for scope in scopes]): 
                with tf.name_scope("loss"): 
                    pred = tf.reduce_sum(tf.get_default_graph().get_tensor_by_name('primary/net/op_output:0') * actions, axis=1) 
                    target_output = tf.get_default_graph().get_tensor_by_name('target/net/op_output:0') 
                    y = reward + gamma * (1. - done_flag) * tf.reduce_max(target_output, axis=1)
                    loss = tf.reduce_sum(tf.square(tf.stop_gradient(y) - pred)) 
                    self.summarize(loss) 
                    self.__train = tf.train.RMSPropOptimizer(learning_rate = 0.05, 
                        decay = 0.99, 
                        momentum = .05, 
                    ).minimize(loss) 
            self.session.run(tf.global_variables_initializer())
            self.q_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="primary")
            # Get all the variables in the Q target network.
            self.q_target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="target")

    def update_target_hard(self): 
        self.session.run([v_t.assign(v) for v_t, v in zip(self.q_target_vars, self.q_vars)])

    def do(self, observation): 
        observation = observation.reshape(1, -1) 

        if random.random() < self.epsilon: 
            action = self._action_space.sample() 

        else: 
            action = np.argmax(self.session.run("primary/net/op_output:0", feed_dict={"primary/inputs:0": observation})) 


        zeros = np.zeros((1, self.output_size))
        zeros[0][action] = 1
        self._action_cache = zeros   
        self._state_cache = observation
       
        if self._step % self._c == 0 and random.random() < self.epsilon: 
            self.epsilon *= self.epsilon_decay 
        self._step += 1
        return action
    
    def reward(self, reward, done): 
        # save rewards in a table 
        self._train_set.append((self._state_cache.reshape(1, -1), self._action_cache.reshape(1, -1), np.array([reward]).reshape((1, 1)) , np.array([done]).reshape(1, 1)))
        if len(self._train_set) > 10000: 
            self._train_set.pop(int(random.random() * len(self._train_set))) 
    
    def _generate_set(self, index):  
        state, action, reward, done = self._train_set[index] 
        if not done: 
            next_state = self._train_set[index + 1][0]
        else: 
            next_state = state 
        return np.concatenate((state, action, reward, done, next_state), axis=1)

    def train(self, batch_size): 
        # generate training dataset 
        if self._step > 1: 
            indicies = np.random.choice(np.arange(0, len(self._train_set) - 1), size=(min(len(self._train_set) - 1, batch_size), ) , replace=False).tolist() 

            set_of_res = np.concatenate(list(map(self._generate_set, indicies))) 
            # train procedure 
            _, logs = self.session.run([self.__train, self._logs], feed_dict = {
                'primary/inputs:0': set_of_res[:, 0:self.input_size], 
                'target/inputs:0': set_of_res[:, -self.input_size:], 
                'learning/inputs/actions:0': set_of_res[:, self.input_size:self.input_size + self.output_size], 
                'learning/inputs/gamma:0': 0.95, 
                'learning/inputs/rewards:0': set_of_res[:, self.input_size + self.output_size], 
                'learning/inputs/done_flags:0': set_of_res[:, self.input_size + self.output_size + 1]
            })
            self._logger.add_summary(logs, global_step=self._step)

            #hard update on c 
            if self._step % self._c == 0: 
                self.update_target_hard() 