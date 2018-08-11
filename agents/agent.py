import tensorflow as tf 
import uuid 

class Agent: 
    def __init__(self, action_space, observation_space, **kwargs): 
        self._action_space = action_space
        self._observation_space = observation_space 
        self._id = uuid.uuid4() 
        print("INITIALIZING", self._id)
        self._step = 0 
        self.init(**kwargs)
        self._logger = tf.summary.FileWriter(logdir="./agents/logs/{}/{}".format(self.__class__.__name__, self._id), graph=tf.get_default_graph()) 

        self._logs = tf.summary.merge_all() 

    def summarize(self, var, detailed=False): 
        with tf.variable_scope("summaries/{}".format(var.name[:-2])):
            mean = tf.reduce_mean(var) 
            with tf.name_scope('stdev'): 
                std = tf.sqrt(tf.reduce_mean(tf.square(var - mean))) 
            with tf.device('/cpu:0'): 
                tf.summary.scalar('mean', mean) 
                tf.summary.scalar('var', std) 
                tf.summary.scalar('min', tf.reduce_min(var)) 
                tf.summary.scalar('max', tf.reduce_max(var)) 
                tf.summary.histogram('hist', var) 

    def init(self, **kwargs): 
        pass 

    def do(self, observation): 
        ''' 
        Takes an action in the cycle: 
        state -> action -> reward -> state' 
        
        Requires: 
        observation: current state 

        Returns 
        --------
        action      an action in the sample space 
        ''' 
        raise NotImplementedError

    def reward(self, reward, done): 
        ''' 
        Registers a reward in the cycle 
        state -> action -> reward -> state' 
        ''' 
        raise NotImplementedError
    def train(self, batch_size): 
        ''' 
        Runs training 
        ''' 
        raise NotImplementedError