from .agent import Agent 

class RandomAgent(Agent): 
    def do(self, observation): 
        return self._action_space.sample() 
    
    def reward(self, reward, done): 
        pass 

    def train(self): 
        return NotImplementedError
