from agents import ExponentialStochatsicAgent
import gym 
import collections 
import numpy as np 

def run_env(env_name, agent, episodes=100, steps=None, render=True, **kwargs): 
    env_mem = collections.deque(maxlen = 100) 
    env = gym.make(env_name)
    agent = agent(env.action_space, env.observation_space, **kwargs) 
    if 'batch_size' in kwargs: 
        batch_size = kwargs['batch_size']
    else: 
        batch_size = 50 
    mean = 0 
    for i in range(0, 100): 
        env_mem.append(0) 
    for ep in range(episodes): 
        observation = env.reset() 
        gen = None 
        if steps == None: 
            gen = iter(int, 1) # infinite generator 
        else: 
            gen = (i for i in range(0, steps)) 
        for count, i in enumerate(gen): 
            action = agent.do(observation) 
            observation, reward, done, _  = env.step(action) 
            agent.reward(reward, done) 
            if render: 
                env.render() 
            if done: 
                break
        mean -= env_mem.popleft()/100 
        mean += count/100 
        env_mem.append(count) 
        print(ep, count, "{:.2f}".format(mean))
        if np.mean(env_mem) > 195.0: 
            print("Environment solved in {:d} episodes! \n Average Score: {:.2f}".format(ep - 100, np.mean(env_mem)))
if __name__ == "__main__": 
    run_env('CartPole-v0', ExponentialStochatsicAgent, 2000, None,render = True, learning_rate = .01, gamma=.9, input_size=4, output_size = 2, epsilon = 1, epsilon_decay=.95,  batch_size = 100)

