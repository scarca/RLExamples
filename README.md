# RL Experiments 

As I've gotten more into the field of Reinforcement Learning, I find myself wanting to try new things. Although most of the development done here is work-related and proprietary, every now and then I manage to do some exploring on the side and publish it in this repository. My hope with this is 

- To design a framework that makes it easier to implement and test new agents with new environments, increasing adaptability 
- To learn hands-on what algorithms work, what don't, and understand why, 
- And to develop an algorithm or two of my own that is comparable to the state of the art (a guy can dream, right?) 

So far, this repository contains 
- `udacitytest.py`, the copy-pasted implementation of the agent for the CartPole-v0 problem 
- An `Agent` class that has a decent framework for a synchronous, non-distributed agent 
- A `NeuralAgent` that isn't used anymore (the train() function is no longer called due to a framework change) 
- An `EpisodicNeuralAgent` that is in essence a non-convolutional [DQN](http://www.arxiv.org/abs/1312.5602) Implementation 
- A stochastic agent that was my someonewhat unsuccessful attempt at mimicing the solution provided by the Udacity team 

There's more to come! I hope to implement 
- An asyncrohonous training framework based on [A3C and other asynchronous methods](https://arxiv.org/pdf/1602.01783.pdf) 
- An implementation of A3C 
- An implementation of [Rainbow](https://arxiv.org/abs/1710.02298) 
- An implementation of [Reactor](https://arxiv.org/abs/1704.04651) 
- Some implementations of [DDPG](https://arxiv.org/abs/1509.02971), [TRPO](https://arxiv.org/abs/1502.05477), and [PPO](https://arxiv.org/abs/1707.06347) 
- Some hybrid agents that hopefully outperform anything listed here. 

If you have ideas for other things I should implement, [shoot me an email](mailto:rajshrimali@gmail.com). Especially email me if you know of an algorithm that beats Reactor, which is the best paper I've found so far. 
