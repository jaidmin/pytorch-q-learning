from ReplayBuffer import ReplayBuffer
from Environment import Environment
import gym
import numpy as np
from Agent import Agent
from stolen_openai_wrappers import wrap_dqn

agent = Agent(2)
_env = wrap_dqn(gym.make("PongDeterministic-v4"))
env = Environment(_env,0,False,[2,3],False,-1,1)

agent.load_weights("./current_model.torch")
# load weights if necessary
agent.train(100,4,env,1000000,400000,0.1,10000)
