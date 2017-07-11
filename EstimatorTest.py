from Estimator import Agent
import gym
from utils import preprocess
import torch.autograd as autograd
from torch.autograd import Variable
from torch import FloatTensor, LongTensor

import numpy as np
import torch
agent = Agent(2)

env = gym.make("Pong-v4")
state = np.expand_dims(np.expand_dims(preprocess(env.reset()),0),0)
#inp = autograd.Variable(torch.FloatTensor([[state.tolist()]]))
inp = Variable(torch.from_numpy(state).float())

print(inp)
print(inp.size())
q_pred = agent.predict_q_values(inp)
print(q_pred)

targ = Variable(FloatTensor([[1]]))
actions = Variable(LongTensor([[1]]))


for i in range(100):
    agent.accumulate_gradients(inp,targ,actions)

    agent.update_parameters()

