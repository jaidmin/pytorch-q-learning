from Estimator import Estimator, LinearEstimator
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from ReplayBuffer import ReplayBuffer
from torch.autograd import Variable
import torch.nn.functional as F

import sys
import time
import datetime

class Agent:
    """for optimization reduce number of passes throught estimator (will be done later) """

    def __init__(self,  num_actions,cart_pole=False, double_q=True):
        self.num_actions = num_actions
        self.double_q = double_q

        if cart_pole:
            self.estimator =  LinearEstimator(num_actions)
            self.target_estimator =  LinearEstimator(num_actions)
            self.buffer = ReplayBuffer(10000)

        else:
            self.estimator = Estimator(num_actions)
            self.target_estimator = Estimator(num_actions)
            self.buffer = ReplayBuffer(1000000)

        self.estimator.cuda()
        self.target_estimator.cuda()


        self.optimizer = optim.RMSprop(self.estimator.parameters(), lr=0.0001)
        self.optimizer.zero_grad()

        self.mse = nn.MSELoss()

        self.synchronize_target_estimator()

    def load_weights(self, weights_file):
        print("loading weights from: {}".format(weights_file))
        self.estimator.load_state_dict(torch.load(weights_file))
        self.synchronize_target_estimator()

    def get_epsilon(self,i,decay_until_step, decay_until_value):
        if i >= decay_until_step:
            return decay_until_value
        else:
            return 1 - (i / decay_until_step) * (1 - decay_until_value)

    def predict_q_values(self, states):
        """
        computes q_values by passing through nn
        :param states: np.array (batch_size, channels, height, width)
        :return: torch.FloatTensor of shape (batch_size, self.num_actions)
        """
        states = Variable(torch.from_numpy(states).float()).cuda()
        return self.estimator.forward(states)

    def choose_e_greedy_action(self, state, epsilon):
        """
        choose greedy action with p(1-eps) else choose random action
        :param states: np.array of shape (batch_size, channels, height, width)
        :param epsilon: float
        :return:
        """
        random = np.random.choice([0,1],p=(epsilon,(1-epsilon)))
        if random == 0:
            return np.random.choice(range(self.num_actions))
        else:
            return self.choose_greedy_action(state)

    def choose_greedy_action(self, state):
        state = np.expand_dims(state,0)
        return np.argmax(self.predict_q_values(state).data.cpu().numpy())

    def update(self, x, targ, actions):
        """
        computes loss; prints loss; does backprop
        :param x: numpy.array of shape (batch_size, channels, height, width)
        :param targ: numpy array shape (batch_size)
        :param actions: numpy array shape (batch_size)
        :return: nothing
        """
        targ = Variable(torch.unsqueeze(torch.from_numpy(targ).float(),-1)).cuda()
        actions = Variable(torch.unsqueeze(torch.from_numpy(actions).long(),-1)).cuda()

        out = self.predict_q_values(x)
        affected_out = torch.gather(out, 1, actions)
        loss = self.mse(affected_out, targ)

        self.optimizer.zero_grad()
        loss.backward()
        #for param in self.estimator.parameters():
        #    param.grad.data.clamp_(-1, 1)
        self.optimizer.step()



    def save_model(self):
        with open("./trained_model.torch", "wb") as f:
            print("saving model: training has succeeded")
            torch.save(self.estimator.state_dict(), f)
            curr_time = time.time()
            print("Current Time: " + time.asctime(time.localtime(curr_time)))
            print("Ran for: " + str(datetime.timedelta(seconds=curr_time - self.start_time)))
            print("Now solved!")
            sys.exit("not an error, training was successful")


    def save_model_during_training(self):
        with open("./current_model.torch", "wb") as f:
            print("saving current_model")
            print("\n")
            torch.save(self.estimator.state_dict(), f)






    def train(self, start_training, batch_size, env, nr_episodes,  decay_until_step, decay_until_value, update_targ_freq):
        self.update_targ_freq = update_targ_freq
        self.start_time = time.time()
        print("Starting training loop at: "+ time.asctime(time.localtime(self.start_time)))
        total_steps = 0
        running_episode_reward = 0
        print("populating replay buffer... ")
        state = env.reset() # return (1,84,84)
        for i in range(start_training):
            eps  = 1
            action = self.choose_e_greedy_action(state,eps)
            next_state, reward, done = env.step(action)
            self.buffer.add(state,action,reward,done,next_state)
            state = next_state
            if done:
                env.reset()
        print("replay buffer populated with {} transitions, learning begins...".format(self.buffer.count))

        for i in range(1, nr_episodes):
            done = False
            state = env.reset()  # return (1,84,84)
            episode_reward = 0
            episode_counter = 0
            #if running_episode_reward > 150:
            #    self.save_model()



            if ((i % 1000) == 0):
                curr_time = time.time()
                print("Current Time: " + time.asctime(time.localtime(curr_time)))
                print("Running for: " + str(datetime.timedelta(seconds=curr_time - self.start_time)))
                print("\n")


            while not done:
                if (total_steps % self.update_targ_freq) == 0:
                    print("synchronizing target estimator !")
                    self.synchronize_target_estimator()


                if (i % 100) == 0:
                    self.save_model_during_training()

                eps = self.get_epsilon(total_steps, decay_until_step, decay_until_value)
                action = self.choose_e_greedy_action(state, eps)
                next_state, reward, done = env.step(action)
                total_steps += 1
                self.buffer.add(state, action, reward, done, next_state)
                s_batch, a_batch, r_batch, d_batch, s2_batch = self.buffer.sample_batch(batch_size)

                if(self.double_q == True):
                    q_targets = self.calculate_double_q_targets(s2_batch,r_batch,d_batch)
                else:
                    q_targets = self.calculate_q_targets(s2_batch,r_batch,d_batch)

                self.update(s_batch,q_targets,a_batch)
                state = next_state
                episode_counter += 1
                episode_reward += reward
                if done:
                    running_episode_reward = running_episode_reward * 0.9 + 0.1 * episode_reward

                    if (i % 10) == 0:
                        print("\n")

                        print("global step: {}".format(total_steps))
                        print("episode: {}".format(i))
                        print("running reward: {}".format(round(running_episode_reward,2)))
                        print("current epsilon: {}".format(round(eps,2)))
                        print("episode_length: {}".format(episode_counter))
                        print("episode reward: {}".format(episode_reward))


                        print("\n")




    def predict_q_targets_values(self, states):
        """
        computes q_values by passing through nn
        :param states: np.array (batch_size, channels, height, width)
        :return: torch.FloatTensor of shape (batch_size, self.num_actions)
        """
        states = Variable(torch.from_numpy(states).float()).cuda()
        return self.target_estimator.forward(states)



    def calculate_q_targets(self,next_states, rewards, done):
        done_mask = done == 1
        next_max_q_values = np.max(self.predict_q_targets_values(next_states).data.cpu().numpy(), axis=1)
        next_max_q_values[done_mask] = 0
        q_targets = rewards + next_max_q_values
        return q_targets

    def calculate_double_q_targets(self, next_states, rewards, done):
        done_mask = done == 1
        next_q_values_target = self.predict_q_targets_values(next_states).data.cpu().numpy()
        next_q_values_online = self.predict_q_values(next_states).data.cpu().numpy()
        next_q_values_online_argmax = np.argmax(next_q_values_online, axis=1)
        q_term = next_q_values_target[list(range(len(next_q_values_online_argmax))), next_q_values_online_argmax]
        q_term[done_mask] = 0
        q_targets = rewards + q_term
        return q_targets

    def synchronize_target_estimator(self):
        primary_weights = list(self.estimator.parameters())
        secondary_weights = list(self.target_estimator.parameters())
        n = len(primary_weights)
        for i in range(0, n):
            secondary_weights[i].data[:] = primary_weights[i].data[:]

