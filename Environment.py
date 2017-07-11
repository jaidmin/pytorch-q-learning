import PIL.Image as Image
import numpy as np


class Environment:
    def __init__(self, env, id, preproc, legal_action_set, do_rescale_reward, reward_max, rewards_min):
        """"Environment wrapper, follows the GYM api. Is used to handle things like preprocessing and the problem of actions
        (for example: the gym pong game has 6 legal actions, 2 of which dont have any effect and then 2 up and 2 down, it would be better
        if just the 2 relevant actions were available.) this is done via legal action set which contains the scalar actions which eventually
        be taaken. everywhere else the actions are still represented as 0,1,2,.. num_actions
        note: deprecated --> will be replaced with gym.wrappers api... didnt even know that existed"""


        self.env = env
        self.id = id
        self.preproc = preproc
        self.legal_action_set = legal_action_set
        self.reward_max = reward_max
        self.reward_min = rewards_min
        self.do_rescale_reward = do_rescale_reward

    @staticmethod
    def preprocess(image_arr):
        """"preprocess the image according to the original atari paper (silver et al) note: do benchmarking of this vs numpy
        preprocessing! note: at the moment i only use one image not multiple images per state"""
        img = Image.fromarray(image_arr, mode="RGB")
        img = img.convert("L")
        img = img.resize((84, 110))
        img = img.crop((0, 13, 84, 97))
        img_arr = np.expand_dims(np.asarray(img.getdata(), dtype=np.uint8).reshape((84, 84)),0)
        return img_arr

    def rescale_reward(self,reward):
        """reward rescaling, if i optimize performance at some point, i might change it and do this in the main training loop"""
        if reward > self.reward_max:
            reward = self.reward_max
        else:
            if reward < self.reward_min:
                reward = self.reward_min
        return reward


    def reset(self):
        return self.env.reset()

    def step(self, action):
        action = self.legal_action_set[action]
        next_state, reward, done, _ = self.env.step(action)
        return next_state, reward, done

    def render(self):
        self.env.render()
