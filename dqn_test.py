# import relevant packages
import torch
import random
from torch import nn
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt 
import gym
import pandas as pd
import numpy as np
import cv2
import os
import random
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack, TransformObservation
import pygame

# ------- Copy set up code from notebook ----------
# get code from DQN notebook to set up
class DQN_model(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN_model, self).__init__()
        self.layer1 = nn.Conv2d(input_shape[0], 16, kernel_size=(8,8), stride=4)
        self.layer2 = nn.Conv2d(16, 32, (4,4), stride=2)
        # output shape after EACH convo would be ((dimension - filter size)/stride +1) **2 (for 2 sides)
                                                                            # * 4 (stack) * output_channel
        dim_size = (((84-8)/4 + 1)-4)/2+1
        self.layer3 = nn.Linear(int((dim_size)**2 * 32), 256)
        self.output = nn.Linear(256, n_actions) 
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.layer3(x))
        x = self.output(x)
        
        return x
    
def choose_action(model, state, device, epsilon=0.001):
    if random.random()<=epsilon: #exploration
        return env.action_space.sample()
    else:
#         squeeze to remove last dim of 1 (for gray scaled val) and add 1 dim at first (1 input instead of batch)
        state = torch.Tensor(state).squeeze().unsqueeze(0).to(device)
        # predict
        pred = model(state)
        return int(torch.argmax(pred.squeeze()).item())
    

# Wrapper to clip reward, taken from documentation
class ClipReward(gym.RewardWrapper):
    def __init__(self, env, min_reward, max_reward):
        super().__init__(env)
        self.min_reward = min_reward
        self.max_reward = max_reward
        self.reward_range = (min_reward, max_reward)
    
    def reward(self, reward):
        return np.clip(reward, self.min_reward, self.max_reward)
    
# observation wrapper for cropping
class AtariCropping(gym.ObservationWrapper):
    def __init__(self, env):
        """A gym wrapper that crops image"""
        super().__init__(env)
        
        old_shape = env.observation_space.shape
        # get new shape after cropping
        new_shape = (old_shape[0]-50,) + old_shape[1:]
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=new_shape)

    def observation(self, img):
        """what happens to each observation"""
        # crop image (top and bottom, top from 34, bottom remove last 16)
        img = img[34:-16, :, :]
        return img
    
class RescaleRange(gym.ObservationWrapper):
    def __init__(self, env):
        """A gym wrapper that rescale low and high value"""
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=env.observation_space.shape)

    def observation(self, img):
        """what happens to each observation"""
        # rescale value from range 0-255 to 0-1
        img = img.astype('float32') / 255.   
        return img

def generate_env(env_name):
    env = gym.make(env_name)
    env = ClipReward(env, -1, 1)
    env = AtariCropping(env)
    # gray scale frame
    env = GrayScaleObservation(env, keep_dim=False)
    env = RescaleRange(env)
    # resize frame to 84×84 image
    env = ResizeObservation(env, (84, 84))
    # stack 4 frames (equivalent to what phi does in paper) 
    env = FrameStack(env, num_stack=4)
    
    return env

# check for mps, cuda or cpu
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# ------- Test script ----------

ENV="BreakoutNoFrameskip-v4"
# build env
env = generate_env(ENV)
print(f"Current Atari environment: {ENV}")

model = DQN_model(env.observation_space.shape, env.action_space.n).to(device)
model.load_state_dict(torch.load("breakout_wtarget_dqn.pt"))

curr_state = env.reset()
curr_state = np.asarray(curr_state)

steps = 10000
for i in range(steps):
    action = choose_action(model, curr_state, device)
    obs, reward, done, _ = env.step(action)
    obs = np.asarray(obs)
    env.render()
    curr_state = obs
    if done: 
        curr_state = env.reset()
        curr_state = np.asarray(curr_state)
# close env
env.reset()
env.close()