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
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack, RecordVideo
from utilities.custom_wrappers import ClipReward, AtariCropping, RescaleRange, MaxAndSkipEnv
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

def generate_env(env_name):
    env = gym.make(env_name, render_mode="rgb_array")
    env = MaxAndSkipEnv(env, skip=4)
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

model_to_env = {
    "pong_wskipframes.pt":"PongNoFrameskip-v4",
    "breakout_wtarget_dqn.pt":"BreakoutNoFrameskip-v4",
    "breakout_pen_loselives.pt":"BreakoutNoFrameskip-v4",
}


# ------- Test script ----------
MODEL="pong_wskipframes.pt"
ENV=model_to_env[MODEL]
# build env
env = generate_env(ENV)

# record every testing episode
# env = RecordVideo(env, video_folder="PongRecords", name_prefix="pong_demo", step_trigger= lambda x: True)

print(f"Current Atari environment: {ENV}")

model = DQN_model(env.observation_space.shape, env.action_space.n).to(device)
model.load_state_dict(torch.load(f"trained_models/{MODEL}"))

curr_state = env.reset()
curr_state = np.asarray(curr_state)

steps = 10000
rewards = [0]
for i in range(steps):
    action = choose_action(model, curr_state, device)
    obs, reward, done, _ = env.step(action)
    obs = np.asarray(obs)
    env.render()
    curr_state = obs
    rewards[-1]+=reward
    if done: 
        curr_state = env.reset()
        curr_state = np.asarray(curr_state)
        rewards.append(0)
# close env
env.reset()
env.close()
print(rewards[:-1])