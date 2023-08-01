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
import argparse

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
    # get unclipped reward for testing
    # env = ClipReward(env, -1, 1) 
    env = AtariCropping(env)
    # gray scale frame
    env = GrayScaleObservation(env, keep_dim=False)
    env = RescaleRange(env)
    # resize frame to 84×84 image
    env = ResizeObservation(env, (84, 84))
    # stack 4 frames (equivalent to what phi does in paper) 
    env = FrameStack(env, num_stack=4)
    
    return env

# ------- PARAMETERS: modify this section for your use case ----------

# model mapping to its atari env
model_to_env = {
    "pong":"PongNoFrameskip-v4",
    "breakout_000025":"BreakoutNoFrameskip-v4",
    "best":"BreakoutNoFrameskip-v4",
}
MODEL = "best_forcefire" #name of model to be tested
RECORD = False #whether to record testing games
RECORD_PREFIX = "demo" #prefix for recorded videos
RECORD_FREQ = 2 #record video per RECORD_FREQ episode

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
ENV = model_to_env[MODEL] #env to run model


# ------- TEST SCRIPT ----------
# build env
env = generate_env(ENV)
# record every testing episode
if RECORD:
    if not os.path.exists("records"):
        # Create a new directory if it does not exist
        os.makedirs("records")
    env = RecordVideo(env, video_folder=f"records/{MODEL}", name_prefix=RECORD_PREFIX, episode_trigger= lambda x: x%RECORD_FREQ==0)

print(f"-----------------")
print(f"Device: {DEVICE}")
print(f"Current Atari environment: {ENV}")
if RECORD:
    print(f"Recording saved to: records/{MODEL}")
print(f"-----------------")

model = DQN_model(env.observation_space.shape, env.action_space.n).to(DEVICE)
model.load_state_dict(torch.load(f"trained_models/{MODEL}.pt", map_location=DEVICE))

curr_state = env.reset()
curr_state = np.asarray(curr_state)

steps = 10000
rewards = [0]
curr_lives = 0
for i in range(steps):
    action = choose_action(model, curr_state, DEVICE)
    obs, reward, done, info = env.step(action)
    obs = np.asarray(obs)
    env.render()
    curr_state = obs
    rewards[-1]+=reward
    
    curr_lives = info['lives']   
        
    if done: 
        curr_state = env.reset()
        curr_state = np.asarray(curr_state)
        rewards.append(0)
        curr_lives=0
        
# close env
env.reset()
env.close()
print(f"-----------------")
print("Unclipped reward each episode: ", rewards[:-1] if len(rewards)>1 else rewards)