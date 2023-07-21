import gym
import numpy as np
from collections import deque
import numpy as np

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

class MaxAndSkipEnv(gym.Wrapper):
    """Return only every 4th frame"""
    def __init__(self, env=None, skip=4):
        super(MaxAndSkipEnv, self).__init__(env)
        # Initialise a double ended queue that can store a maximum of two states
        self._obs_buffer = deque(maxlen=2)
        # _skip = 4
        self._skip       = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        info = None
        for _ in range(self._skip):
            # Take a step 
            obs, reward, done, info = self.env.step(action)
            # Append the new state to the double ended queue buffer 
            self._obs_buffer.append(obs)
            # Update the total reward by summing the (reward obtained from the step taken) + (the current 
            # total reward)
            total_reward += reward
            # If the game ends, break the for loop 
            if done:
                break

        # max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        max_frame = self._obs_buffer[1]

        # max_frame = self._obs_buffer[1]
        return max_frame, total_reward, done, info
    