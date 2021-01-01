import gym
import multiprocessing
import numpy
import cv2
import numpy as np
import time
import copy

import mitdeeplearning as mdl

batch_size = 2

env = gym.make("Pong-v0", frameskip=5)
env.seed(1)  # for reproducibility



class Memory:
    def __init__(self):
        self.clear()

    # Resets/restarts the memory buffer
    def clear(self):
        self.observations = []
        self.actions = []
        self.rewards = []

    # Add observations, actions, rewards to memory
    def add_to_memory(self, new_observation, new_action, new_reward):
        self.observations.append(new_observation)
        '''TODO: update the list of actions with new action'''
        self.actions.append(new_action)  # TODO
        # ['''TODO''']
        '''TODO: update the list of rewards with new reward'''
        self.rewards.append(new_reward)  # TODO
        # ['''TODO''']



def model(x):
    return [np.random.choice(4) for i in x]


def parallel_episode(i):
    print(i)
    observation = env.reset()
    done = False

    obs = []
    while not done:
        action = model(observation)
        observation, reward, done, info = env.step(action)
        obs.append(observation)

    return np.array(obs)


# tic = time.time()
# with multiprocessing.Pool(processes=batch_size) as pool:
#     results = pool.map(parallel_episode, range(batch_size))
# print(time.time()-tic)

batch_size = 2


envs = [copy.deepcopy(env) for _ in range(batch_size)]
memories = [Memory() for _ in range(batch_size)]
next_observations = [single_env.reset() for single_env in envs]
previous_frames = [obs for obs in next_observations]
done = [False] * batch_size
actions = [0] * batch_size
rewards = [0] * batch_size

def play(memory):
    for o in memory.observations:
        cv2.imshow('hi', cv2.resize(o, (500,500)))
        cv2.waitKey(20)


while True:

    current_frames = [obs for obs in next_observations]
    diff_frames = [mdl.lab3.pong_change(prev, curr) for (prev, curr) in zip(previous_frames, current_frames)]

    diff_frames_not_done = [diff_frames[b] for b in range(batch_size) if not done[b]]
    actions_not_done = model(diff_frames_not_done)

    actions = [None] * batch_size
    ind_not_done = 0
    for b in range(batch_size):
        if not done[b]:
            actions[b] = actions_not_done[ind_not_done]
            ind_not_done += 1


    for b in range(batch_size):
        if done[b]:
            continue
        next_observations[b], rewards[b], done[b], info = envs[b].step(actions[b])
        previous_frames[b] = current_frames[b]
        memories[b].add_to_memory(diff_frames[b], actions[b], rewards[b])

    if all(done):
        break

import pdb; pdb.set_trace()

# 1. current_frame
# 2. diff = pong_change
# 3. action = model(diff)
# 4. obs = step(action)
# 5. prev = curr


observations = []
for i, single_env in enumerate(envs):
    if done[i]:
        continue
    current_frame, reward, done[i], _ = single_env.step(action[i])
    observations.append(
        mdl.lab3.pong_change(previous_frames[i], current_frame))

    previous_frames[i] = current_frame


    actions = model(observations)
import pdb; pdb.set_trace()



# done = False
# previous_frame = env.reset()
# action = 0

while not done:
    current_frame, reward, done, _ = env.step(action)
    observation = mdl.lab3.pong_change(previous_frame, current_frame)

    action = model(observation)
    cv2.imshow('hi', cv2.resize(observation, (500, 500)))
    cv2.waitKey(5)


    previous_frame = current_frame



import pdb; pdb.set_trace()
pass
