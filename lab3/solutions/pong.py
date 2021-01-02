import time
import functools
import argparse
import multiprocessing
from multiprocessing import Pool
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import copy


import tensorflow as tf
import gym
import cv2
import numpy as np

import mitdeeplearning as mdl

parser = argparse.ArgumentParser(description='Train a neural network given input data')
parser.add_argument('--batch_size', type=int)
parser.add_argument('--learning_rate', type=float)
parser.add_argument('--draw', dest='draw', action='store_true')
args = parser.parse_args()

print(args)


physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


env = gym.make("Pong-v0", frameskip=5)
env.seed(1)  # for reproducibility

n_actions = env.action_space.n

### Define the agent's action function ###


# Function that takes observations as input, executes a forward pass through model,
#   and outputs a sampled action.
# Arguments:
#   model: the network that defines our agent
#   observation: observation which is fed as input to the model
# Returns:
#   action: choice of agent action
def choose_action(model, observation, single=True):
    # create a batch dimension if only a single example was provided
    observations = np.expand_dims(observation, axis=0) if single else observation

    # add batch dimension to the observation
    # observation = np.expand_dims(observation, axis=0)
    '''TODO: feed the observations through the model to predict the log probabilities of each possible action.'''
    logits = model.predict(observations)  # TODO
    # logits = model.predict('''TODO''')

    # pass the log probabilities through a softmax to compute true probabilities
    # prob_weights = tf.nn.softmax(logits)
    '''TODO: randomly sample from the prob_weights to pick an action.
  Hint: carefully consider the dimensionality of the input probabilities (vector) and the output action (scalar)'''

    action = tf.random.categorical(logits, num_samples=1)
    action = action.numpy().flatten()

    return action[0] if single else action


### Reward function ###

# Helper function that normalizes an np.array x
def normalize(x):
    x -= np.mean(x)
    x /= np.std(x)
    return x.astype(np.float32)



### Agent Memory ###

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


# Combine a list of Memory objects into a single Memory (e.g., for batching)
def aggregate_memories(memories):
    batch_memory = Memory()
    for memory in memories:
         for step in zip(memory.observations, memory.actions, memory.rewards):
             batch_memory.add_to_memory(*step)
    return batch_memory


memory = Memory()

### Loss function ###


# Arguments:
#   logits: network's predictions for actions to take
#   actions: the actions the agent took in an episode
#   rewards: the rewards the agent received in an episode
# Returns:
#   loss
def compute_loss(logits, actions, rewards):
    '''TODO: complete the function call to compute the negative log probabilities'''
    neg_logprob = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=actions)  # TODO
    # neg_logprob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits='''TODO''', labels='''TODO''')
    '''TODO: scale the negative log probability by the rewards'''
    loss = tf.reduce_mean(neg_logprob * rewards)  # TODO
    # loss = tf.reduce_mean('''TODO''')
    return loss


### Training step (forward and backpropagation) ###


def train_step(model, optimizer, observations, actions, discounted_rewards):
    with tf.GradientTape() as tape:
        # Forward propagate through the agent network
        logits = model(observations)
        '''TODO: call the compute_loss function to compute the loss'''
        loss = compute_loss(logits, actions, discounted_rewards)  # TODO
        # loss = compute_loss('''TODO''', '''TODO''', '''TODO''')
    '''TODO: run backpropagation to minimize the loss using the tape.gradient method'''
    grads = tape.gradient(loss, model.trainable_variables)  # TODO
    # grads = tape.gradient('''TODO''', model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


### Define the Pong agent ###

# Functionally define layers for convenience
# All convolutional layers will have ReLu activation
Conv2D = functools.partial(
    tf.keras.layers.Conv2D, padding='same', activation='relu')
Flatten = tf.keras.layers.Flatten
Dense = tf.keras.layers.Dense


# Defines a CNN for the Pong agent
def create_pong_model():
    model = tf.keras.models.Sequential([
        # Convolutional layers
        # First, 16 7x7 filters and 4x4 stride
        Conv2D(filters=32, kernel_size=5, strides=2),

        # TODO: define convolutional layers with 32 5x5 filters and 2x2 stride
        Conv2D(filters=48, kernel_size=5, strides=2),  # TODO
        # Conv2D('''TODO'''),

        # TODO: define convolutional layers with 48 3x3 filters and 2x2 stride
        Conv2D(filters=64, kernel_size=3, strides=2),  # TODO
        Conv2D(filters=64, kernel_size=3, strides=2),  # TODO
        # Conv2D('''TODO'''),
        Flatten(),

        # Fully connected layer and output
        Dense(units=128, activation='relu'),
        # TODO: define the output dimension of the last Dense layer.
        # Pay attention to the space the agent needs to act in
        Dense(units=n_actions, activation=None)  # TODO
        # Dense('''TODO''')
    ])
    return model


pong_model = create_pong_model()


### Pong reward function ###


# Compute normalized, discounted rewards for Pong (i.e., return)
# Arguments:
#   rewards: reward at timesteps in episode
#   gamma: discounting factor. Note increase to 0.99 -- rate of depreciation will be slower.
# Returns:
#   normalized discounted reward
def discount_rewards(rewards, gamma=0.99):
    discounted_rewards = np.zeros_like(rewards)
    R = 0
    for t in reversed(range(0, len(rewards))):
        # NEW: Reset the sum if the reward is not 0 (the game has ended!)
        if rewards[t] != 0:
            R = 0
        # update the total discounted reward as before
        R = R * gamma + rewards[t]
        discounted_rewards[t] = R

    return normalize(discounted_rewards)


# env.reset()
# for i in range(1000):
#   observation, _,_,_ = env.step(0)
#   observation_pp = mdl.lab3.preprocess_pong(observation)
#   cv2.imshow('hi', cv2.resize(observation_pp, (256, 256)))
#   cv2.waitKey(5)



### Training Pong ###

# Hyperparameters
learning_rate = args.learning_rate
MAX_ITERS = 100  # increase the maximum number of episodes, since Pong is more complex!

# Model and optimizer
pong_model = create_pong_model()

optimizer = tf.keras.optimizers.Adam(learning_rate)

# plotting
smoothed_reward = mdl.util.LossHistory(smoothing_factor=0.9)
smoothed_reward.append(-21) # start the reward at the minimum (0-21) for baseline comparison
# plotter = mdl.util.PeriodicPlotter(
#     sec=5, xlabel='Iterations', ylabel='Rewards')
batch_size = args.batch_size



def parallelized_collect_rollout(batch_size, envs, model, choose_action):

    assert len(envs) == batch_size, "Number of parallel environments must be equal to the batch size."

    memories = [Memory() for _ in range(batch_size)]
    next_observations = [single_env.reset() for single_env in envs]
    previous_frames = [obs for obs in next_observations]
    done = [False] * batch_size
    rewards = [0] * batch_size

    tic = time.time()
    while True:

        current_frames = [obs for obs in next_observations]
        diff_frames = [mdl.lab3.pong_change(prev, curr) for (prev, curr) in zip(previous_frames, current_frames)]

        diff_frames_not_done = [diff_frames[b] for b in range(batch_size) if not done[b]]
        actions_not_done = choose_action(model, np.array(diff_frames_not_done), single=False)

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

    return memories



def collect_rollout(batch_size, env, model, choose_action):

    memories = []

    for b in range(batch_size):
        memory = Memory()
        next_observation = env.reset()
        previous_frame = next_observation
        done = False

        while not done:
            current_frame = next_observation
            diff_frame = mdl.lab3.pong_change(previous_frame, current_frame)

            action = choose_action(model, diff_frame)

            next_observation, reward, done, info = env.step(action)

            previous_frame = current_frame
            memory.add_to_memory(diff_frame, action, reward)

        memories.append(memory)

    return memories


# mdl.lab3.save_video_of_memory(memory[0])
# collect_rollout(batch_size, env, model, choose_action)




envs = [copy.deepcopy(env) for _ in range(batch_size)]

for i_episode in range(MAX_ITERS):


    tic = time.time()
    # memories = collect_rollout(batch_size, env, pong_model, choose_action)
    memories = parallelized_collect_rollout(batch_size, envs, pong_model, choose_action)
    batch_memory = aggregate_memories(memories)
    print(time.time()-tic)


    # def parallel_episode(i):
    #     return run_episode(env=copy.deepcopy(env), model=pong_model)
    #
    # tic = time.time()
    # memories = [parallel_episode(batch) for batch in range(batch_size)]
    # print(time.time()-tic)

    # models = [tf.keras.models.load_model('model.h5') for b in range(batch_size)]
    # tic = time.time()
    # with Pool(processes=batch_size) as pool:
    #     memories = pool.map(parallel_episode, models)#range(batch_size))
    # print(time.time()-tic)


    # batch_memory = Memory()
    # for memory in memories:
    #      for step in zip(memory.observations, memory.actions, memory.rewards):
    #          batch_memory.add_to_memory(*step)





    ### Train with this batch!!!

    # while True:
        # is the episode over? did you crash or do so well that you're done?
        # if done:
    total_reward = sum(batch_memory.rewards) / batch_size
    # print(total_reward, round(smoothed_reward.get()[-1], 2))
    # print(total_reward)
    # print("episode time:", time.time()-tic)

    # determine total reward and keep a record of this
    smoothed_reward.append(total_reward)

    iters = i_episode * batch_size
    last_smoothed_reward = smoothed_reward.get()[-1]
    print(f"{iters} \t {round(last_smoothed_reward, 3)}")

    # begin training
    train_step(
        pong_model,
        optimizer,
        observations=np.stack(batch_memory.observations, 0),
        actions=np.array(batch_memory.actions),
        discounted_rewards=discount_rewards(batch_memory.rewards))

    if i_episode % 500 == 0:
        mdl.lab3.save_video_of_model(pong_model, "Pong-v0", suffix=str(i_episode))
