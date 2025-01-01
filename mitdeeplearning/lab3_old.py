import io
import base64
from IPython.display import HTML
import gym
import numpy as np
import cv2


def play_video(filename, width=None):
    encoded = base64.b64encode(io.open(filename, "r+b").read())
    video_width = 'width="' + str(width) + '"' if width is not None else ""
    embedded = HTML(
        data="""
        <video controls {0}>
            <source src="data:video/mp4;base64,{1}" type="video/mp4" />
        </video>""".format(video_width, encoded.decode("ascii"))
    )

    return embedded


def preprocess_pong(image):
    I = image[35:195]  # Crop
    I = I[::2, ::2, 0]  # Downsample width and height by a factor of 2
    I[I == 144] = 0  # Remove background type 1
    I[I == 109] = 0  # Remove background type 2
    I[I != 0] = 1  # Set remaining elements (paddles, ball, etc.) to 1
    I = cv2.dilate(I, np.ones((3, 3), np.uint8), iterations=1)
    I = I[::2, ::2, np.newaxis]
    return I.astype(np.float)


def pong_change(prev, curr):
    prev = preprocess_pong(prev)
    curr = preprocess_pong(curr)
    I = prev - curr
    # I = (I - I.min()) / (I.max() - I.min() + 1e-10)
    return I


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
        self.actions.append(new_action)
        self.rewards.append(new_reward)


def aggregate_memories(memories):
    batch_memory = Memory()

    for memory in memories:
        for step in zip(memory.observations, memory.actions, memory.rewards):
            batch_memory.add_to_memory(*step)

    return batch_memory


def parallelized_collect_rollout(batch_size, envs, model, choose_action):
    assert (
        len(envs) == batch_size
    ), "Number of parallel environments must be equal to the batch size."

    memories = [Memory() for _ in range(batch_size)]
    next_observations = [single_env.reset() for single_env in envs]
    previous_frames = [obs for obs in next_observations]
    done = [False] * batch_size
    rewards = [0] * batch_size

    while True:
        current_frames = [obs for obs in next_observations]
        diff_frames = [
            pong_change(prev, curr)
            for (prev, curr) in zip(previous_frames, current_frames)
        ]

        diff_frames_not_done = [
            diff_frames[b] for b in range(batch_size) if not done[b]
        ]
        actions_not_done = choose_action(
            model, np.array(diff_frames_not_done), single=False
        )

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


def save_video_of_model(model, env_name, suffix=""):
    import skvideo.io
    from pyvirtualdisplay import Display

    display = Display(visible=0, size=(400, 300))
    display.start()

    env = gym.make(env_name)
    obs = env.reset()
    prev_obs = obs

    filename = env_name + suffix + ".mp4"
    output_video = skvideo.io.FFmpegWriter(filename)

    counter = 0
    done = False
    while not done:
        frame = env.render(mode="rgb_array")
        output_video.writeFrame(frame)

        if "CartPole" in env_name:
            input_obs = obs
        elif "Pong" in env_name:
            input_obs = pong_change(prev_obs, obs)
        else:
            raise ValueError(f"Unknown env for saving: {env_name}")

        action = model(np.expand_dims(input_obs, 0)).numpy().argmax()

        prev_obs = obs
        obs, reward, done, info = env.step(action)
        counter += 1

    output_video.close()
    print("Successfully saved {} frames into {}!".format(counter, filename))
    return filename


def save_video_of_memory(memory, filename, size=(512, 512)):
    import skvideo.io

    output_video = skvideo.io.FFmpegWriter(filename)

    for observation in memory.observations:
        output_video.writeFrame(cv2.resize(255 * observation, size))

    output_video.close()
    return filename
