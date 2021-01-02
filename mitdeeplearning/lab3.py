import io
import base64
from IPython.display import HTML
import gym
import numpy as np
import cv2

def play_video(filename):
    encoded = base64.b64encode(io.open(filename, 'r+b').read())
    embedded = HTML(data='''
        <video controls>
            <source src="data:video/mp4;base64,{0}" type="video/mp4" />
        </video>'''.format(encoded.decode('ascii')))

    return embedded

def preprocess_pong(image):
    I = image[35:195] # Crop
    I = I[::2, ::2, 0] # Downsample width and height by a factor of 2
    I[I == 144] = 0 # Remove background type 1
    I[I == 109] = 0 # Remove background type 2
    I[I != 0] = 1 # Set remaining elements (paddles, ball, etc.) to 1
    return I.astype(np.float).reshape(80, 80, 1)

def new_preprocess_pong(image):
    I = image[35:195] # Crop
    # I = np.mean(I, axis=-1, keepdim=True)
    I = I[::2, ::2, 0] # Downsample width and height by a factor of 2
    I[I == 144] = 0 # Remove background type 1
    I[I == 109] = 0 # Remove background type 2
    I[I != 0] = 1 # Set remaining elements (paddles, ball, etc.) to 1
    I = cv2.dilate(I, np.ones((3, 3), np.uint8), iterations=1)
    I = I[::2, ::2, np.newaxis]
    return I.astype(np.float)


def pong_change(prev, curr):
    prev = new_preprocess_pong(prev)
    curr = new_preprocess_pong(curr)
    I = prev - curr
    I = (I - I.min()) / (I.max() - I.min() + 1e-10)
    return I




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
        frame = env.render(mode='rgb_array')
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


def save_video_of_memory(memory, filename, size=(512,512)):
    import skvideo.io

    output_video = skvideo.io.FFmpegWriter(filename)

    for observation in memory.observations:
        output_video.writeFrame(cv2.resize(255*observation, size))
        
    output_video.close()
    return filename
