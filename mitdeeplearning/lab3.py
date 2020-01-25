import io
import base64
from IPython.display import HTML
import gym
import numpy as np

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


def save_video_of_model(model, env_name, obs_diff=False, pp_fn=None):
    import skvideo.io
    from pyvirtualdisplay import Display
    display = Display(visible=0, size=(400, 300))
    display.start()

    if pp_fn is None:
        pp_fn = lambda x: x

    env = gym.make(env_name)
    obs = env.reset()
    obs = pp_fn(obs)
    prev_obs = obs

    filename = env_name + ".mp4"
    output_video = skvideo.io.FFmpegWriter(filename)

    counter = 0
    done = False
    while not done:
        frame = env.render(mode='rgb_array')
        output_video.writeFrame(frame)

        if obs_diff:
            input_obs = obs - prev_obs
        else:
            input_obs = obs
        action = model(np.expand_dims(input_obs, 0)).numpy().argmax()

        prev_obs = obs
        obs, reward, done, info = env.step(action)
        obs = pp_fn(obs)
        counter += 1

    output_video.close()
    print("Successfully saved {} frames into {}!".format(counter, filename))
    return filename
