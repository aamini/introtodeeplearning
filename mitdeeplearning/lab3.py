import io
import base64
from IPython.display import HTML


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
