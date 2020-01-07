import io
import base64
from IPython.display import HTML


def play_video(filename):
    encoded = base64.b64encode(io.open('./agent.mp4', 'r+b').read())
    embedded = HTML(data='''
        <video controls>
            <source src="data:video/mp4;base64,{0}" type="video/mp4" />
        </video>'''.format(encoded.decode('ascii')))

    return embedded
