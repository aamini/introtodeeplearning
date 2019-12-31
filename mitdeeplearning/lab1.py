import os
import regex as re
import subprocess
import urllib

from IPython.display import Audio


DATA_URL = 'https://raw.githubusercontent.com/aamini/introtodeeplearning_labs/2019/lab1/data/irish.abc'

def load_training_data():
    stream = urllib.request.urlopen(DATA_URL)
    text = stream.read().decode("utf-8")
    songs = extract_song_snippet(text)
    return songs

def extract_song_snippet(text):
    pattern = '\n\n(.*?)\n\n'
    search_results = re.findall(pattern, text, overlapped=True, flags=re.DOTALL)
    songs = [song for song in search_results]
    print("Found {} songs in text".format(len(songs)))
    return songs

def save_song_to_abc(song, filename="tmp"):
    save_name = "{}.abc".format(filename)
    with open(save_name, "w") as f:
        f.write(song)
    return filename

def abc2wav(abc_file):
    # path_to_tool = './introtodeeplearning_labs/lab1/abc2wav'
    path_to_tool = os.path.join(os.path.dirname(__file__), 'bin', 'abc2wav')

    cmd = "{} {}".format(path_to_tool, abc_file)
    return os.system(cmd)

def play_wav(wav_file):
    return Audio(wav_file)

def play_song(song):
    basename = save_song_to_abc(song)
    ret = abc2wav(basename+'.abc')
    if ret == 0: #did not suceed
        return play_wav(basename+'.wav')
    return None

def play_generated_song(generated_text):
    songs = extract_song_snippet(generated_text)
    if len(songs) == 0:
        print("No valid songs found in generated text. Try training the \
            model longer or increasing the amount of generated music to \
            ensure complete songs are generated!")

    for song in songs:
        play_song(song)
    print("None of the songs were valid, try training longer to improve \
        syntax.")
