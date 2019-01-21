import os
import subprocess
import regex as re


def extract_song_snippet(generated_text):
    pattern = '\r\n\r\n(.*?)\r\n\r\n'
    search_results = re.findall(pattern, generated_text, overlapped=True, flags=re.DOTALL)
    songs = [song for song in search_results]
    return songs

def save_song_to_abc(song, filename="tmp"):
    with open("{}.abc".format(filename), "w") as f:
        f.write(song)

def abc2wav(abc_file):
    path = os.path.realpath(__file__)
    path_to_tool = '.'
    subprocess.run([os.path.join(path_to_tool, "abc2wav"), abc_file])

def extract_audio_snippet(generated_text):
    song = extract_audio_snippet(generated_text)[0]
    save_song_to_abc(song)
