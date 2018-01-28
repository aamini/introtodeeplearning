import midi_manipulation
import numpy as np
import glob


def create_dataset(min_length):
    songs = glob.glob('data/*.mid*')
    
    encoded_songs = []
    discarded = 0
    for song in songs:
        encoded_song = midi_manipulation.midiToNoteStateMatrix(song)
        encoded_song = make_one_hot_notes(encoded_song)
        if len(encoded_song) >= min_length:
            encoded_songs.append(encoded_song)
        else:
            discarded += 1
    print("{} songs processed".format(len(songs)))
    print("{} songs discarded".format(discarded))
    return encoded_songs

def make_one_hot_notes(song):
    """
    Makes the song one_hot by choosing the highest note 
    from each chord (presumably the melody)
    """
    new_song = np.zeros(song.shape)
    for i in range(len(song)):
        nonzeros = np.nonzero(song[i])
        if len(nonzeros[0]) > 0:
            new_song[i, nonzeros[0][-1]] = 1
    return new_song


def get_batch(encoded_songs, batch_size, timesteps, input_size, output_size):
    
    rand_song_indices = np.random.randint(len(encoded_songs), size=batch_size)
    batch_x = np.zeros((batch_size, timesteps, input_size))
    batch_y = np.zeros((batch_size, output_size))
    for i in range(batch_size):
        song_ind = rand_song_indices[i]
        start_ind = np.random.randint(encoded_songs[song_ind].shape[0]-timesteps-1)
        batch_x[i] = encoded_songs[song_ind][start_ind:start_ind+timesteps]
        batch_y[i] = encoded_songs[song_ind][start_ind+timesteps]
    return batch_x, batch_y