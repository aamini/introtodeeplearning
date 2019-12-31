import mitdeeplearning as mdl

songs = mdl.lab1.load_training_data()

basename = mdl.lab1.save_song_to_abc(songs[0])
ret = mdl.lab1.abc2wav(basename+'.abc')

import pdb; pdb.set_trace()
