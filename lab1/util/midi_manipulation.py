import midi
import numpy as np


lowerBound = 24
upperBound = 102
span = upperBound-lowerBound


def midiToNoteStateMatrix(midifile, squash=True, span=span):
	pattern = midi.read_midifile(midifile)
	
	timeleft = [track[0].tick for track in pattern]
	
	posns = [0 for track in pattern]
	
	statematrix = []
	time = 0
	
	state = [[0,0] for x in range(span)]
	statematrix.append(state)
	condition = True
	while condition:
		if time % (pattern.resolution / 4) == (pattern.resolution / 8):
			# Crossed a note boundary. Create a new state, defaulting to holding notes
			oldstate = state
			state = [[oldstate[x][0],0] for x in range(span)]
			statematrix.append(state)
		for i in range(len(timeleft)): #For each track
			if not condition:
				break
			while timeleft[i] == 0:
				track = pattern[i]
				pos = posns[i]
				
				evt = track[pos]
				if isinstance(evt, midi.NoteEvent):
					if (evt.pitch < lowerBound) or (evt.pitch >= upperBound):
						pass
						# print "Note {} at time {} out of bounds (ignoring)".format(evt.pitch, time)
					else:
						if isinstance(evt, midi.NoteOffEvent) or evt.velocity == 0:
							state[evt.pitch-lowerBound] = [0, 0]
						else:
							state[evt.pitch-lowerBound] = [1, 1]
				elif isinstance(evt, midi.TimeSignatureEvent):
					if evt.numerator not in (2, 4):
						# We don't want to worry about non-4 time signatures. Bail early!
						# print "Found time signature event {}. Bailing!".format(evt)
						out =  statematrix
						condition = False
						break
				try:
					timeleft[i] = track[pos + 1].tick
					posns[i] += 1
				except IndexError:
					timeleft[i] = None
			
			if timeleft[i] is not None:
				timeleft[i] -= 1
		
		if all(t is None for t in timeleft):
			break
		
		time += 1
	S = np.array(statematrix)
	statematrix = np.hstack((S[:, :, 0], S[:, :, 1]))
	statematrix = np.asarray(statematrix)
	one_hot_matrix = np.array([1 if S[i][j].any() else 0 for i in range(len(S)) for j in range(len(S[i]))]).reshape((S.shape[0], S.shape[1]))
	return one_hot_matrix

def noteStateMatrixToMidi(statematrix, name="example", span=span):
	statematrix = np.array(statematrix)
	pattern = midi.Pattern()
	track = midi.Track()
	pattern.append(track)
	
	span = upperBound-lowerBound
	tickscale = 55
	
	lastcmdtime = 0
	prevstate = [0 for x in range(span)]
	for time, state in enumerate(statematrix + [prevstate[:]]):  
		offNotes = []
		onNotes = []
		for i in range(span):
			n = state[i]
			p = prevstate[i]
			if p == 1:
				if n == 0:
					offNotes.append(i)
			elif n == 1:
				onNotes.append(i)
		for note in offNotes:
			track.append(midi.NoteOffEvent(tick=(time-lastcmdtime)*tickscale, pitch=note+lowerBound))
			lastcmdtime = time
		for note in onNotes:
			track.append(midi.NoteOnEvent(tick=(time-lastcmdtime)*tickscale, velocity=40, pitch=note+lowerBound))
			lastcmdtime = time
		
		prevstate = state
	
	eot = midi.EndOfTrackEvent(tick=1)
	track.append(eot)
	
	midi.write_midifile("{}.mid".format(name), pattern)

