#!/bin/bash

abcfile=$1
suffix=${abcfile%.abc}
abc2midi $abcfile -o "$suffix.mid"
timidity "$suffix.mid" -Ow "$suffix.wav"
rm "$suffix.abc" "$suffix.mid"
