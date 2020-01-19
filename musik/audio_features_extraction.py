# A first program as librosa client to extract and study features from 
# individual data set (uploaded in personal google drive)

# Running with conda make sure proper environment is activated which contains the modules
# libroas and gdown
# With FMP installed on local machine conda activate FMP
# on a new linux machine, use pip to install the packages before running

import numpy as np
import librosa as rosa
import librosa.display as disp
import gdown as gd
import matplotlib.pyplot as plt
import os

# TODO: Conver into lists to process more than one songs
song_url = 'https://drive.google.com/uc?id=1oB3wFoZgHGE5gnP88KV505AcFrnEHfz_'
song_location = 'C:\\personal\\musik\q1.mp3'


# if same file already has been downloaded skip it

if os.path.exists(song_location):
	print('File already exists, skipping Download')
else: 
	gd.download(song_url, song_location, quiet=False, proxy=None)


# Taken from Librosa tutorial
y, sr = rosa.load(song_location)

# Set the hop length; at 22050 Hz, 1024 samples ~= 46
hop_length = 1024

# Separate harmonics and percussives into two waveforms
y_harmonic, y_percussive = rosa.effects.hpss(y)

print('Audio file loaded, harmonics, percussion separated...')

# Beat track on the percussive signal
tempo, beat_frames = rosa.beat.beat_track(y=y_percussive,
                                             sr=sr)

print('Computing MFCCs')

# Compute MFCC features from the raw signal
mfcc = rosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)

plt.subplot(2,2,1)
disp.specshow(mfcc, x_axis='time', y_axis='mel', sr=sr, hop_length=hop_length)
# And the first-order differences (delta features)
mfcc_delta = rosa.feature.delta(mfcc)

plt.subplot(2,2,2)
disp.specshow(mfcc_delta, x_axis='time', y_axis='mel', sr=sr, hop_length=hop_length)

# Stack and synchronize between beat events
# This time, we'll use the mean value (default) instead of median
beat_mfcc_delta = rosa.util.sync(np.vstack([mfcc, mfcc_delta]),beat_frames)

#disp.specshow(beat_mfcc_delta, x_axis='time', sr=sr, hop_length=hop_length)


print('Computing chromagram')
# Compute chroma features from the harmonic signal
chromagram = rosa.feature.chroma_cqt(y=y_harmonic,
                                        sr=sr)

plt.subplot(2,2,3)
disp.specshow(chromagram, cmap='gray_r', y_axis='chroma', sr=sr, hop_length=hop_length)

# Aggregate chroma features between beat events
# We'll use the median value of each feature between beat frames
beat_chroma = rosa.util.sync(chromagram,
                                beat_frames,
                                aggregate=np.median)

# Finally, stack all beat-synchronous features together
#beat_features = np.vstack([beat_chroma, beat_mfcc_delta])

plt.subplot(2,2,4)
disp.specshow(beat_mfcc_delta, x_axis='time', sr=sr, hop_length=hop_length)

plt.show()
print('Done with feature extraction!')