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
import logging


logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# Links from eigene data-set
piya_say_naina = 'https://drive.google.com/uc?id=1oB3wFoZgHGE5gnP88KV505AcFrnEHfz_'
khawaja = 'https://drive.google.com/uc?id=165I1Uf8LqaAiBm750NOxLbeDy8isJf3R'
yt_rumi_qawali='https://www.youtube.com/watch?v=FjggJH45RRE'
zikar_parivash_ka = 'https://drive.google.com/uc?id=1L5VRFhhCF3oaQWyfynip_z_jVUQpqp07'

local_directory = 'C:\\personal\\musik'
url_local_map = {piya_say_naina : local_directory+'\\q1.mp3',
				 khawaja : local_directory+'\\q2.mp3',
				 zikar_parivash_ka : local_directory+'\\g1.mp3'}

				 
logger.info(url_local_map)

# if same file already has been downloaded skip it

if os.path.exists(url_local_map[khawaja]):
	logger.info('File %s already exists, skipping Download', url_local_map[khawaja])
else: 
	gd.download(khawaja, url_local_map[khawaja], quiet=False, proxy=None)

aSR = 22050
observeDuration = 120
# Taken from Librosa tutorial
y, sr = rosa.load(url_local_map[khawaja], sr=aSR, mono=True, offset=0.0, duration=observeDuration)

if (sr != aSR): 
	logger.warning("Determined sample rate=%d different from assumed!", sr)

# Set the hop length; at 22050 Hz, 1024 samples ~= 46ms
hop_length = 1024

# Separate harmonics and percussives into two waveforms
y_harmonic, y_percussive = rosa.effects.hpss(y)


logger.info('Audio file loaded, harmonics, percussion separated...')

ax = plt.subplot(2,1,1)
C_harmonic = np.abs(rosa.cqt(y_harmonic, sr=sr, hop_length=hop_length, n_bins=60))
rosa.display.specshow(rosa.amplitude_to_db(C_harmonic, ref=np.max),
                        sr=sr, x_axis='time', y_axis='cqt_note', hop_length=hop_length)

plt.colorbar(format='%+2.0f dB')
plt.title('Harmonic CQT')
plt.tight_layout()

C_percussive = np.abs(rosa.cqt(y_percussive, sr=sr, hop_length=hop_length, n_bins=60))
ax = plt.subplot(2,1,2)
rosa.display.specshow(rosa.amplitude_to_db(C_percussive, ref=np.max),
                        sr=sr, x_axis='time', y_axis='cqt_note', hop_length=hop_length)

plt.colorbar(format='%+2.0f dB')
plt.title('Percussive CQT')
plt.tight_layout()

plt.show()
"""
# Beat track on the percussive signal
tempo, beat_frames = rosa.beat.beat_track(y=y_percussive,
                                             sr=sr)

logger.info('Computing MFCCs')

# Compute MFCC features from the raw signal
mfcc = rosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=5)

ax = plt.subplot(2,2,1)
disp.specshow(mfcc, x_axis='time', y_axis='mel', sr=sr, hop_length=hop_length)

ax.title.set_text('mfcc')
# And the first-order differences (delta features)
mfcc_delta = rosa.feature.delta(mfcc)
ax = plt.subplot(2,2,2)
disp.specshow(mfcc_delta, x_axis='time', y_axis='mel', sr=sr, hop_length=hop_length)
ax.title.set_text('delta-mfcc')
# Stack and synchronize between beat events
# This time, we'll use the mean value (default) instead of median
beat_mfcc_delta = rosa.util.sync(np.vstack([mfcc, mfcc_delta]),beat_frames)
ax=plt.subplot(2,2,3)
disp.specshow(beat_mfcc_delta, x_axis='time', sr=sr, hop_length=hop_length)
ax.title.set_text('beat-mfcc')


logger.info('Computing chromagram')
# Compute chroma features from the harmonic signal
chromagram = rosa.feature.chroma_cqt(y=y_harmonic,
                                        sr=sr)

ax = plt.subplot(2,2,4)
disp.specshow(chromagram, cmap='gray_r', y_axis='chroma', sr=sr, hop_length=hop_length)
ax.title.set_text('chroma')

plt.show()
logger.info('Done with feature extraction!')
"""