# Script for evaluating self-similarity of MFCC matrices
# for two locally available qawali samples

import numpy as np
import librosa as rosa
import librosa.display as disp
import gdown as gd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import sys
from pathlib import Path
import logging


logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

local_directory = Path('/home/fsheikh/musik')
graphs_subdir = local_directory / 'graphs'
q_piya = local_directory / 'piya_say_naina.mp3'
q_khawja = local_directory / 'khawaja.mp3'

# Constants (should probably be parameterized)
ActualSampleRate = 22050
ObserveDurationInSec = 60
# With above sample rate, this length translates to 46ms
HopLength = 1024

if not os.path.exists(str(q_piya)):
    logger.error('File %s does not exist, cannot continue', str(q_piya))
    sys.exit(1);
if not os.path.exists((q_khawja)):
    logger.error('File %s does not exist, cannot continue', str(q_khawja))
    sys.exit(1)

# Load snippet from song
y_piya, sr_k = rosa.load(str(q_piya), sr=ActualSampleRate, mono=True, offset=0.0, duration=ObserveDurationInSec)
y_khawja, sr_p = rosa.load(str(q_khawja), sr=ActualSampleRate, mono=True, offset=0.0, duration=ObserveDurationInSec)

if (sr_k != sr_p):
    logger.error("Sample rates dont match sr_k=%d sr_p=%d", sr_k, sr_p)
    sys.exit(1)

logger.info('Audio file loaded...')

mfcc_piya = rosa.feature.mfcc(y_piya, sr=sr_p, hop_length=HopLength)
mfcc_khawja = rosa.feature.mfcc(y_khawja, sr=sr_p, hop_length=HopLength)
logger.info("mfcc piya: Rows=%d Columns=%d", mfcc_piya.shape[0], mfcc_piya.shape[1])
logger.info("mfcc khawaja: Rows=%d Columns=%d", mfcc_khawja.shape[0], mfcc_khawja.shape[1])

norm_piya = np.linalg.norm(mfcc_piya, axis=1, keepdims=True)
mean_piya = np.mean(mfcc_piya, axis=1, keepdims=True)
norm_khawja = np.linalg.norm(mfcc_khawja, axis=1, keepdims=True)
mean_khawja = np.mean(mfcc_khawja, axis=1, keepdims=True)

X_piya = (mfcc_piya - mean_piya) / norm_piya
X_khawja = (mfcc_khawja - mean_khawja) / norm_khawja

xsim = rosa.segment.cross_similarity(X_piya, X_khawja)
xsim1 = rosa.segment.cross_similarity(X_piya, X_khawja, k=5)


fig = plt.figure(figsize=(10,6))

plt.subplot(2,2,1)
plt.title('piya_mfcc')
rosa.display.specshow(X_piya, sr=sr_p, x_axis='time', y_axis='time', hop_length=HopLength)
plt.colorbar()
plt.subplot(2,2,2)
plt.title('khawja_mfcc')
rosa.display.specshow(X_khawja, sr=sr_k, x_axis='time', y_axis='time', hop_length=HopLength)
plt.colorbar()
plt.subplot(2,2,3)
plt.title('cross-similarity')
rosa.display.specshow(xsim, sr=sr_k, x_axis='time', y_axis='time', hop_length=HopLength)
plt.subplot(2,2,4)
plt.title('cross-similarity-5nn')
rosa.display.specshow(xsim1, sr=sr_k, x_axis='time', y_axis='time', hop_length=HopLength)

plt.tight_layout()
fig.savefig('smm.png')
logger.info('***Done calculating similarity matrices ***\n')
