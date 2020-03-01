# Librosa client to extract and study features from 
# custom data set (uploaded in personal google drive)
# Youtube on wishlist

# Running with conda make sure proper environment is activated which contains the modules
# librosa and gdown
# With FMP installed on local machine conda activate FMP
# on a Linux machine, use pip to install the packages before running

import numpy as np
import librosa as rosa
import librosa.display as disp
import gdown as gd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from pathlib import Path
import logging


logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Change this location based on host machine, this
# location will be used to download songs and a sub-directory
# will contain graphs/extracted features
local_directory = Path('C:\\personal\\musik\\')
graphs_subdir = local_directory / 'graphs'

# Links from eigene data-set
piya_say_naina = 'https://drive.google.com/uc?id=1oB3wFoZgHGE5gnP88KV505AcFrnEHfz_'
khawaja = 'https://drive.google.com/uc?id=165I1Uf8LqaAiBm750NOxLbeDy8isJf3R'
zikar_parivash_ka = 'https://drive.google.com/uc?id=1L5VRFhhCF3oaQWyfynip_z_jVUQpqp07'
sikh_chaj = 'https://drive.google.com/uc?id=1PHVdbiAq_QDSAkbvV54WiNtFmxljROzd'
shikwa = 'https://drive.google.com/uc?id=16GVaIjXHwHWNHhPhGo4y6O5TwTxVJefx'
is_karam_ka = 'https://drive.google.com/uc?id=121UBXqQS0iwU5ZsznKyK4zWBEMcTf4FV'
mast_nazron_say = 'https://drive.google.com/uc?id=14uKXY_DTpVY5vx88HhBAz33azJofYXMd'
mohay_apnay_rang = 'https://drive.google.com/uc?id=1rrx6KSKnLzqy7095Wkac4doLdJ8r06H-'
meray_sohnaya = 'https://drive.google.com/uc?id=1Wjs2tO5X5GOJgI3nyGJ3RNjAl0ENEwXB'
mera_imaan_pak = 'https://drive.google.com/uc?id=1ebmOKPeAZ7Ouj8wJ9M3kflbTJEmXOd_e'
maye_ni_maye = 'https://drive.google.com/uc?id=18PITR8ZsTSaF4XuXALL5h9j8Tj4REJaW'
kise_da_yaar = 'https://drive.google.com/uc?id=1SWQ4Av4ck5Fy8XqX9lbv-f0MWGHD8iIL'

# Youtube wish-list
yt_rumi_qawali='https://www.youtube.com/watch?v=FjggJH45RRE'

url_map = {piya_say_naina : 'piya_say_naina.mp3',
		   khawaja : 'khawaja.mp3',
		   zikar_parivash_ka : 'zikar_parivash_ka.mp3',
		   shikwa : 'shikwa.mp3',
		   is_karam_ka : 'is_karam_ka.mp3',
		   mast_nazron_say : 'mast_nazron_say.mp3',
		   mohay_apnay_rang : 'mohay_apnay_rang.mp3',
		   sikh_chaj : 'sikh_chaj.mp3',
		   meray_sohnaya : 'meray_sohnaya.mp3',
		   mera_imaan_pak : 'mera_imaan_pak.mp3',
		   maye_ni_maye : 'maye_ni_maye.mp3',
		   kise_da_yaar : 'kise_da_yaar.mp3'
		   }


# Constants (should probably be parameterized)
ActualSampleRate = 22050
ObserveDurationInSec = 120
# With above sample rate, this length translates to 46ms
HopLength = 1024
FreqBins = 96

# Main loop
for song in url_map:

	local_song_path = local_directory / url_map[song]
	logger.info('Attempting to load %s', local_song_path)
	# if same file already has been downloaded skip it
	if os.path.exists(local_song_path):
		logger.info('File %s already exists, skipping Download', url_map[song])
	else: 
		logger.info('Downloading %s', url_map[song])
		gd.download(song, str(local_song_path), quiet=False, proxy=None)

	# Load snippet from song
	y, sr = rosa.load(str(local_song_path), sr=ActualSampleRate, mono=True, offset=0.0, duration=ObserveDurationInSec)

	if (sr != ActualSampleRate): 
		logger.warning("Determined sample rate=%d different from assumed!", sr)

	# Separate harmonics and percussives into two waveforms
	y_harmonic, y_percussive = rosa.effects.hpss(y)

	logger.info('Audio file loaded, harmonics, percussion separated...')

	
	fig, (ax1, ax2) = plt.subplots(2,1)
	C_harmonic = np.abs(rosa.cqt(y_harmonic, sr=sr, hop_length=HopLength, n_bins=FreqBins))
	rosa.display.specshow(rosa.amplitude_to_db(C_harmonic, ref=np.max),
                        sr=sr, x_axis='time', y_axis='cqt_hz', hop_length=HopLength, ax=ax1)
	
	ax1.set_title('Harmonic CQT')

	C_percussive = np.abs(rosa.cqt(y_percussive, sr=sr, hop_length=HopLength, n_bins=FreqBins))
	rosa.display.specshow(rosa.amplitude_to_db(C_percussive, ref=np.max),
                        sr=sr, x_axis='time', y_axis='cqt_note', hop_length=HopLength, ax=ax2)

	ax2.set_title('Percussive CQT')
	# Colorbar does not seem to work at all without a current image	
	# fig.colorbar(cm.get_cmap('RdBu_r'), ax=ax2, format='%+2.0f dB')
	fig.tight_layout()

	fig.savefig(str(graphs_subdir / local_song_path.stem) + '.png')
	logger.info('********************************************')

logger.info('CQT features extracted!')