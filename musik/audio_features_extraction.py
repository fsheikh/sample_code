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
#local_directory = Path('C:\\personal\\musik\\')
local_directory = Path('/home/fsheikh/musik')
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
mere_saaqi = 'https://drive.google.com/uc?id=17NTxevTz827ClaR1ZISjzbMn9RZBYxoi'
ruthi_rut = 'https://drive.google.com/uc?id=1A-YO4pTd4u0rK-KrHOxTBjjP4ArfwpdJ'
rashk_e_qamar = 'https://drive.google.com/uc?id=17y9uvNrCG0kwSbv3alkH0XjdlkMf5zNC'

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
    kise_da_yaar : 'kise_da_yaar.mp3',
    mere_saaqi : 'mere_saaqi.mp3',
    ruthi_rut : 'ruthi_rut.mp3',
    rashk_e_qamar : 'rashk_e_qamar.mp3'
}


#url_map = {khawaja : 'khawaja.mp3' }
# Constants (should probably be parameterized)
ActualSampleRate = 22050
ObserveDurationInSec = 120
# With above sample rate, this length translates to 46ms
HopLength = 1024
# Starting with C1(midi:24) we like to go till C8(midi:108)
FreqBins = 84 
# librosa uses C1 as default start point for some of its APIs
C1Midi = 24;

# Collecting mean, std.deviation and norm of harmonic and percussive CQT (3*2 features)
midi_features = np.empty([FreqBins, 7], dtype=float)
# First column contains midi number of notes/freqeuncies
midi_features[:,0] = np.arange(C1Midi,C1Midi+FreqBins, dtype=float)

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
    logger.info("Dimensions: Rows=%d Columns=%d", C_harmonic.shape[0], C_harmonic.shape[1])
    midi_features[:,1] = np.linalg.norm(C_harmonic, axis=1)
    midi_features[:,2] = np.mean(C_harmonic, axis=1)
    midi_features[:,3] = np.std(C_harmonic, axis=1)
    logger.info("Max Harmonic Norm=%12.8f with midi=%d", np.max(midi_features[:,1]), np.argmax(midi_features[:,1]) + C1Midi)
    rosa.display.specshow(rosa.amplitude_to_db(C_harmonic, ref=np.max),
            sr=sr, x_axis='time', y_axis='cqt_hz', hop_length=HopLength, ax=ax1)

    ax1.set_title('Harmonic CQT')

    C_percussive = np.abs(rosa.cqt(y_percussive, sr=sr, hop_length=HopLength, n_bins=FreqBins))
    midi_features[:,4] = np.linalg.norm(C_percussive, axis=1)
    midi_features[:,5] = np.mean(C_percussive, axis=1)
    midi_features[:,6] = np.std(C_percussive, axis=1)
    rosa.display.specshow(rosa.amplitude_to_db(C_percussive, ref=np.max),
            sr=sr, x_axis='time', y_axis='cqt_note', hop_length=HopLength, ax=ax2)
    logger.info("Max Percussive Norm=%12.8f with midi=%d", np.max(midi_features[:,4]), np.argmax(midi_features[:,4]) + C1Midi)
    feature_file = str(graphs_subdir / local_song_path.stem)
    logger.info("Writing features to file %s", feature_file+'.txt')
    np.savetxt(feature_file+'.txt', midi_features, 'midi=%6.4f: normH=%6.4f, meanH=%6.4f, std-devH=%6.4f, normP=%6.4f, meanP=%6.4f, std-devP=%6.4f')

    ax2.set_title('Percussive CQT')
    # Colorbar does not seem to work at all without a current image
    # fig.colorbar(cm.get_cmap('RdBu_r'), ax=ax2, format='%+2.0f dB')
    fig.tight_layout()

    fig.savefig(feature_file + '.png')
    logger.info('***Done extracting features from %s ***', url_map[song])
