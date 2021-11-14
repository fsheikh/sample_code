# Python3 program detecting qawali genre in songs.

# Copyright (C) 2020-21 Faheem Sheikh (fahim.sheikh@gmail.com)
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>

import argparse
import os
import sys
from pathlib import Path
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import librosa as rosa
import librosa.display as disp
from lmfit import Model
from lmfit.models import GaussianModel


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)
stdoutHandler = logging.StreamHandler(sys.stdout)
logger.addHandler(stdoutHandler)

# Objects of this class contain raw-data and features extracted from the songs
# Supports member processing functions implementing heuristics for Qawali categorization
class QDetect:
    # Raw data stored on when saved on disk in numpy format
    RAW_DATA_FILE = 'songs-data.npy'
    EX_FEATURES_FILE = 'tt-features.npy'
    FEATURE_DIR = 'features'
    SUPPORTED_SAMPLE_RATE = 44100
    FRAME_LENGTH = 1024
    CQT_BINS = 84
    def __init__(self, songDir, reload=False):
        # Raw data extracted from songs represented in a map with
        # song name (without extension) as key
        self.m_rawData = {}
        # Suitable features extracted from raw data represented in a map
        # with song-name and feature-name tuple as key
        self.m_featuresEx = {}

        self.m_songList = []
        if songDir is None:
            logger.error("Song directory not given, no data to process!")
            raise RuntimeError
        elif not os.path.isdir(songDir):
            logger.error("Directory {} not found, no data to process".format(songDir))
            raise RuntimeError

        self.m_songDir = songDir
        audioMapPath = Path(self.m_songDir) / QDetect.RAW_DATA_FILE
        if not reload and not os.path.isfile(audioMapPath):
            logger.info("Audio map {} does not exist, need to reload data?\n".format(audioMapPath))
            raise RuntimeError

        # When reload option is used, existing data is simply overwritten
        if reload:
            logger.info("Data reload requested, all mp3/au songs in directory %s will be processed", songDir)
            for root, dirname, songs in os.walk(songDir):
                self.m_songList += [os.path.splitext(s)[0] for s in songs if s.endswith('.mp3')]
                self.m_songList += [os.path.splitext(s)[0] for s in songs if s.endswith('.au')]

            logger.info("Following songs will used for evaluation {}...".format(self.m_songList))

            for song in self.m_songList:
                songPath = Path(songDir) / Path(song+'.mp3')
                songPathA = Path(songDir) / Path(song+'.au')
                if songPath.exists():
                    self.m_rawData[song], sr = rosa.load(path=songPath, sr=None, mono=True, offset=0.0, dtype='float32')
                    logger.warning("Unexpected sample rate {} for {}".format(sr, song)) if sr != QDetect.SUPPORTED_SAMPLE_RATE else None
                elif songPathA.exists():
                    self.m_rawData[song], sr = rosa.load(path=songPathA, sr=None, offset=0.0, dtype='float32')
                    logger.warning("Unexpected sample rate {} for {}".format(sr, song)) if sr != QDetect.SUPPORTED_SAMPLE_RATE else None
                else:
                    logger.warning("Song paths={%s, %s} not found", songPath, songPathA)
        else:
            # load from existing audio data map
            self.m_rawData = np.load(audioMapPath, allow_pickle=True).item()
        if reload:
            # delete existing file
            audioMapPath.unlink() if audioMapPath.exists() else None
            np.save(audioMapPath, self.m_rawData)
        
    # Extracts features from separated rhythm sources of a qawali song
    # extracted features are tabla CQT and taali-mfcc
    def decompose(self, generatePlots=False):
        # make a sub-directory to store extracted features inside song-data directory
        featureDir = Path(self.m_songDir) / Path(QDetect.FEATURE_DIR)
        if not featureDir.exists():
            os.mkdir(str(featureDir.absolute()))
        # remove existing map
        featureExFile = featureDir / QDetect.EX_FEATURES_FILE
        featureExFile.unlink() if featureExFile.exists() else None

        # Feature extraction from each song in the data set:
        # In the first stage we separate tabla (drum instrument)
        # and taali (periodic hand-clapping) source from the given file with 
        # NVM factorization
        for song in self.m_rawData:
            logger.info("Separating {} into tabla/taali sources".format(song))
            magSpectrum = np.abs(rosa.stft(self.m_rawData[song], n_fft=QDetect.FRAME_LENGTH))
            comp, act = rosa.decompose.decompose(magSpectrum, n_components=4, sort=True)
            # Taali/clapping is short-time event contributing to both low and high frequency components
            taaliBasis = np.array([1,3])
            # Tabla/Rythym instrument is continaully playing Bass components with most energy
            # in low frequencies.
            tablaBasis = np.array([0,1])

            # Collect individual components for tabla/taali sources
            taaliComponents = np.take(comp, taaliBasis, axis=1)
            taaliActivations = np.take(comp, taaliBasis, axis=0)
            tablaComponents = np.take(comp, tablaBasis, axis=1)
            tablaActivations = np.take(comp, tablaBasis, axis=0)

            # Reconstruct tabla/taali spectrum based on separated elements
            taaliSpec = taaliComponents.dot(taaliActivations)
            tablaSpec = tablaComponents.dot(tablaActivations)

            # we are interested in tabla-CQT and Taali-MFCC, no direct conversion
            # for former, we convert to audio and then compute CQT
            # for later librosa is able to compute MFCC from spectrogram directly
            tablaAudio = rosa.istft(tablaSpec)
            tablaCQT = np.abs(rosa.cqt(tablaAudio, sr=QDetect.SUPPORTED_SAMPLE_RATE,
                                hop_length=QDetect.FRAME_LENGTH, n_bins=QDetect.CQT_BINS))

            taaliMel = rosa.feature.melspectrogram(S=taaliSpec, sr=QDetect.SUPPORTED_SAMPLE_RATE,
                                                    hop_length=QDetect.FRAME_LENGTH, n_fft=QDetect.FRAME_LENGTH)

            # mel-to-normalized mfcc conversion
            taaliMfcc = rosa.feature.mfcc(S=rosa.power_to_db(taaliMel), sr=QDetect.SUPPORTED_SAMPLE_RATE, n_mfcc=13)
            mfccEnergy = np.linalg.norm(taaliMfcc, axis=1, keepdims=True)
            normalizedTaaliMfcc = taaliMfcc / (mfccEnergy + 1e-8) # avoid divide by zero

            # features calculated, save in feature-map to be used for classification later
            self.m_featuresEx[(song, 'tabla')] = tablaCQT
            self.m_featuresEx[(song, 'taali')] = normalizedTaaliMfcc

            if generatePlots:
                fig = plt.figure(10,10)
                plt.subplot(2,1,1)
                rosa.display.specshow(normalizedTaaliMfcc, x_axis='time', y_axis='time')
                plt.title('MFCC of separated taali source')
                plt.colorbar()
                plt.tight_layout()
                plt.subplot(2,1,2)
                rosa.display.specshow(rosa.amplitude_to_db(tablaCQT, ref=np.max), sr=QDetect.SUPPORTED_SAMPLE_RATE,
                                        x_axis='time', y_axis='cqt_hz', hop_length=QDetect.FRAME_LENGTH)
                plt.title('CQT of tabla separated source')
                pltFile = featureDir / song + '-tt.png'
                pltFile.unlink() if pltFile.exists() else None
                fig.savefig(str(pltFile))
                plt.close(fig)

            np.save(featureExFile, self.m_featuresEx)

    # Classifies a song as a Qawali based on extracted tabla-taali features
    def classify(self):
        logger.info("Placeholder for classification of song as Qawali/Non-Qawali based on extracted features")

if __name__ == '__main__':
    qParser = argparse.ArgumentParser(description="Qawali genre detection program")
    qParser.add_argument("sdir", type=str, help="folder/directory containing songs to be evaluated")
    qParser.add_argument("--reload", action='store_true', help="reload data from songs (required at least once per songs directory)")
    qParser.add_argument("--extract", action='store_true', help="extract suitable audio features from raw data (required at least once)")

    qArgs = qParser.parse_args()

    qGenre = QDetect(qArgs.sdir, qArgs.reload)

    if qArgs.extract:
        qGenre.decompose()

    qGenre.classify()


    # TODO: may be dump results in a format which can be easily processed/plotted.

