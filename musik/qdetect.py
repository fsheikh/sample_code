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
from enum import Enum
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
#stdoutHandler = logging.StreamHandler(sys.stdout)
#logger.addHandler(stdoutHandler)


# Tri-state decision enum
class Decision(Enum):
    YES = 0
    NO = 1
    MAYBE = 2

# Objects of this class contain raw-data and features extracted from the songs
# Supports member processing functions implementing heuristics for Qawali categorization
class QDetect:
    # Raw data stored on when saved on disk in numpy format
    RAW_DATA_FILE = 'songs-data.npy'
    EX_FEATURES_FILE = 'tt-features.npy'
    FEATURE_DIR = 'features'
    TABLA_SUFFIX = 'tabla'
    TAALI_SUFFIX = 'taali'
    SUPPORTED_SAMPLE_RATES = [44100, 22050]
    FRAME_LENGTH = 1024
    CQT_BINS = 84
    def __init__(self, songDir, reload=False, sampleRate=44100):
        # Raw data extracted from songs represented in a map with
        # song name (without extension) as key
        self.m_rawData = {}
        # Suitable features extracted from raw data represented in a map
        # with song-name and feature-name tuple as key
        self.m_featuresEx = {}

        self.m_songList = []
        self.m_sr = 0

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

        # Walk through songs directory to collect names of all supported format songs
        for root, dirname, songs in os.walk(songDir):
            self.m_songList += [os.path.splitext(s)[0] for s in songs if s.endswith('.mp3')]
            self.m_songList += [os.path.splitext(s)[0] for s in songs if s.endswith('.au')]

        logger.info("Following songs will used for evaluation {}...".format(self.m_songList))

        # When reload option is used, existing data is simply overwritten
        if reload:
            logger.info("Data reload requested, all supported songs in directory %s will be processed", songDir)
            for song in self.m_songList:
                songPath = Path(songDir) / Path(song+'.mp3')
                songPathA = Path(songDir) / Path(song+'.au')
                if songPath.exists():
                    self.m_rawData[song], sr = rosa.load(path=songPath, sr=None, mono=True, offset=0.0, dtype='float32')
                    if sr not in QDetect.SUPPORTED_SAMPLE_RATES:
                        logger.error("Unexpected sample rate {} for {}, cannot continue".format(sr, song))
                        raise RuntimeError
                    else:
                        self.m_sr = sr
                elif songPathA.exists():
                    self.m_rawData[song], sr = rosa.load(path=songPathA, sr=None, offset=0.0, dtype='float32')
                    if sr not in QDetect.SUPPORTED_SAMPLE_RATES:
                        logger.error("Unexpected sample rate {} for {}, cannot continue".format(sr, song))
                        raise RuntimeError
                    else:
                        self.m_sr = sr
                else:
                    logger.warning("Song paths={%s, %s} not found", songPath, songPathA)
            # delete existing file
            audioMapPath.unlink() if audioMapPath.exists() else None
            self.m_rawData['rate'] = self.m_sr
            np.save(audioMapPath, self.m_rawData)
        else:
            # load from existing audio data map
            self.m_rawData = np.load(audioMapPath, allow_pickle=True).item()
            self.m_sr = self.m_rawData['rate']
            logger.info("Dataset loaded from {} with sample-rate {}".format(audioMapPath, self.m_sr))

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
            # Ignore entry for sample-rate
            if song == 'rate':
                continue
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
            taaliActivations = np.take(act, taaliBasis, axis=0)
            tablaComponents = np.take(comp, tablaBasis, axis=1)
            tablaActivations = np.take(act, tablaBasis, axis=0)

            # Reconstruct tabla/taali spectrum based on separated elements
            taaliSpec = taaliComponents.dot(taaliActivations)
            tablaSpec = tablaComponents.dot(tablaActivations)

            # we are interested in tabla-CQT and Taali-MFCC, no direct conversion
            # for former, we convert to audio and then compute CQT
            # for later librosa is able to compute MFCC from spectrogram directly
            tablaAudio = rosa.istft(tablaSpec)
            tablaCQT = np.abs(rosa.cqt(tablaAudio, sr=self.m_sr,
                                hop_length=QDetect.FRAME_LENGTH, n_bins=QDetect.CQT_BINS))

            taaliMel = rosa.feature.melspectrogram(S=taaliSpec, sr=self.m_sr,
                                                    hop_length=QDetect.FRAME_LENGTH, n_fft=QDetect.FRAME_LENGTH)

            # mel-to-normalized mfcc conversion
            taaliMfcc = rosa.feature.mfcc(S=rosa.power_to_db(taaliMel), sr=self.m_sr, n_mfcc=13)
            mfccEnergy = np.linalg.norm(taaliMfcc, axis=1, keepdims=True)
            normalizedTaaliMfcc = taaliMfcc / (mfccEnergy + 1e-8) # avoid divide by zero

            # features calculated, save in feature-map to be used for classification later
            self.m_featuresEx[song + '.' + QDetect.TABLA_SUFFIX] = tablaCQT
            self.m_featuresEx[song + '.' + QDetect.TAALI_SUFFIX] = normalizedTaaliMfcc

            if generatePlots:
                fig = plt.figure(10,10)
                plt.subplot(2,1,1)
                rosa.display.specshow(normalizedTaaliMfcc, x_axis='time', y_axis='time')
                plt.title('MFCC of separated taali source')
                plt.colorbar()
                plt.tight_layout()
                plt.subplot(2,1,2)
                rosa.display.specshow(rosa.amplitude_to_db(tablaCQT, ref=np.max), sr=self.m_sr,
                                        x_axis='time', y_axis='cqt_hz', hop_length=QDetect.FRAME_LENGTH)
                plt.title('CQT of tabla separated source')
                pltFile = featureDir / song + '-tt.png'
                pltFile.unlink() if pltFile.exists() else None
                fig.savefig(str(pltFile))
                plt.close(fig)

            np.save(featureExFile, self.m_featuresEx)

    @staticmethod
    def in_interval(r_low, r_high, x_val):
        """
        [r_low, r_high) interval in which the value is be checked.
        x_val value checked for falling inside the given interval
        """
        retVal = True if r_low <= x_val < r_high else False
        return retVal

    @staticmethod
    def isTabla(cqtPower, mb1=53, mb2=60, mbSpread=20):
        """
        cqtPower: CQT power in each octave, orgnized by midi-notes
        https://musicinformationretrieval.com/midi_conversion_table.html
        [mb1, mb2]: Midi-notes internal for may-be decision
        mbSpread: variance of CQT power allowed for may-be decision
        """
        # C midi-note in first and last octave
        C1 = 24
        C8 = 108
        # First half of third octave
        C3 = 48
        F3 = 53

        if cqtPower.size != (C8-C1):
            logger.error("Unexpected tabla CQT-power size: {}".format(cqtPower.size))
            raise ValueError

        # TODO: peak-power spread/variance allowed for positive decision
        # Should this be parameterized
        tightDev = 4

        # Model for detect peak CQT power for tabla instrument
        cqtRange = np.arange(C1, C8)
        tablaD = Decision.NO

        # Tabla used in most Qawali renditions has a characteristic timber with its
        # making different to classical table (drone versus melody), its repeatitve,
        # follows only a handful of taals (sequence of tabla Bols). All of this means
        # that tabla is highly detectable simply by looking at spectrum power, with the
        # expectation that tabla's main frequency component lies within third Octave. This
        # is captured by modeling CQT power of tabla source in upper-half of third octave
        # with energy spread within tight bounds
        # Of-course with performances containing vocals, other instruments, taali and
        # gaps where tabla is not playing at all, it is possible for tabla frequency spectrum
        # to diverge from this hard pattern, we attempt to catch it by defining a bigger range
        # of frequecies to find peak power and also allow for a variable energy spread.
        tablaModel = GaussianModel()
        cqtParams = tablaModel.guess(cqtPower, x=cqtRange)
        mFit = tablaModel.fit(cqtPower, cqtParams, x=cqtRange)
        modelCenter = mFit.params['center'].value
        modelDeviation = mFit.params['sigma'].value

        # Ideal case, tabla pitch power centered in expected midi-range and is well-centered
        if QDetect.in_interval(C3, F3, modelCenter) and QDetect.in_interval(0, tightDev, modelDeviation):
            tablaD = Decision.YES
        elif QDetect.in_interval(mb1, mb2, modelCenter) and QDetect.in_interval(0, mbSpread, modelDeviation):
            tablaD = Decision.MAYBE
        else:
            logger.info("Tabla not detected, model calculated pitch-power mean={} and std-dev {}".format(modelCenter, modelDeviation))

        return tablaD

    @staticmethod
    def isTaali(mfccTaali, allNeg=True):
        MfccCount = 13
        M5 = 5
        M6 = 6
        M7 = 7
        taaliD = Decision.NO
        if mfccTaali.size != MfccCount:
            logger.error("Unexpected number {} of Mfcc given to taali detector".format(mfccTaali.size))
            raise ValueError()


        # We look for an oscillating pattern between 5-7 Mfcc's either +, -, +, or -, +, -
        # Two exceptions are allowed:
        #   Mfcc7 is negative but still higher than Mfcc6
        #   all three interesting Mfcc are lower than zero
        # in both these cases we return indeterminate decision from taali feature, so that
        # qawali genre decision is deferred to tabla feature
        if mfccTaali[M5] > 0 and mfccTaali[M7] > 0 and mfccTaali[M6] < 0:
            taaliD = Decision.YES
        elif mfccTaali[M5] < 0 and mfccTaali[M7] < 0 and mfccTaali[M6] > 0:
            taaliD = Decision.YES
        elif mfccTaali[M5] > 0 and mfccTaali[M6] < 0 and mfccTaali[M7] > mfccTaali[M6]:
            taaliD = Decision.MAYBE
        elif allNeg and mfccTaali[M5] < 0 and mfccTaali[M6] < 0 and mfccTaali[M7] < 0:
            taaliD = Decision.MAYBE
        else:
            logger.info("Taali not detected with m5={} m6={} m7={}".format(mfccTaali[M5], mfccTaali[M6], mfccTaali[M7]))

        return taaliD

    # Classifies a song as a Qawali based on extracted tabla-taali features
    def classify(self):
        # Feature map should be available for classification to work
        featureMapPath = Path(self.m_songDir) / QDetect.FEATURE_DIR / QDetect.EX_FEATURES_FILE
        if not featureMapPath.exists():
            logger.error("No features map exist, were features extracted with decomopse?")
            raise RuntimeError

        ttMap = np.load(featureMapPath, allow_pickle=True).item()


        counters = {}
        counters['noQ'] = 0
        counters['both'] = 0
        counters[QDetect.TABLA_SUFFIX] = 0
        counters[QDetect.TAALI_SUFFIX] = 0
        for song in self.m_songList:
            logger.info("\r\nClassification starting for song: {}".format(song))

            # Get the featues, pass them to internal heuristic based function to
            # detect tabla and taali music sources, combine individual decisions to
            # classify given song as Qawali genre or otherwise
            cqtTablaPower = np.linalg.norm(ttMap[song + '.' + QDetect.TABLA_SUFFIX], axis=1)
            mfccTaali = np.median(ttMap[song + '.' + QDetect.TAALI_SUFFIX], axis=1)
            tablaD = QDetect.isTabla(cqtTablaPower)
            taaliD = QDetect.isTaali(mfccTaali)
            if tablaD == Decision.YES and taaliD == Decision.YES:
                counters['both'] = counters['both'] + 1
                logger.info("{} categorized as Qawali after detecting tabla and taali".format(song))
            elif tablaD == Decision.YES and taaliD == Decision.MAYBE:
                counters[QDetect.TABLA_SUFFIX] = counters[QDetect.TABLA_SUFFIX] + 1
                logger.info("{} categorized as Qawali after detecting tabla and suspecting taali".format(song))
            elif taaliD == Decision.YES and tablaD == Decision.MAYBE:
                counters[QDetect.TAALI_SUFFIX] = counters[QDetect.TAALI_SUFFIX] + 1
                logger.info("{} categorized as Qawali after detecting taali and suspecting tabla".format(song))
            else:
                counters['noQ'] = counters['noQ'] + 1
                logger.info("{} is not a Qawali tabla {} taali {}".format(song, tablaD, taaliD))


        total = len(self.m_songList)
        if (total - counters['noQ']) != counters['both'] + counters[QDetect.TABLA_SUFFIX] + counters[QDetect.TAALI_SUFFIX]:
            logger.info("Discrepancy in classification results?")
            raise ValueError

        logger.info("\r\n--------------------Classification Results----------------------------\r\n")
        logger.info("Total={} non-Qawalis={} TablaTaali={} Tabla={} Taali={}".format(total,
                    counters['noQ'], counters['both'], counters[QDetect.TABLA_SUFFIX], counters[QDetect.TAALI_SUFFIX]))

if __name__ == '__main__':
    qParser = argparse.ArgumentParser(description="Qawali genre detection program")
    qParser.add_argument("songs_dir", type=str, help="folder/directory containing songs to be evaluated")
    qParser.add_argument("--reload", action='store_true', help="reload data from songs (required at least once per songs directory)")
    qParser.add_argument("--extract", action='store_true', help="extract suitable audio features from raw data (required at least once)")

    qArgs = qParser.parse_args()

    qGenre = QDetect(qArgs.songs_dir, qArgs.reload)

    if qArgs.extract:
        qGenre.decompose()

    qGenre.classify()


    # TODO: may be dump results in a format which can be easily processed/plotted.

