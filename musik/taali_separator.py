# Module attempts to separate Tabli/Taali background percussion from Qawali
# songs using NUSSL utilities.

# Copyright (C) 2020  Faheem Sheikh (fahim.sheikh@gmail.com)
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

import librosa as rosa
import librosa.display as disp
from lmfit import Model
from lmfit.models import GaussianModel, LorentzianModel, RectangleModel
from lmfit.models import LognormalModel, VoigtModel, StudentsTModel
import numpy as np
import nussl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import sys
from pathlib import Path
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)
stdoutHandler = logging.StreamHandler(sys.stdout)
logger.addHandler(stdoutHandler)

class TaaliSeparator:
    def __init__(self, songList=[], songDir=None):
        # Constructor makes a map of NUSSL objects with song
        # names as key, if songList is given, only songs in list
        # are used, otherwise all songs from given directory used.
        # In case neither argument is supplied, an empty object is constructed
        # assuming songs have already been loaded by a previous call
        # Empty object is needed to call member functions on a saved dictionary (on disk)
        # Contains audio data as read from songs indexed by song names in given directory
        self.m_sourceMap = {}
        # Dictionary contained tabla-taali source separated audio-data as numpy arrray
        # indexed by song name
        self.m_separatedMap = {}
        # reconstructed audio data from above dictionary will be stored in a file named:
        self.m_ttFile = 'tt-map.npy'
        self.m_audioFile = 'song-map.npy'
        self.m_taaliRef = np.array([])

        self.m_songList = []
        if songDir is None and not songList:
            logger.error("Song list and directory not given, empty object constructed!")
            return
        elif not os.path.isdir(songDir):
            logger.error("Directory %s not found", songDir)
            raise RuntimeError

        self.m_songDir = songDir
        self.m_songList = songList
        audioMapPath = os.path.join(self.m_songDir, self.m_audioFile)
        if os.path.isfile(audioMapPath):
            logger.info("Audio map %s already exist, loading data...\n\n", audioMapPath)
            self.m_sourceMap = np.load(audioMapPath, allow_pickle=True).item()
            print(self.m_sourceMap)
            return

        if not songList:
            logger.info("Song list empty, all mp3/au songs in directory %s will be added", songDir)
            for root, dirname, songs in os.walk(songDir):
                self.m_songList += [os.path.splitext(s)[0] for s in songs if s.endswith('.mp3')]
                self.m_songList += [os.path.splitext(s)[0] for s in songs if s.endswith('.au')]
        logger.info("Following songs will be evaluated...")
        print(self.m_songList)

        for song in self.m_songList:
            songPath = os.path.join(songDir, song+'.mp3')
            songPathA = os.path.join(songDir, song+'.au')
            if os.path.isfile(songPath):
                self.m_sourceMap[song] = nussl.AudioSignal(path_to_input_file=songPath,
                                                                sample_rate=44100, offset=0.0, duration=60.0)
                self.m_sourceMap[song].to_mono(overwrite=True)
                logger.info("Song=%s SampleRate=%d", song, self.m_sourceMap[song].sample_rate)
            elif os.path.isfile(songPathA):
                self.m_sourceMap[song] = nussl.AudioSignal(path_to_input_file=songPathA,
                                                                sample_rate=44100, offset=0.0, duration=30.0)
                logger.info("Song=%s SampleRate=%d", song, self.m_sourceMap[song].sample_rate)

            else:
                logger.error("Song paths={%s, %s} not found", songPath, songPathA)

        print(self.m_sourceMap)
        np.save(audioMapPath, self.m_sourceMap)

    @staticmethod
    def e2_c3_c5_b5(freqIndex, sampleRate, fftPoints):
        E2Freq = 82.41
        C3Freq = 138.6
        C5Freq = 523.25
        B5Freq = 987.77
        adjustedFreq = freqIndex * sampleRate / fftPoints
        if (E2Freq < adjustedFreq < C3Freq) or (C5Freq < adjustedFreq < B5Freq):
            logger.info("Freq found with [E2,C3] | [C5, B5] %12.8f index=%d", adjustedFreq, freqIndex)
            return True
        return False

    @staticmethod
    def tablaTaaliRange(freqIndex, sampleRate, fftPoints):
        # Midi notes to approx center frequencies
        E2 = 82.41
        C3 = 138.6
        freqHz = freqIndex * sampleRate / fftPoints
        if E2 < freqHz < C3:
            logger.info("Freq %12.8f in tabla-taali range [E2, C3]", freqHz)
            return True
        return False

    def rosa_decompose(self, generatePlots=False):
        localDir = os.path.join(self.m_songDir, 'rosa-decompose')
        if not os.path.isdir(localDir):
            os.mkdir(localDir)
        for song in self.m_sourceMap:
            sampleRate = self.m_sourceMap[song].sample_rate
            if sampleRate != 44100:
                logger.warn("Processing song=%s with sample rate=%d...\n", song, sampleRate)
            magSpectrum = np.abs(rosa.stft(self.m_sourceMap[song].audio_data[0,:], n_fft=1024))
            comps, acts = rosa.decompose.decompose(magSpectrum, n_components=4, sort=True)
            taaliBasis = np.array([1,3])
            tablaBasis = np.array([0,1])
            logger.info("Separating tabla taali sources for %s", song)
            # Create placeholders for selected components and activations
            taaliComponents = np.take(comps, taaliBasis, axis=1)
            taaliActivations = np.take(acts, taaliBasis, axis=0)
            tablaComponents = np.take(comps, tablaBasis, axis=1)
            tablaActivations = np.take(acts, tablaBasis, axis=0)

            # Construct tabla-taali spectrum
            taaliSpectrum = taaliComponents.dot(taaliActivations)
            taaliMel = rosa.feature.melspectrogram(S=taaliSpectrum, sr=sampleRate, hop_length=1024, n_fft=1024)
            tablaSpectrum = tablaComponents.dot(tablaActivations)
            tablaAudio = rosa.istft(tablaSpectrum)
            if generatePlots:
                taaliMfcc = rosa.feature.mfcc(S=rosa.power_to_db(taaliMel), sr=sampleRate, n_mfcc=13)
                mfccEnergy = np.linalg.norm(taaliMfcc, axis=1, keepdims=True)
                normalizedMfcc = taaliMfcc / (mfccEnergy + 1e-8)
                medianEnergy = np.median(normalizedMfcc, axis=1)
                logger.info("Mfcc median energy=%12.6g for song=%s", medianEnergy.mean(), song)
                fig = plt.figure(figsize=(10,10))
                plt.subplot(2,2,1)
                plt.bar(np.arange(0, 13), medianEnergy)
                plt.xlabel('MFCC number')
                plt.subplot(2,2,2)
                disp.specshow(normalizedMfcc, x_axis='time', y_axis='time')
                plt.title('MFCC of Taali extracted audio')
                plt.colorbar()
                fig.tight_layout()
                plt.subplot(2,2,3)
                tablaCqt = np.abs(rosa.cqt(tablaAudio, sr=sampleRate, hop_length=1024, n_bins=84))
                disp.specshow(rosa.amplitude_to_db(tablaCqt, ref=np.max),
                    sr=sampleRate, x_axis = 'time', y_axis='cqt_hz', hop_length=1024)
                plt.title('Tabla Separated CQT')
                pitchEnergy = np.linalg.norm(tablaCqt, axis=1)
                plt.subplot(2,2,4)
                plt.title('Table separated audio pitch energy profile')
                plt.bar(np.arange(0, pitchEnergy.size), pitchEnergy)
                plt.xlabel('Pitch Number')
                fig.savefig(os.path.join(localDir, song + '-tabla-taali-features.png'))
                plt.close(fig)
                self.m_separatedMap[(song, 'tabla')] = tablaCqt
                self.m_separatedMap[(song, 'taali')] = normalizedMfcc

        # Save separated source audio data on disk for later processing
        np.save(os.path.join(localDir, self.m_ttFile), self.m_separatedMap)

    @staticmethod
    # Limit array to indices less than value
    def limit_index(array, limit):
        if limit < min(array):
            return 0
        return max(np.where(array <= limit)[0])

    @staticmethod
    # Checks if x is in the interval [r_low, r_high)
    def in_interval(r_low, r_high, x):
        if r_low <= x < r_high:
            return True
        else:
            return False

    @staticmethod
    def detect_tabla(featureData):
        # https://www.inspiredacoustics.com/en/MIDI_note_numbers_and_center_frequencies
        C1 =  24
        C8 = 108
        A2Sharp = 46
        C3 = 48
        F3 = 53
        B3 = 59
        logger.info("Starting tabla detection")
        if featureData.size != (C8-C1):
            logger.error("CQT energy data missing octaves? bins=%d", featureData.size)

        spreadS = 4
        spreadB = 18
        cqtXRange = np.arange(C1, C8)
        cqtModel = GaussianModel()
        cqtParams = cqtModel.guess(featureData, x=cqtXRange)
        mFit = cqtModel.fit(featureData, cqtParams, x=cqtXRange)
        result = "No"
        # Tabla source in a song is characterized by a peak in pitch power
        # at the start of third ocatve. Allow first half of octave as range
        # and some tolerance for energy spread. Maybe decision is made if the peak appears
        # shifted in the C3 octave OR if the variance higher but the peak location is still
        # in the same half octave band
        center = mFit.params['center'].value
        var = mFit.params['sigma'].value
        if TaaliSeparator.in_interval(C3, F3, center) and TaaliSeparator.in_interval(0, spreadS, var):
            result = "Yes"
        elif TaaliSeparator.in_interval(A2Sharp, B3, center) and TaaliSeparator.in_interval(0, spreadB, var):
            result = "Maybe"
        else:
            logger.warning("tabla not detected with pitch power centered=%6.8f with variance=%6.8f", center, var)

        print('Parameter    Value       Stderr')
        for name, param in mFit.params.items():
            try:
                print('{:7s} {:11.5f} {:11.5f}'.format(name, param.value, param.stderr))
            except TypeError:
                if param.stderr is None:
                    # Warning will spam
                    # logger.warning("No stadard error reported in param:%s", name)
                    print('{:7s} {:11.5f}'.format(name, param.value))
                else:
                    logger.warning("Model=%s fitting failed?", featureName)
        print('-------------------------------')

        return result

    @staticmethod
    def detect_taali(featureData):
        MfccCount = 13
        M5 = 5
        M6 = 6
        M7 = 7
        result = "No"
        logger.info("Starting taali detection")
        if featureData.size != MfccCount:
            logger.error("Missing Mfcc's called with size=%d", featureData.size)
        # We look for an alternating cycle +,-,+ between 5-7 Mfcc
        # Exception being reverse -, +, -
        # and the case where 7th coefficient did not raise enough
        if featureData[M5] > 0 and featureData[M6] < 0 and featureData[M7] > 0:
            result = "Yes"
        elif featureData[M5] < 0 and featureData[M6] > 0 and featureData[M7] < 0:
            result = "Yes"
        elif featureData[M5] > 0 and featureData[M6] < 0 and featureData[M7] > featureData[M6]:
            result = "Maybe"
        elif featureData[M5] < 0 and featureData[M6] < 0 and featureData[M7] < 0:
            result = "Maybe"
        else:
            logger.info("Taali not detected with m5=%6.8f m6=%6.8f m7=%6.8f", featureData[M5], featureData[M6], featureData[M7])
        return result

    # Directory containing feature map is given as input
    def tt_classify(self, fmDir=None):
        if fmDir is None:
            raise RuntimeError
        else:
            # Load feature map containing CQT and MFCC features useful for detecting
            # tabla and taali sources
            ttMap = np.load(os.path.join(fmDir, self.m_ttFile), allow_pickle=True).item()
            print(ttMap)
        noQawali = 0
        songName = ""
        # Features are stored against a tuple containing song-name, feature name
        for songFeature in ttMap:
            # Each song has two associated tuples, we skip the if its the same song
            # since in that case it has already been processed.
            if songName == songFeature[0]:
                continue
            songName = songFeature[0]
            logger.info("\r\nClassification Loop for %s...\r\n", songName)
            # Two main features, cqt for detecting Tabla and Mfcc for detecting taali
            cqtPower = np.linalg.norm(ttMap[(songName, 'tabla')], axis=1)
            mfccMedian = np.median(ttMap[(songName, 'taali')], axis=1)
            # Get classification decisions for tabla and taali source separately
            tablaD = TaaliSeparator.detect_tabla(cqtPower)
            taaliD = TaaliSeparator.detect_taali(mfccMedian)
            if (tablaD == "Yes" and taaliD == "Yes"):
                logger.info("Qawali detected after detecting both tabla and taali")
            elif (tablaD == "Maybe" and taaliD == "Yes"):
                logger.info("Qawali detected due to taali and suspicison of tabla")
            elif (tablaD == "Yes" and taaliD == "Maybe"):
                logger.info("Qawali detected due to tabla and suspicison of taali")
            else:
                logger.info("No Cigar tabla=%s taali=%s", tablaD, taaliD)
                noQawali= noQawali + 1


        logger.info("\r\n----------Results------------------\r\n")
        logger.info("Songs processed=%d Non-Qawalis=%d", len(ttMap)/2, noQawali)

    def __del__(self):
        logger.info("%s Destructor called", __name__)



if __name__ == '__main__':
    # Initialize TaaliSeparator Object
    ts = TaaliSeparator([], '/home/fsheikh/musik/qawali')
    #ts = TaaliSeparator([], '/home/fsheikh/musik/gtzan/genres/pop')
    #ts = TaaliSeparator([], '/home/fsheikh/musik/gtzan/genres/blues')
    #ts = TaaliSeparator([], '/home/fsheikh/musik/gtzan/genres/classical')
    #ts = TaaliSeparator([], '/home/fsheikh/musik/gtzan/genres/country')
    #ts = TaaliSeparator([], '/home/fsheikh/musik/gtzan/genres/disco')
    #ts = TaaliSeparator([], '/home/fsheikh/musik/gtzan/genres/hiphop')
    #ts = TaaliSeparator([], '/home/fsheikh/musik/gtzan/genres/jazz')
    #ts = TaaliSeparator([], '/home/fsheikh/musik/gtzan/genres/metal')
    #ts = TaaliSeparator([], '/home/fsheikh/musik/gtzan/genres/reggae')
    #ts = TaaliSeparator([], '/home/fsheikh/musik/gtzan/genres/rock')
    #ts = TaaliSeparator()
    # Call various source separation algorithms
    ts.rosa_decompose(True)
    ts.tt_classify('/home/fsheikh/musik/qawali/rosa-decompose')
    #ts.tt_classify('/home/fsheikh/musik/gtzan/genres/jazz/rosa-decompose')
    #ts.tt_classify('/home/fsheikh/musik/gtzan/genres/pop/rosa-decompose')
    #ts.tt_classify('/home/fsheikh/musik/gtzan/genres/blues/rosa-decompose')
    #ts.tt_classify('/home/fsheikh/musik/gtzan/genres/classical/rosa-decompose')
    #ts.tt_classify('/home/fsheikh/musik/gtzan/genres/country/rosa-decompose')
    #ts.tt_classify('/home/fsheikh/musik/gtzan/genres/hiphop/rosa-decompose')
    #ts.tt_classify('/home/fsheikh/musik/gtzan/genres/metal/rosa-decompose')
    #ts.tt_classify('/home/fsheikh/musik/gtzan/genres/reggae/rosa-decompose')
    #ts.tt_classify('/home/fsheikh/musik/gtzan/genres/rock/rosa-decompose')
