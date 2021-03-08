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
        #  indexed by song name
        self.m_separatedMap = {}
        # reconstructed audio data from above dictionary will be stored in a file named:
        self.m_ttFile = 'tt-map.npy'

        self.m_songList = []
        if songDir is None and not songList:
            logger.error("Song list and directory not give, empty object constructed!")
            return
        elif not os.path.isdir(songDir):
            logger.error("Directory %s not found", songDir)
            raise RuntimeError

        self.m_songDir = songDir
        self.m_songList = songList
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

    def nussl_repet(self):

        localDir = os.path.join(self.m_songDir, 'nussl-repet')
        if not os.path.isdir(localDir):
            os.mkdir(localDir)
        for song in self.m_sourceMap:
            #separator = nussl.separation.primitive.Repet(self.m_sourceMap[song], min_period=0.5, max_period=2.0,
            #    high_pass_cutoff=500.0, mask_type='binary')
            separator = nussl.separation.primitive.RepetSim(self.m_sourceMap[song], high_pass_cutoff=400.0,
                mask_type='binary')
            estimates = separator()
            # we want to take CQT of foreground/background estimates, plot them, save the plots
            # and write estimates to a file
            fig, (ax1, ax2) = plt.subplots(2,1)
            sampleRate = self.m_sourceMap[song].sample_rate
            fg_cqt = np.abs(rosa.cqt(estimates[0].audio_data[0,:], sr=sampleRate, hop_length=1024, n_bins=84))
            bg_cqt = np.abs(rosa.cqt(estimates[1].audio_data[0,:], sr=sampleRate, hop_length=1024, n_bins=84))
            disp.specshow(rosa.amplitude_to_db(fg_cqt, ref=np.max),
                sr=sampleRate, x_axis='time', y_axis='cqt_hz', hop_length=1024, ax=ax1)
            ax1.set_title('NUSSL REPET Foreground: CQT')

            disp.specshow(rosa.amplitude_to_db(bg_cqt, ref=np.max),
                sr=sampleRate, x_axis='time', y_axis='cqt_hz', hop_length=1024, ax=ax2)
            ax2.set_title('NUSSL REPET Background: CQT')
            fig.tight_layout()
            fig.savefig(os.path.join(localDir, song + '-cqt.png'))
            plt.close(fig)
            logger.info("REPET plots generated, now writing estimates to audio file")
            estimates[0].write_audio_to_file(os.path.join(localDir, song + '-fg.wav'))
            estimates[1].write_audio_to_file(os.path.join(localDir, song + '-bg.wav'))

    def nussl_timbre(self):

        localDir = os.path.join(self.m_songDir, 'nussl-timbre')
        if not os.path.isdir(localDir):
            os.mkdir(localDir)
        for song in self.m_sourceMap:
            separator = nussl.separation.primitive.TimbreClustering(self.m_sourceMap[song], num_sources=3, n_components=16, mask_type='binary')
            estimates = separator()
            # we want to take CQT of each component, expectation is that taali is high frequency, harmonium/
            # singer voice in the middle and tabla in the lower band. TODO: Need to check NUSSL implementation to find
            # in which order the components are returned, labelling is based on the hope that higher frequency bands
            # are represented first in the sources
            fig, (ax1, ax2, ax3) = plt.subplots(3,1)
            sampleRate = self.m_sourceMap[song].sample_rate
            print("NMF returned estimates=", len(estimates))
            taali_cqt = np.abs(rosa.cqt(estimates[0].audio_data[0,:], sr=sampleRate, hop_length=1024, n_bins=84))
            baja_cqt = np.abs(rosa.cqt(estimates[1].audio_data[0,:], sr=sampleRate, hop_length=1024, n_bins=84))
            tabla_cqt = np.abs(rosa.cqt(estimates[2].audio_data[0,:], sr=sampleRate, hop_length=1024, n_bins=84))
            disp.specshow(rosa.amplitude_to_db(tabla_cqt, ref=np.max),
                sr=sampleRate, x_axis='time', y_axis='cqt_hz', hop_length=1024, ax=ax1)
            ax1.set_title('NUSSL Timbre Taali?: CQT')

            disp.specshow(rosa.amplitude_to_db(baja_cqt, ref=np.max),
                sr=sampleRate, x_axis='time', y_axis='cqt_hz', hop_length=1024, ax=ax2)
            ax2.set_title('NUSSL Timbre Voice/Harmonium: CQT')

            disp.specshow(rosa.amplitude_to_db(taali_cqt, ref=np.max),
                sr=sampleRate, x_axis='time', y_axis='cqt_hz', hop_length=1024, ax=ax3)
            ax3.set_title('NUSSL Timbre Tabla?: CQT')
            fig.tight_layout()
            fig.savefig(os.path.join(localDir, song + '-cqt.png'))
            plt.close(fig)
            logger.info("NMF tibmre plots generated, now writing estimates to audio file")
            estimates[0].write_audio_to_file(os.path.join(localDir, song + '-taali.wav'))
            estimates[1].write_audio_to_file(os.path.join(localDir, song + '-baja.wav'))
            estimates[2].write_audio_to_file(os.path.join(localDir, song + '-tabla.wav'))

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

    def rosa_decompose(self, writeSeparatedAudio=False, generatePlots=False):
        localDir = os.path.join(self.m_songDir, 'rosa-decompose')
        if not os.path.isdir(localDir):
            os.mkdir(localDir)
        for song in self.m_sourceMap:
            sampleRate = self.m_sourceMap[song].sample_rate
            if sampleRate != 44100:
                logger.warn("Processing song=%s with sample rate=%d...\n", song, sampleRate)
            mfccs = rosa.feature.mfcc(self.m_sourceMap[song].audio_data[0,:], sr=sampleRate, n_mfcc=13)
            magSpectrum = np.abs(rosa.stft(self.m_sourceMap[song].audio_data[0,:], n_fft=1024))
            comps, acts = rosa.decompose.decompose(magSpectrum, n_components=32, sort=True)
            # Failed list comprehension :-()
            #tbBasis = [np.where(comps == x, comps) for x in np.argmax(comps, axis=0) if TaaliSeparator.e2_c3_c5_b5(x, sampleRate, 1024)]
            nmfMax = np.argmax(comps, axis=0)
            #tbBasis = np.empty((1,0), dtype=np.int64)
            tbBasis = np.array([3,4,5,6,28,29,30,31])
            '''for x in nmfMax:
                if TaaliSeparator.tablaTaaliRange(x, sampleRate, 1024):
                    # np.where returns a tuple, where latest documentation
                    # reports dArray?
                    tbBasis = np.append(tbBasis, np.where(nmfMax == x)[0])
            '''
            logger.info("Suspected NVM basis w.r.t tabla/taali")
            # Create placeholders for selected components and activations
            ttComponents = np.take(comps, tbBasis, axis=1)
            ttActivations = np.take(acts, tbBasis, axis=0)

            # Construct tabla-taali spectrum
            ttSpectrum = ttComponents.dot(ttActivations)
            ttAudio = rosa.istft(ttSpectrum)
            if generatePlots:
                ttMfcc = rosa.feature.mfcc(ttAudio, sr=sampleRate, n_mfcc=13)
                fig = plt.figure(figsize=(10,10))
                plt.subplot(2,1,1)
                disp.specshow(rosa.amplitude_to_db(ttSpectrum, ref=np.max),
                    sr=sampleRate, x_axis='time', y_axis='log', hop_length=1024)
                plt.title('Reconstructed tabla taali spectrum')
                plt.subplot(2,1,2)
                disp.specshow(ttMfcc, x_axis='time')
                plt.title('MFCC of tabla-taali extracted audio')
                plt.colorbar()
                fig.tight_layout()
                anFig, anAx = plt.subplots(nrows=1, ncols=2, figsize=(10,10))
                disp.specshow(rosa.amplitude_to_db(comps, ref=np.max),
                    sr=sampleRate, y_axis='log', hop_length=1024, ax=anAx[0])
                anAx[0].set_title('NMF Components')
                disp.specshow(acts, x_axis='time', ax=anAx[1])
                anAx[1].set_title('NMF activations')
                anFig.tight_layout()
                fig.savefig(os.path.join(localDir, song + '-tt-spec.png'))
                anFig.savefig(os.path.join(localDir, song + '-nmf.png'))
                plt.close(fig)
                plt.close(anFig)

            if writeSeparatedAudio:
                rosa.output.write_wav(os.path.join(localDir, song + '-tt.wav'), ttAudio, sr=sampleRate)
            else:
                self.m_separatedMap[song] = ttAudio

        # Save separated source audio data on disk for later processing
        np.save(os.path.join(localDir, self.m_ttFile), self.m_separatedMap)

    def cqt_model_fitting(self):
        # Minimum frequence for CQT, middle frequency and highest frequency used
        # all notated with midi notes
        C1 = 24
        C4 = 60
        C5 = 72
        C6 = 84
        C8 = 108
        # Corresponds to 7 octaves from C1 to C8
        FreqBins = C8 - C1
        localDir = os.path.join(self.m_songDir, 'model-fitting')
        if not os.path.isdir(localDir):
            os.mkdir(localDir)
        for song in self.m_sourceMap:
            logger.info("\nProcessing song=%s...", song)
            sampleRate = self.m_sourceMap[song].sample_rate
            overallCqt = np.abs(rosa.cqt(self.m_sourceMap[song].audio_data[0,:], sr=sampleRate, hop_length=1024, n_bins=FreqBins))
            cqtMed = rosa.decompose.nn_filter(overallCqt, aggregate=np.median, axis=-1)
            fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(10,10))
            #plt.subplot(2,2,1)
            #disp.specshow(rosa.amplitude_to_db(cqtMed, ref=np.max),
            #    sr=sampleRate, x_axis='time', y_axis='cqt_hz', hop_length=1024)
            #plt.title('Median filtered CQT')
            plt.subplot(2,2,1)
            pitchPower  = np.linalg.norm(cqtMed, axis=1)
            plt.title('Median CQT power estimate')
            # CQT starts with midi number 24 as minimum
            fullMidiRange = np.arange(C1, C8)
            plt.bar(fullMidiRange, pitchPower)

            # Divide up CQT power in three unequal ranges which will serve as independent variable
            # for model fitting
            # For tabla taali *mostly* the impact will overtake the other instruments in terms of
            # cqt power, so notes till C4 are considered. For harmonium and voice we focus on C4 to
            # C6 excluding the lower ocatves in order to save us from tabla/taali impact. The last
            # remaing range is then for higher pitches which we don't typically expect in the prelude
            # duration of Qawali presentation
            pxLower = pitchPower[C1-C1:C4-C1]
            mLower = fullMidiRange[C1-C1:C4-C1]
            pxMiddle = pitchPower[C4-C1 : C8-C1]
            mMiddle = fullMidiRange[C4-C1 : C8-C1]
            pxHigher = pitchPower[C4-C1:C8-C1]
            mHigher = fullMidiRange[C4-C1: C8-C1]

            # TODO: Tabla taali is the distinguishing feature, we expect power from these
            # sources to be a well-centered peak function, use Gaussian for fitting there.
            lowerModel = GaussianModel()
            lInitParams = lowerModel.guess(pxLower, x=mLower)
            resultLower = lowerModel.fit(pxLower, lInitParams, x=mLower)
            logger.info("Printing lower model results...")
            print(resultLower.fit_report(min_correl=0.5))


            # For middle and high octave models we are not sure of the distribution
            # hence relyin on central limit theorem
            middleModel = LorentzianModel()
            mInitParams = middleModel.make_params()
            mInitParams['center'].set(value=75, min=C5, max=C6)
            resultMiddle = middleModel.fit(pxMiddle, mInitParams, x=mMiddle)
            logger.info("Printing middle model results...")
            print(resultMiddle.fit_report(min_correl=0.5))

            highVModel = LorentzianModel(prefix='hvm_')
            highHModel = LorentzianModel(prefix='hhm_')
            hParams = highVModel.make_params()
            hParams['hvm_center'].set(value=70, min=66, max=76)
            hParams.update(highHModel.make_params())
            hParams['hhm_center'].set(value=76, min=82, max=86)
            highModel = highVModel + highHModel
            resultHigh = highModel.fit(pxHigher, hParams, x=mHigher)
            #logger.info("Printing high model results...")
            #print(resultHigh.fit_report(min_correl=0.5))
            plt.subplot(2,2,2)
            plt.plot(mLower, resultLower.init_fit, 'k--', label='initial fit')
            plt.plot(mLower, resultLower.best_fit, 'r-', label='best fit')
            plt.title('Tabla/Taali Fit')
            plt.grid()
            plt.legend(loc='best')
            plt.subplot(2,2,4)
            plt.plot(mMiddle, resultMiddle.init_fit, 'k--', label='initial fit')
            plt.plot(mMiddle, resultMiddle.best_fit, 'r-', label='best fit')
            plt.grid()
            plt.title('Voice/Harmonium Fit')
            plt.legend(loc='best')
            plt.subplot(2,2,3)
            plt.plot(mHigher, resultHigh.init_fit, 'k--', label='initial fit')
            plt.plot(mHigher, resultHigh.best_fit, 'r-', label='best fit')
            plt.legend(loc='best')
            plt.grid()
            fig.savefig(os.path.join(localDir, song + '-model-fit.png'))
            plt.close(fig)

    @staticmethod
    # Limit array to indices less than value
    def limit_index(array, limit):
        if limit < min(array):
            return 0
        return max(np.where(array <= limit)[0])

    @staticmethod
    def fit_distribution(featureName, featureData, songName, frame):
        logger.info("Curve fitting for feature=%s song=%s and Frame=%d", featureName, songname, frame)
        C1 =  24
        C8 = 108
        cqtXRange = np.arange(C1, C8)
        mfccXRange = np.arange(0, 20)
        # TODO: After first run of results define model constraints
        if featureName == 'cqt':
            cqtModel = GaussianModel()
            cqtParams = cqtModel.guess(featureData, x=cqtXRange)
            mFit = cqtModel.fit(featureData, cqtParams, x=cqtXRange)
        elif featureName == 'mfcc':
            mfccModel = LorentzianModel()
            mfccParams = mfccModel.fit(featureData, x=mfccXRange)
            mFit = mfccModel.fit(featureData, mfccParams, x=mfccXRange)
        else:
            logger.error("Feature not supported in curve fitting")
            raise RuntimeError

        print('-------------------------------')
        print('Parameter    Value       Stderr')
        for name, param in mFit.params.items():
            print('{:7s} {:11.5f} {:11.5f}'.format(name, param.value, param.stderr))
        print('-------------------------------')

    # Directory containing source separated map should be given as input
    def tt_classify(self, ssDir=None):
        if ssDir is None:
            raise RuntimeError
        else:
            # Load reconstructed audio only containing tabla/taali sources
            ttMap = np.load(os.path.join(ssDir, self.m_ttFile), allow_pickle=True).item()
            print(ttMap)
        sampleRate = 44100
        frameSize = 1024
        observeDurtation = 5
        observeWindow = sampleRate * observeDurtation // fameSize
        for song in ttMap:
            logger.info("Classification Loop for %s...", song)
            # Two main features, cqt power and mean mfcc evaluated over observe duration
            overallCqt = np.abs(rosa.cqt(ttMap[song], sr=sampleRate, hop_length=1024, n_bins=84))
            overallMfcc = rosa.feature.mfcc(ttMap[song], sr=sampleRate, n_mfcc=20)

            if overallCqt.shape[-1] != overallMfcc.shape[-1]:
                logger.error("Cqt and MFCC features are not time aligned")
                raise RuntimeError

            # Additional median filtering on CQT to smooth rough areas
            cqtMed = rosa.decompose.nn_filter(overallCqt, aggregate=np.median, axis=-1)

            totalFrames = overallCqt.shape[-1] // observeWindow
            logger.info("Total frames to be considered for classification %6.4f=", totalFrames)

            # Prepare for per-frame decision
            mfccDecisions = np.empty(totalFrames)
            cqtDecisions = np.empty(totalFrames)
            for frameIdx in np.arange(0, totalFrames):
                startFrame = frameIdx * observeWindow
                endFrame = (frameIdx + 1) * observeWindow
                pitchPower = np.linalg.norm(cqtMed[:,startFrame:endFrame], axis=1)
                mfccMean = np.mean(overallMfcc[:,startFrame:endFrame], axis=1)
                cqtDecisions[frameIdx] = fit_distribution('cqt', pitchPower, song, index)
                mfccDecisions[frameIdx] = fit_distribution('mfcc', mfccMean, song, index)

            # Outside the observation loop, comeup with a heuristic for overall decision
            # from the results of individual frames
            fig = plt.figure(figsize=(10,10))
            plt.subplot(2,1,1)
            plt.bar(np.arange(24,108), pitchPower)
            plt.grid()
            plt.title('CQT power profile')
            plt.subplot(2,1,2)
            disp.specshow(overallMfcc, x_axis='time', sr=sampleRate)
            plt.title('MFCC of tabla-taali extracted audio')
            plt.grid()
            fig.savefig(os.path.join(ssDir, song + '-test-classify.png'))
            plt.close(fig)


    def __del__(self):
        logger.info("%s Destructor called", __name__)



if __name__ == '__main__':
    # Initialize TaaliSeparator Object
    #ts = TaaliSeparator([], '/home/fsheikh/musik/desi-mix')
    #ts = TaaliSeparator([], '/home/fsheikh/musik/qawali')
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
    # Call various source separation algorithms
    # ts.nussl_timbre()
    #ts.cqt_model_fitting()
    #ts.rosa_decompose()
    ts = TaaliSeparator()
    ts.tt_classify('/home/fsheikh/musik/qawali/rosa-decompose')
