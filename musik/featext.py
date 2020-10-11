# Multiple feature detection from a song URL

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

# Module containing class whose objects represent a song
# downloaded from internet or a read from a local file. Method
# support extraction of multiple features mainly using librosa utilities

import numpy as np
import librosa as rosa
import librosa.display as disp
import gdown as gd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from pathlib import Path
import logging
from youtube_dl import YoutubeDL


logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class AudioFeatureExtractor:

    # Change this location based on host machine, this
    # location will be used to download songs and a sub-directory
    # will contain graphs/extracted features
    # e.g. on windows local_directory = Path('C:\\personal\\musik\\')
    local_directory = Path('/home/fsheikh/musik')
    graphs_subdir = local_directory / 'graphs'
    gztan_subdir = local_directory / 'gtzan' / 'genres'
    C1Midi = 24
    C8Midi = 108
    FreqBins = C8Midi - C1Midi
    SubBands = 6
    # Youtube-dl parameters for downloading from youtube
    YdlParams = {'postprocessors': [{ 'key': 'FFmpegExtractAudio', 'preferredcodec' : 'mp3', 'preferredquality' : '128'}],
                 'logger' : logger,
                 'outtmpl' : str(local_directory),
                 'noplaylist' : True
                }

    def __init__(self, song_name, url='None', offset=0.0):
        logger.info('Instantiating %s for song=%s with url=%s', self.__class__.__name__, song_name, url)
        self.m_songName = song_name
        self.m_url = url
        if "gtzan" in url:
            self.m_songPath = AudioFeatureExtractor.gztan_subdir / song_name
            self.m_observeDurationInSec = 30
        else:
            self.m_songPath = AudioFeatureExtractor.local_directory / song_name
            self.m_observeDurationInSec = 30
        self.m_output = str(AudioFeatureExtractor.graphs_subdir / Path(song_name).stem) + '_' + str(offset)
        # Constants per instance (should probably be parameterized)
        self.m_sampleRate = 22050
        # With above sample rate, this length translates to 46ms
        self.m_hopLength = 1024
        # Collecting mean, std.deviation and norm of harmonic and percussive CQT (3*2 features)
        self.m_perMidi = np.empty([AudioFeatureExtractor.FreqBins, 4], dtype=float)
        # First column contains midi number of notes/freqeuncies
        self.m_perMidi[:,0] = np.arange(AudioFeatureExtractor.C1Midi, AudioFeatureExtractor.C8Midi, dtype=float)

        logger.info('Attempting to load %s', self.m_songPath)
        # if same file already has been downloaded skip it
        if os.path.exists(self.m_songPath):
            logger.info('File %s already exists, skipping Download', song_name)
        else:
            logger.info('Downloading %s from %s', song_name, url)
            if "youtube" in self.m_url:
                # update the output template in parameters and ask youtube-dl to download
                # Need to update the download path it seems library appends extenstion based on encoding
                AudioFeatureExtractor.YdlParams['outtmpl'] = str(AudioFeatureExtractor.local_directory / self.m_songPath.stem)
                ydl_opts = {'nocheckcertificate':True}
                with YoutubeDL(AudioFeatureExtractor.YdlParams) as ydl:
                    ydl.download([self.m_url])
            else:
                # Download from the own data-set shared via google drive
                gd.download(self.m_url, str(self.m_songPath), quiet=False, proxy=None)
        # Load snippet from song
        self.m_dataSong, self.m_sr = rosa.load(str(self.m_songPath), sr=self.m_sampleRate, mono=True,
                                        offset=offset, duration=self.m_observeDurationInSec)

        if (self.m_sr != self.m_sampleRate):
            logger.warning("Determined sample rate=%d different from assumed!", self.m_sr)

    def extract_cqt(self):

        # Separate harmonics and percussive components into two waveforms
        y_harmonic, y_percussive = rosa.effects.hpss(self.m_dataSong)

        logger.info('Harmonics, percussion separated...')

        fig, (ax1, ax2) = plt.subplots(2,1)
        C_harmonic = np.abs(rosa.cqt(y_harmonic, sr=self.m_sr, hop_length=self.m_hopLength, n_bins=AudioFeatureExtractor.FreqBins))
        C_percussive = np.abs(rosa.cqt(y_percussive, sr=self.m_sr, hop_length=self.m_hopLength, n_bins=AudioFeatureExtractor.FreqBins))
        logger.info("CQT Dimensions: Rows=%d Columns=%d", C_harmonic.shape[0], C_harmonic.shape[1])
        self.m_perMidi[:,1] = np.linalg.norm(C_harmonic, axis=1)
        self.m_perMidi[:,2] = np.mean(C_harmonic, axis=1)
        self.m_perMidi[:,3] = np.std(C_harmonic, axis=1)
        logger.info("Max Harmonic Norm=%12.8f with midi=%d", np.max(self.m_perMidi[:,1]), np.argmax(self.m_perMidi[:,1]) + AudioFeatureExtractor.C1Midi)
        rosa.display.specshow(rosa.amplitude_to_db(C_harmonic, ref=np.max),
            sr=self.m_sr, x_axis='time', y_axis='cqt_hz', hop_length=self.m_hopLength, ax=ax1)
        np.savetxt(self.m_output+'-cqt.txt', self.m_perMidi, 'midi=%6.4f: normH=%6.4f, meanH=%6.4f, std-devH=%6.4f')
        ax1.set_title('Harmonic CQT')

        rosa.display.specshow(rosa.amplitude_to_db(C_percussive, ref=np.max),
            sr=self.m_sr, x_axis='time', y_axis='cqt_hz', hop_length=self.m_hopLength, ax=ax2)
        ax2.set_title('Percussive CQT')
        fig.tight_layout()
        fig.savefig(self.m_output + '-cqt.png')
        plt.close(fig)
        logger.info('***CQT extracted from %s ***\n', self.m_songName)

    def extract_pcqt_dft(self, FFTSizeC=128, FFTSizeT=2048):

        if (FFTSizeC % 2 != 0 or FFTSizeT % 2 != 0):
            log_error() << "Better to choose a power of 2 size for FFT"
            exit()

        y_percussive = rosa.effects.percussive(self.m_dataSong, margin=10.0)

        #rosa.output.write_wav(self.m_output + '-yp.wav', y_percussive, self.m_sr)
        Cqt = np.abs(rosa.cqt(y_percussive, sr=self.m_sr, hop_length=self.m_hopLength, n_bins=AudioFeatureExtractor.FreqBins))
        logger.info("CQT Dimensions: Rows=%d Columns=%d", Cqt.shape[0], Cqt.shape[1])
        if (AudioFeatureExtractor.FreqBins != Cqt.shape[0]):
            logger.error("Unexpected CQT dimensions")
            exit()

        logger.info("Calculated Threshold on CQT")

        # CQT size corresponds to m_observeDurationInSec * m_sr / m_hopLength
        # We move the window every 4 seconds which means
        #WindowHop = int(4 * self.m_sr/ self.m_hopLength)
        #windowIndices = np.arange(0, ThresholdCqt.shape[1], WindowHop)
        fftMatrix = np.fft.fft2(Cqt, [FFTSizeC, FFTSizeT])
        fftShifted = np.abs(np.fft.fftshift(fftMatrix)**2)
        logger.info("Configured Size of fft matrix mxn=%dx%d", fftMatrix.shape[0], fftMatrix.shape[1])
        '''
        # Overlapping FFT
        for index in windowIndices:
            logger.info("Processing window index=%d", index)
            fftFull = np.fft.fft2(ThresholdCqt[:,index: index + FFTSize], [AudioFeatureExtractor.FreqBins, FFTSize], [0,1])
            fftFull = np.abs(np.fft.fftshift(fftFull)**2)
            # Keep only low energy freq components
            if index < ThresholdCqt.shape[1] - WindowHop:
                fftMatrix[:, index: index + WindowHop] = fftFull[:, 0 : WindowHop]
            else:
                logger.warning("window index=%d not processed!", index)
        '''
        fftdB = rosa.amplitude_to_db(fftShifted, ref=np.max)
        logger.info("Cqt DFT calculation done")
        fig = plt.figure(figsize=(10,6))
        plt.subplot(2,1,1)
        rosa.display.specshow(rosa.amplitude_to_db(Cqt, ref=np.max),
            sr=self.m_sr, x_axis='time', y_axis='cqt_hz', hop_length=self.m_hopLength)
        plt.colorbar()
        plt.title('Percussive CQT')
        plt.subplot(2,1,2)
        rosa.display.specshow(fftdB, sr=self.m_sr, x_axis=None, y_axis=None)
        plt.colorbar()
        plt.title('2DFT of CQT')
        fig.tight_layout()
        fig.savefig(self.m_output + '-pcqt2dft.png')
        plt.close(fig)
        logger.info('***CQT-DFT extracted from %s ***\n', self.m_songName)


    def extract_beats(self):

        # Extract percussive waveform
        y_percussive = rosa.effects.percussive(self.m_dataSong, margin=10.0)
        logger.info('Percussion extracted...')

        tempo, beat_timestamps = rosa.beat.beat_track(y_percussive, sr=self.m_sr, hop_length=self.m_hopLength, trim=True, units='time')
        tempo_envelope = 60.0 / np.gradient(beat_timestamps)
        stable_tempo_count = np.count_nonzero(tempo_envelope == tempo)
        logger.info("Beat timestamps length=%d, raw-tempo=%6.4f, stable-tempo-count=%d, mean-tempo=%6.4f, std-dev-tempo=%6.4f",
                    beat_timestamps.size, tempo, stable_tempo_count, np.mean(tempo_envelope), np.std(tempo_envelope))
        fig = plt.figure(figsize=(10,6))

        plt.subplot(2,1,1)
        plt.vlines(beat_timestamps, 0, 1, alpha=0.5, color='r', linestyle='--', label='Beats')
        plt.title('Beat Timestamps')
        plt.subplot(2,1,2)
        plt.plot(beat_timestamps, tempo_envelope)
        plt.title('Tempo Envelope')

        fig.tight_layout()
        fig.savefig(self.m_output + '-beat.png')
        plt.close(fig)
        logger.info('***Beats extracted from %s ***\n', self.m_songName)


    def extract_mfcc_similarity(self):
        logger.info('Calculating percussive self-similarity based on MFCC')
        y_percussive = rosa.effects.percussive(self.m_dataSong, margin=10.0)
        logger.info('Percussion extracted...')
        mfcc_y = rosa.feature.mfcc(y_percussive, sr=self.m_sr, hop_length=self.m_hopLength)
        logger.info("mfcc: Rows=%d Columns=%d", mfcc_y.shape[0], mfcc_y.shape[1])
        norm_mfcc = np.linalg.norm(mfcc_y, axis=1, keepdims=True)
        mean_mfcc = np.mean(mfcc_y, axis=1, keepdims=True)
        mfcc_x = (mfcc_y - mean_mfcc) / norm_mfcc
        mfcc_sim = rosa.segment.cross_similarity(mfcc_y, mfcc_y, k=3)
        fig = plt.figure(figsize=(10,6))
        plt.subplot(2,1,1)
        plt.title('mean centered normalized mfcc')
        rosa.display.specshow(mfcc_y, sr=self.m_sr, x_axis='time', y_axis='time', hop_length=self.m_hopLength)
        plt.colorbar()
        plt.subplot(2,1,2)
        plt.title('self-similarity mfcc')
        rosa.display.specshow(mfcc_sim, sr=self.m_sr, x_axis='time', y_axis='time', hop_length=self.m_hopLength)
        fig.tight_layout()
        fig.savefig(self.m_output + '-mfcc.png')
        plt.close(fig)
        logger.info('***MFCC extracted from %s ***\n', self.m_songName)

    def extract_low_timber(self, N=2048):
        # Timber on percussive component
        y_percussive = rosa.effects.percussive(self.m_dataSong, margin=10.0)
        logger.info('Percussion extracted')

        # Experiments on local data set indicated that at least for Qawali songs
        # contrast and flatness seem like good candidates

        #y_centroid = rosa.feature.spectral_centroid(self.m_dataSong, sr=self.m_sr, n_fft=N, hop_length=self.m_hopLength)
        #logger.info('Centroid calculated with size=%d', y_centroid.shape[-1])
        #fig = plt.figure(figsize=(10,6))
        #plt.subplot(3,2,1)
        #plt.title('Spectral Centroid')
        #frame_time = np.arange(0, y_centroid.shape[-1] / self.m_sr, 1 / self.m_sr)
        #plt.semilogy(self.m_hopLength * frame_time, y_centroid.T)
        #plt.ylabel('Hz')
        #plt.xlabel('Time (sec)')
        #plt.grid(True)

        #y_bw = rosa.feature.spectral_bandwidth(self.m_dataSong, sr=self.m_sr, n_fft=N, hop_length=self.m_hopLength)
        #logger.info("Spectral bandwidth calculated")
        #plt.subplot(3,2,2)
        #plt.title('Spectral Bandwidth')
        #frame_time = np.arange(0, y_bw.shape[-1] / self.m_sr, 1 / self.m_sr)
        #plt.semilogy(self.m_hopLength * frame_time, y_bw.T)
        #plt.ylabel('Hz')
        #plt.xlabel('Time (sec)')
        #plt.grid(True)

        fig = plt.figure(figsize=(10,6))
        plt.subplot(2,1,1)
        y_contrast = rosa.feature.spectral_contrast(self.m_dataSong, sr=self.m_sr, n_fft=N, hop_length=self.m_hopLength, fmin=160.0, n_bands=6)
        logger.info("Spectral constrat calculated")
        plt.title("Spectral Contrast")
        rosa.display.specshow(y_contrast, sr=self.m_sr, x_axis='time', hop_length=self.m_hopLength)
        plt.ylabel('Octave subbands')
        plt.tight_layout()

        y_flatness = rosa.feature.spectral_flatness(self.m_dataSong, n_fft=N, hop_length=self.m_hopLength)
        logger.info('Spectral flatness calculated')
        plt.subplot(2,1,2)
        plt.title('Spectral flatness')
        frame_time = np.arange(0, y_flatness.shape[-1] / self.m_sr, 1 / self.m_sr)
        # Tonality defined by G Peeters "A large set of audio features for sound description"
        plt.plot(self.m_hopLength * frame_time, 10 * np.log10(y_flatness.T))
        plt.ylim([-60, 0])
        plt.ylabel('Tonality')
        plt.xlabel('Time (sec)')
        plt.grid(True)

        #y_rolloff = rosa.feature.spectral_rolloff(self.m_dataSong, sr=self.m_sr, n_fft=N, hop_length=self.m_hopLength)
        #logger.info('Spectral Roll-off calculated')
        #plt.subplot(3,2,5)
        #plt.title('Spectral Rolloff')
        #frame_time = np.arange(0, y_rolloff.shape[-1] / self.m_sr, 1 / self.m_sr)
        #plt.semilogy(self.m_hopLength * frame_time, y_rolloff.T)
        #plt.ylabel('Hz')
        #plt.xlabel('Time (sec)')
        #plt.grid(True)

        #y_zerocross = rosa.feature.zero_crossing_rate(self.m_dataSong, frame_length=N, hop_length=self.m_hopLength)
        #logger.info("Zero crossing rate calculated")
        #plt.subplot(3,2,6)
        #plt.title('Zero crossing rate')
        #frame_time = np.arange(0, y_zerocross.shape[-1] / self.m_sr, 1 / self.m_sr)
        #plt.plot(self.m_hopLength * frame_time, y_zerocross.T)
        #plt.ylabel('ZC Rate')
        #plt.xlabel('Time (sec)')
        #plt.grid(True)

        fig.savefig(self.m_output + '-percussive-low-timber.png')
        plt.close(fig)
        logger.info('***Low timber features extracted from %s ***\n', self.m_songName)

    # Features directly extracted from Audio, which can help distinguish
    # Qawalis from other genres, time_reduce is duration in seconds for which
    # a reduction on time samples is peformed, for now the reduction function
    # supported in L2 norm. Default no reduction is applied
    def extract_qawali_features(self, time_reduce=0):
        qFeatures = {"FrameSize": self.m_hopLength,
                        "SampleRate": self.m_sr,
                        "PitchEnergy": np.array(np.zeros(self.m_hopLength)),
                        "SpectralContrast": np.array(np.zeros(self.m_hopLength)),
                        "SpectralFlatness": np.array(np.zeros(self.m_hopLength))}

        if time_reduce > self.m_observeDurationInSec:
            logger.error("Time reduction=%d invalid duration", time_reduce)
            raise ValueError

        harmonicSignal = rosa.effects.harmonic(self.m_dataSong, margin=10.0)
        percussiveSignal = rosa.effects.percussive(self.m_dataSong, margin=10.0)
        #rosa.output.write_wav(self.m_output + 'percussive-segment.wav', percussiveSignal, self.m_sr)
        # Previous experiments showed that Qawali recordings distinguish themselves
        # with energy of fundamental pitches and their harmonics. This is presumably since pitch
        # profile across this genre is similar (?)
        pitchEstimates = np.abs(rosa.cqt(harmonicSignal, sr=self.m_sr, hop_length=self.m_hopLength, n_bins=AudioFeatureExtractor.FreqBins))
        pitchFrames = np.arange(0, pitchEstimates.shape[-1] / self.m_sr, 1 / self.m_sr)
        logger.info("Pitch estimates size=%dx%d pitch frames=%d", pitchEstimates.shape[0], pitchEstimates.shape[1], pitchFrames.size)
        fig = plt.figure(figsize=(10,6))
        plt.subplot(2,2,1)
        rosa.display.specshow(rosa.amplitude_to_db(pitchEstimates, ref=np.max),
            sr=self.m_sr, x_axis='time', y_axis='cqt_hz', hop_length=self.m_hopLength)
        plt.title('Harmonic Pitch Profile')
        pitchEnergy  = np.linalg.norm(pitchEstimates, axis=1)
        #logger.info("Pitch energy size=%d", pitchEnergy.size)
        plt.subplot(2,2,2)
        plt.title('Pitch Energy')
        plt.bar(np.arange(0,pitchEnergy.size), pitchEnergy)
        # TODO: Convert to midi numbers
        plt.xlabel('Pitch Number')
        # In order to detect taali/tabla combination we want to look at low level timbre
        # features, manual experiments based on some liteature study on timbral features revealed
        # following two features appear to be distinctive for qawali recordings.
        # Spectral constrat fmin*n_bands can't seem to be higher than a threshold (could not detect a direct relation to Nyquist rate?)
        plt.subplot(2,2,3)
        specContrast = rosa.feature.spectral_contrast(percussiveSignal, sr=self.m_sr, n_fft=2048, hop_length=self.m_hopLength,
                                                     fmin=220.0, n_bands=AudioFeatureExtractor.SubBands - 1, quantile=0.1)
        qFeatures['SpectralContrast'] = specContrast
        plt.title("Spectral Contrast")
        rosa.display.specshow(specContrast, sr=self.m_sr, x_axis='time', hop_length=self.m_hopLength)
        plt.ylabel('Octave subbands')
        plt.tight_layout()

        aggregateContrast = np.linalg.norm(specContrast, axis=1)
        subbands = np.arange(0, AudioFeatureExtractor.SubBands)
        plt.subplot(2,2,4)
        plt.bar(subbands, aggregateContrast)
        fig.savefig(self.m_output + '-qfeatures.png')
        plt.close(fig)
        logger.info("Qawali related features computed!")

        # Report features based on reduction parameter after normalization
        if time_reduce != 0:
            # Number of frames in one second
            framesPerSecond = pitchEstimates.shape[1] / self.m_observeDurationInSec
            pEnergy = np.empty((pitchEstimates.shape[0], int(self.m_observeDurationInSec / time_reduce)))
            cEnergy = np.empty((specContrast.shape[0], int(self.m_observeDurationInSec / time_reduce)))
            for frameIdx in np.arange(0, self.m_observeDurationInSec, time_reduce):
                startFrame = int(frameIdx * framesPerSecond)
                endFrame = startFrame + int(time_reduce * framesPerSecond)
                outIndex = int(frameIdx/time_reduce)
                pEnergy[:,outIndex] = np.linalg.norm(pitchEstimates[:,startFrame:endFrame], axis=1)
                cEnergy[:,outIndex] = np.linalg.norm(specContrast[:,startFrame:endFrame], axis=1)
            qFeatures['PitchEnergy'] = pEnergy / pEnergy.max(axis=1, keepdims=True)
            qFeatures['SpectralContrast'] = cEnergy / cEnergy.max(axis=1, keepdims=True)
        else:
            #qFeatures['PitchEnergy'] = pitchEstimates / pitchEstimates.max(axis=1, keepdims=True)
            #qFeatures['SpectralContrast'] = specContrast / specContrast.max(axis=1, keepdims=True)
            qFeatures['PitchEnergy'] = pitchEstimates
            qFeatures['SpectralContrast'] = specContrast

        # Blindly dividing by max can backfire in case of all zero-values
        #np.nan_to_num(qFeatures['PitchEnergy'], copy=False)
        #np.nan_to_num(qFeatures['SpectralContrast'], copy=False)
        return qFeatures
"""
        specFlatness = rosa.feature.spectral_flatness(percussiveSignal, n_fft=2048, hop_length=self.m_hopLength)
        flatnessDelta = np.median(specFlatness)
        # if median flatness is less than -20dB, we replace it with the minimum value
        #if flatnessDelta < 0.01:
        #    flatnessDelta = 0.01
        #flatnessPeaks = rosa.util.peak_pick(specFlatness[0,:], 10,10,10,10, flatnessDelta, 5)
        # Numpy "where" is useful but does not work with 'and' return is a tuple with
        # actual indices in second element
        stableFrames = np.where(np.logical_and(specFlatness < 0.05, specFlatness > 0.01))[1]
        qFeatures['SpectralFlatness'] = np.array(np.zeros(specFlatness.size))
        qFeatures['SpectralFlatness'][stableFrames] = 1.0
        plt.subplot(2,2,4)
        plt.title('Spectral flatness')
        frame_time = np.arange(0, specFlatness.shape[-1] / self.m_sr, 1 / self.m_sr)
        logger.info("Time frame size=%d", frame_time.size)
        #plt.plot(self.m_hopLength * frame_time, specFlatness.T)
        plt.vlines(self.m_hopLength * frame_time[stableFrames], 0.01, 0.05, color='r', linestyles='dashed', alpha=0.8, label='Stable Frames')
"""
