# Module containing class whose objects represent a song
# downloaded from internet or a read from a local file. Method
# support extraction of features from this song

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
    C1Midi = 24
    C8Midi = 108
    FreqBins = C8Midi - C1Midi
    # Youtube-dl parameters for downloading from youtube
    YdlParams = {'postprocessors': [{ 'key': 'FFmpegExtractAudio', 'preferredcodec' : 'mp3', 'preferredquality' : '128'}],
                 'logger' : logger,
                 'outtmpl' : str(local_directory),
                 'noplaylist' : True
                }

    def __init__(self, song_name, url='None'):
        logger.info('Instantiating %s for song=%s with url=%s', self.__class__.__name__, song_name, url)
        self.m_songName = song_name
        self.m_songPath = AudioFeatureExtractor.local_directory / song_name
        self.m_url = url
        self.m_output = str(AudioFeatureExtractor.graphs_subdir / Path(song_name).stem)
        # Constants per instance (should probably be parameterized)
        self.m_sampleRate = 22050
        self.m_observeDurationInSec = 120
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
                with YoutubeDL(AudioFeatureExtractor.YdlParams) as ydl:
                    ydl.download([self.m_url])
            else:
                # Download from the own data-set shared via google drive
                gd.download(self.m_url, str(self.m_songPath), quiet=False, proxy=None)
        # Load snippet from song
        self.m_dataSong, self.m_sr = rosa.load(str(self.m_songPath), sr=self.m_sampleRate, mono=True,
                                        offset=0.0, duration=self.m_observeDurationInSec)

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

    def extract_cqt_dft(self, FFTSize=256):

        if (FFTSize % 2 != 0):
            log_error() << "Better to choose a power of 2 size for FFT"
            exit()

        HalfFFT = int(FFTSize / 2) + 1;

        Cqt = np.abs(rosa.cqt(self.m_dataSong, sr=self.m_sr, hop_length=self.m_hopLength, n_bins=AudioFeatureExtractor.FreqBins))
        logger.info("CQT Dimensions: Rows=%d Columns=%d", Cqt.shape[0], Cqt.shape[1])
        if (AudioFeatureExtractor.FreqBins != Cqt.shape[0]):
            logger.error("Unexpected CQT dimensions")
            exit()

        timeIndices = np.arange(0, Cqt.shape[1], HalfFFT)
        fftMatrix = np.zeros((Cqt.shape[0], timeIndices.size * HalfFFT))
        logger.info("Size of fft matrix mxn=%dx%d", fftMatrix.shape[0], fftMatrix.shape[1])

        # 50% overlapping FFT
        for index in timeIndices:
            if index == 0:
                paddingMatrix = np.zeros((AudioFeatureExtractor.FreqBins, HalfFFT))
                paddedCqt = np.append(paddingMatrix, Cqt[:,0:HalfFFT], axis=1)
                logger.info("padded mxn=%dx%d append HalfFFT=%d", paddedCqt.shape[0], paddedCqt.shape[1], HalfFFT)
                fftMatrix[:, 0 : HalfFFT] = abs(np.fft.rfft2(paddedCqt, [AudioFeatureExtractor.FreqBins, FFTSize], [0,1]))
            else:
                fftMatrix[:,index : index + HalfFFT] = abs(np.fft.rfft2(Cqt[:,index - HalfFFT - 1: index + HalfFFT - 1],
                                                   [AudioFeatureExtractor.FreqBins, FFTSize], [0,1]))
        logger.info("Cqt DFT calculation done")
        fig = plt.figure(figsize=(10,6))
        plt.subplot(2,1,1)
        rosa.display.specshow(rosa.amplitude_to_db(Cqt, ref=np.max),
            sr=self.m_sr, x_axis='time', y_axis='cqt_hz', hop_length=self.m_hopLength)
        plt.title('Raw CQT')
        plt.subplot(2,1,2)
        rosa.display.specshow(rosa.amplitude_to_db(fftMatrix, ref=np.max), sr=self.m_sr, x_axis='time', y_axis='cqt', hop_length=self.m_hopLength)
        plt.title('2DFT of CQT')
        fig.tight_layout()
        fig.savefig(self.m_output + '-cqt2dft.png')
        plt.close(fig)
        logger.info('***CQT-DFT extracted from %s ***\n', self.m_songName)


    def extract_beats(self):

        # Separate harmonics and percussives into two waveforms
        y_harmonic, y_percussive = rosa.effects.hpss(self.m_dataSong)
        logger.info('Harmonics, percussion separated...')

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
        logger.info('Calculating self-similarity based on MFCC')
        mfcc_y = rosa.feature.mfcc(self.m_dataSong, sr=self.m_sr, hop_length=self.m_hopLength)
        logger.info("mfcc: Rows=%d Columns=%d", mfcc_y.shape[0], mfcc_y.shape[1])
        norm_mfcc = np.linalg.norm(mfcc_y, axis=1, keepdims=True)
        mean_mfcc = np.mean(mfcc_y, axis=1, keepdims=True)
        mfcc_x = (mfcc_y - mean_mfcc) / norm_mfcc
        mfcc_sim = rosa.segment.cross_similarity(mfcc_x, mfcc_x, k=3)
        fig = plt.figure(figsize=(10,6))
        plt.subplot(2,1,1)
        plt.title('mean centered normalized mfcc')
        rosa.display.specshow(mfcc_x, sr=self.m_sr, x_axis='time', y_axis='time', hop_length=self.m_hopLength)
        plt.colorbar()
        plt.subplot(2,1,2)
        plt.title('self-similarity mfcc')
        rosa.display.specshow(mfcc_sim, sr=self.m_sr, x_axis='time', y_axis='time', hop_length=self.m_hopLength)
        fig.tight_layout()
        fig.savefig(self.m_output + '-mfcc.png')
        plt.close(fig)
        logger.info('***MFCC extracted from %s ***\n', self.m_songName)
