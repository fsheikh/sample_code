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
import numpy as np
import nussl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from pathlib import Path
import logging


logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class TaaliSeparator:
    def __init__(self, songList=[], songDir=None):
        # Constructor makes a map of NUSSL objects with song
        # names as key, if songList is given, only songs in list
        # are used, otherwise all songs from given directory used.
        self.m_songList = []
        if songDir is None:
            logger.error("No musik directory specified!")
            raise RuntimeError
        elif not os.path.isdir(songDir):
            logger.error("Directory %s not found", songDir)

        self.m_songDir = songDir
        self.m_songList = songList
        if not songList:
            logger.info("Song list empty, all mp3 songs in directory %s will be added", songDir)
            for root, dirname, songs in os.walk(songDir):
                self.m_songList += [s.rstrip('.mp3') for s in songs if s.endswith('.mp3')]
        logger.info("Following songs will be evaluated...")
        print(self.m_songList)

        self.m_separationMap = {}
        for song in self.m_songList:
            songPath = os.path.join(songDir, song+'.mp3')
            if os.path.isfile(songPath):
                self.m_separationMap[song] = nussl.AudioSignal(path_to_input_file=songPath,
                                                                sample_rate=44100, offset=0.0, duration=60.0)
                self.m_separationMap[song].to_mono(overwrite=True)
                logger.info("Song=%s SampleRate=%d", song, self.m_separationMap[song].sample_rate)
            else:
                logger.error("Song path=%s not found", songPath)

        print(self.m_separationMap)

    def nussl_repet(self):

        localDir = os.path.join(self.m_songDir, 'nussl-repet')
        if not os.path.isdir(localDir):
            os.mkdir(localDir)
        for song in self.m_separationMap:
            #separator = nussl.separation.primitive.Repet(self.m_separationMap[song], min_period=0.5, max_period=2.0,
            #    high_pass_cutoff=500.0, mask_type='binary')
            separator = nussl.separation.primitive.RepetSim(self.m_separationMap[song], high_pass_cutoff=400.0,
                mask_type='binary')
            estimates = separator()
            # we want to take CQT of foreground/background estimates, plot them, save the plots
            # and write estimates to a file
            fig, (ax1, ax2) = plt.subplots(2,1)
            sampleRate = self.m_separationMap[song].sample_rate
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
        for song in self.m_separationMap:
            separator = nussl.separation.primitive.TimbreClustering(self.m_separationMap[song], num_sources=3, n_components=16, mask_type='binary')
            estimates = separator()
            # we want to take CQT of each component, expectation is that taali is high frequency, harmonium/
            # singer voice in the middle and tabla in the lower band. TODO: Need to check NUSSL implementation to find
            # in which order the components are returned, labelling is based on the hope that higher frequency bands
            # are represented first in the sources
            fig, (ax1, ax2, ax3) = plt.subplots(3,1)
            sampleRate = self.m_separationMap[song].sample_rate
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

    def rosa_decompose(self):
        localDir = os.path.join(self.m_songDir, 'rosa-decompose')
        if not os.path.isdir(localDir):
            os.mkdir(localDir)
        for song in self.m_separationMap:
            logger.info("Processing song=%s...", song)
            sampleRate = self.m_separationMap[song].sample_rate
            overallCqt = np.abs(rosa.cqt(self.m_separationMap[song].audio_data[0,:], sr=sampleRate, hop_length=1024, n_bins=84))
            cqtMed = rosa.decompose.nn_filter(overallCqt, aggregate=np.median, axis=0)
            fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(10,10))
            disp.specshow(rosa.amplitude_to_db(overallCqt, ref=np.max),
                sr=sampleRate, x_axis='time', y_axis='cqt_hz', hop_length=1024, ax=ax[0])
            ax[0].set_title('Overall CQT')
            disp.specshow(rosa.amplitude_to_db(cqtMed, ref=np.max),
                sr=sampleRate, x_axis='time', y_axis='cqt_hz', hop_length=1024, ax=ax[1])
            ax[1].set_title('Median filtered CQT')

            anFig, anAx = plt.subplots(nrows=1, ncols=2, figsize=(10,10))
            comps, acts = rosa.decompose.decompose(cqtMed, n_components=4, sort=True)
            disp.specshow(rosa.amplitude_to_db(comps, ref=np.max),
                sr=sampleRate, y_axis='log', hop_length=1024, ax=anAx[0])
            anAx[0].set_title('NMF Components')
            disp.specshow(acts, x_axis='time', ax=anAx[1])
            anAx[1].set_title('NMF activations')
            fig.tight_layout()
            anFig.tight_layout()
            fig.savefig(os.path.join(localDir, song + '-freq-transform.png'))
            anFig.savefig(os.path.join(localDir, song + '-nmf.png'))
            plt.close(fig)
            plt.close(anFig)

    def __del__(self):
        logger.info("%s Destructor called", __name__)



if __name__ == '__main__':
    # Initialize TaaliSeparator Object
    # ts = TaaliSeparator(['khawaja', 'piya_say_naina'], '/home/fsheikh/musik')
    ts = TaaliSeparator([], '/home/fsheikh/musik')
    # Call various source separation algorithms
    # ts.nussl_timbre()
    ts.rosa_decompose()