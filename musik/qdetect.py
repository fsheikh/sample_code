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
    EX_FEATURES_FILE = 'feature-data.npy'
    SUPPORTED_SAMPLE_RATE = 44100
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
        audioMapPath = Path(self.m_songDir / QDetect.RAW_DATA_FILE)
        if not reload and not os.path.isfile(audioMapPath):
            logger.info("Audio map {} does not exist, need to reload data?\n\n", audioMapPath)
            raise RuntimeError

        # When reload option is used, existing data is simply overwritten
        if reload:
            logger.info("Data reload requested, all mp3/au songs in directory %s will be processed", songDir)
            for root, dirname, songs in os.walk(songDir):
                self.m_songList += [os.path.splitext(s)[0] for s in songs if s.endswith('.mp3')]
                self.m_songList += [os.path.splitext(s)[0] for s in songs if s.endswith('.au')]

            logger.info("Following songs will used for evaluation {}...".format(self.m_songList))

            for song in self.m_songList:
                songPath = Path(songDir / song+'.mp3')
                songPathA = Path(songDir / song+'.au')
                if songPath.exists():
                    self.m_rawData[song], sr = rosa.load(path=songPath, sr=None, mono=True, offset=0.0, dtype='float32')
                elif songPathA.exists():
                    self.m_rawData[song] = rosa.load(path=songPathA, sr=None, offset=0.0, dtype='float32')
                else:
                    logger.warning("Song paths={%s, %s} not found", songPath, songPathA)
        else:
            # load from existing audio data map
            self.m_rawData = np.load(audioMapPath, allow_pickle=True).item()
        # Debug only
        print("Data to be processed {}".format(self.m_rawData))
        if reload:
            # delete existing file
            audioMapPath.unlink() if audioMapPath.exists() else None
            np.save(audioMapPath, self.m_sourceMap)

        def decompose(self):
            logger.info("Placeholder for decomposition of raw-data into features")

        def classify(self):
            logger.info("Placeholde for classification of song as Qawali/Non-Qawali based on extracted features")

if __name__ == '__main__':
    qParser = argparse.ArgumentParser(description="Qawali genre detection program")
    qParser.add_argument("sdir", type=str, help="folder/directory containing songs to be evaluated")
    qParser.add_argument("--reload", type=str, dest="reload", help="reload data from songs (required at least once per songs directory)")
    qParser.add_argument("--extract", type=bool, dest="extract", help="extract suitable audio features from raw data (required at least once)")

    qArgs = qParser.parse_args()

    qGenre = QDetect(qArgs.sdir, qArgs.reload)

    if qArgs.extract:
        qGenre.decompose()

    qGenre.classify()


    # TODO: may be dump results in a format which can be easily processed/plotted.

