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
                logger.info("Song=%s SampleRate=%d", song, self.m_separationMap[song].sample_rate)
            else:
                logger.error("Song path=%s not found", songPath)

        print(self.m_separationMap)

        def __del__(self):
            logger.info("%s Destructor called", __name__)

if __name__ == '__main__':
    # Initialize TaaliSeparator Object
    #ts = TaaliSeparator(['khawaja', 'piya_say_naina'], '/home/fsheikh/musik')
    ts = TaaliSeparator([], '/home/fsheikh/musik')
    # Call various source separation algorithms
    # Collect results, plot for later evaluation