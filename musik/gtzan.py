# GTZAN data-set information wrapper

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

# Contains utility functions to load song and genre related information
# from extracted GTZAN data-set http://marsyas.info/downloads/datasets.html
import os
import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# Given directory path where gtzan has been unpacked
# and associated genre defined in that dataset, constructs a map
# with full location to song sample and its associated genre
# limit parameter determines how many songs from each genre should be loaded
# helpful to extract training/evaluation items

class GtzanMap:
    Genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

    def __init__(self, directoryPath=''):
        self.m_map = {}
        if not os.path.exists(directoryPath):
            logging.error("Genre directory=%s within dataset not found", directoryPath)
            raise ValueError
        else:
            self.m_directory = directoryPath

    def cmap(self, genre='', limit=100):
        genre_map = {}
        if self.m_directory == None or genre == None:
            logging.error("Invalid inputs to map constructor")
            raise ValueError

        genrePath = os.path.join(self.m_directory, genre)

        # GTZAN data-set contains wave files
        itemCount = 0
        for wavFile in os.listdir(genrePath):
            itemCount = itemCount + 1
            if not wavFile.endswith(".wav"):
                logging.warning("File=%s not in wave format, skipping", waveFile)
            else:
                mapKey = "gtzan_"+ os.path.splitext(os.path.basename(wavFile))[0]
                relativeSongPath = genre + "/" + wavFile
                logging.info("Adding %s to map at key = %s with genre=%s", relativeSongPath, mapKey, genre[0:2])
                genre_map[mapKey] = (relativeSongPath, genre[0:2])
            if itemCount == limit:
                break

        return genre_map
