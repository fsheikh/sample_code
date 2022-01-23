# Copyright (C) 2020-2021  Faheem Sheikh (fahim.sheikh@gmail.com)
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

# qawali dataset builder(qdsb)
# Constructs reference qawali dataset from associated metadata information
# It reads/downloads original songs, then extracts short duration audio from them,
# finally writing back to a user-defined data-set location in a compressed format.

import argparse
from ffmpy import FFmpeg as ffm
import gdown as gd
import json
import librosa as rosa
import logging
from pathlib import Path
import os
import subprocess
from youtube_dl import YoutubeDL
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# Loads json metadata describing QawalRang dataset, constructs dataset
# and reports statistics from it.
class QawalRang:

    # These constants may be converted to params at some later point
    # Ensure all songs are sampled at CD-quality
    SampleRate = 44100
    # Source/Input format is mp3 supported by most sources, whether disks or online
    # Dataset entries will also be produces in this format
    InFormat = '.mp3'
    # Intermediate format is wav used for processing since most audio libraries support
    # IO in this format
    InterFormat = '.wav'
    # Sources chart
    QawalRangSources = 'sources.png'
    # Artist chart
    QawalRangArtists = 'artist.png'
    # Number of songs sourced from web (TODO code into metadata, not expected to change though)
    WebSize = 6
    def  __init__(self, target_path, metadata_file, offline_location=None):
        logger.info("Qawali dataset construction started")
        self.m_qmap = {}
        self.m_target = target_path
        self.m_local = offline_location
        if offline_location is None:
            logger.warning("No offline location specified, all songs will be downloaded")
        with open(metadata_file) as j_file:
            self.m_qmap = json.load(j_file)

    # downloads a song file locally, input is one element of json array describing
    # each qawali item, duration and start offset are included in each json entry
    def __download(self, song):
        if song['name'] == None or song['url'] == None:
            logger.error("Cannot download song {} from file {}".format(song['name'], song['url']))
            return
        else:
            logger.info("Download requested for {} from {}".format(song['name'], song['url']))
            out_song = Path(self.m_target) / song['fid']
            tmp_song = Path(self.m_target) / song['name']
            if out_song.with_suffix(QawalRang.InFormat).exists():
                logger.info("song {} already exists, skipping...".format(str(tmp_song), str(out_song)))
                return
            # Two online sources supported for now
            # youtube for publically available qawali performances
            # google-drive uploaded from personal collections
            if not tmp_song.with_suffix(QawalRang.InFormat).exists():
                if "youtube" in song['url']:
                    ydl_params = {
                    'format': 'bestaudio/best',
                    'postprocessors': [{
                        'key': 'FFmpegExtractAudio', 'preferredcodec' : 'mp3', 'preferredquality' : '128'
                    }],
                    'logger' : logger,
                    'noplaylist' : True,
                    'progress_hooks' : [QawalRang.download_progress]
                    }
                    ydl_params['outtmpl'] = str(tmp_song.with_suffix(QawalRang.InFormat))
                    ydl_opts = {'nocheckcertificate':True}
                    with YoutubeDL(ydl_params) as ydl:
                        try:
                            ydl.download([song['url']])
                        except Exception as e:
                            logger.error("Failed to download from youtube with error: {}".format(e))
                            return
                elif "google" in song['url']:
                    try:
                        out_name = gd.download(song['url'], str(tmp_song.with_suffix(QawalRang.InFormat)), quiet=True, proxy=None)
                        logger.info("Song {} downloaded from google-drive".format(out_name))
                    except Exception as e:
                        logger.error("Failed to download from google-drive with error: {}".format(e))
                        return
                else:
                    logger.error("Unsupported link: {}".format(song['url']))
                    return

            # We should have a downloaded audio file, trim it to required duration
            try:
                arg_start = str(song['start'])
                arg_duration = str(song['duration'])
                trimmer = ffm(inputs={str(tmp_song.with_suffix(QawalRang.InFormat)): [ '-ss', arg_start]},
                            outputs={str(out_song.with_suffix(QawalRang.InFormat)): ['-t', arg_duration]}
                            )
                print(trimmer.cmd)
                t_stdout, t_stderr = trimmer.run(input_data=None, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except Exception as e:
                logger.error("Failed to trim downloaded song {} with error: {}".format(str(tmp_song), e))
            # remove full downloaded file afterwards
            os.remove(str(tmp_song.with_suffix(QawalRang.InFormat)))

    # Reads a local song, extracts given duration, writes it in an intermediate format
    # before converting it into a compressed form
    def __write(self, song):
        # full path to offline location of song
        song_location = Path(self.m_local) / song['name']
        in_song = song_location.with_suffix(QawalRang.InFormat)
        if not in_song.exists():
            logger.error("{} not found on local file system, cannot write to dataset".format(in_song))
            return
        out_song = Path(self.m_target / song['fid'])
        if out_song.with_suffix(QawalRang.InFormat).exists():
            logger.warning("{} already exists skipping...".format(out_song))
            return
        logger.info("Loading local file {}".format(str(in_song)))
        song_data, sr = rosa.load(path=in_song, sr= QawalRang.SampleRate, mono=True, offset=float(song['start']),
                              duration=float(song['duration']), dtype='float32')
        if sr != QawalRang.SampleRate:
            logger.error("Extracted song has different sample rate {}".format(sr))
            return
        rosa.output.write_wav(str(out_song.with_suffix(QawalRang.InterFormat)), y=song_data, sr=QawalRang.SampleRate)
        try:
            compressor = ffm(inputs={str(out_song.with_suffix(QawalRang.InterFormat)): None},
                            outputs={str(out_song.with_suffix(QawalRang.InFormat)): None}
                            )
            # TODO: Redirect stderr/stdout to logger
            c_out, c_err = compressor.run(input_data=None, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception as e:
            logger.error("Failed to compress {} with error {}".format(str(out_song), e))
        # delete intermediate file
        os.remove(str(out_song.with_suffix(QawalRang.InterFormat)))

    # Makes the data-set according to supplied metadata
    def make(self):
        logger.info("Making dataset...")
        for qawali in self.m_qmap['qawalian']:
            if qawali['download']:
                self.__download(qawali)
            else:
                if self.m_local is None:
                    logger.error("No local path and no URL specified for {} skipping...".format(qawali['name']))
                else:
                    self.__write(qawali)

    def clean(self):
        logger.info("Deleting all songs from location {}".format(self.m_target))

    # Reports statistics from data-set
    def info(self):
        logger.info("Extracting information from dataset {}".format(self.m_target))
        # songs sourced from youtube, personal collection and web
        sourceLabels = ['youtube', 'personal', 'web']
        qawaliSizes = [0, -QawalRang.WebSize, QawalRang.WebSize]
        sourceColors = plt.get_cmap('magma')(np.linspace(0.25, 0.75, len(sourceLabels)))
        songsAnalysed = 0
        # map to chart arstit distribution
        artistMap = {}
        for qawali in self.m_qmap['qawalian']:
            qPath = self.m_target / qawali['fid']
            if not qPath.with_suffix(QawalRang.InFormat).exists():
                logger.info("Song {} not part of dataset? will NOT be included in statistics".format(qawali['id']))
            else:
                songsAnalysed = songsAnalysed + 1
                if 'youtube' in qawali['url']:
                    qawaliSizes[0] = qawaliSizes[0] + 1
                if 'google' in qawali['url']:
                    qawaliSizes[1] = qawaliSizes[1] + 1
                if qawali['artist'] in artistMap:
                    artistMap[qawali['artist']] = artistMap[qawali['artist']] + 1
                else:
                    artistMap[qawali['artist']] = 1

        # pie-chart requires manual conversion to percentages, okay here since labels are limited
        qawaliSizes = [ qs * 100 / songsAnalysed for qs in qawaliSizes if songsAnalysed != 0 ]
        print(qawaliSizes)

        fig = plt.figure(figsize=(10,8))
        axes = fig.add_subplot(111)
        axes.pie(qawaliSizes, labels=sourceLabels, colors=sourceColors, autopct="%1.00f%%", startangle=90)
        axes.axis('equal')
        plt.tight_layout()
        plt.savefig(QawalRang.QawalRangSources)
        plt.close(fig)

        # all songs with one artists are added to 'others' category
        artistMap['others'] = 0
        for artist in artistMap:
            if artistMap[artist] == 1:
                artistMap['others'] = artistMap['others'] + 1

        # copy dictionary while removing artist with single song to their credit
        filteredArtistMap = {key:val for key, val in artistMap.items() if key is 'others' or val != 1}
        artistNames = filteredArtistMap.keys()
        artistCount = filteredArtistMap.values()
        print(artistNames)
        print(artistCount)
        artistColors = plt.get_cmap('plasma')(np.linspace(0.25, 0.95), len(artistNames))
        artistAxis = np.arange(len(filteredArtistMap))
        fig2 = plt.figure(figsize=(10,8))
        axes2 = fig2.add_subplot(111)
        axes2.barh(artistAxis, artistCount, align='center')
        axes2.set_yticks(artistAxis)
        axes2.set_yticklabels(artistNames)
        axes2.invert_yaxis()
        axes2.set_xlabel('Number of songs')
        axes2.set_title('QawalRang: Artist map')
        plt.tight_layout()
        plt.savefig(QawalRang.QawalRangArtists)
        plt.close(fig2)

    @staticmethod
    def download_progress(prog):
        if prog['status'] == 'downloading':
            logger.info("Download prgogress: {}".format(prog['_percent_str']))
        return

if __name__ == "__main__":
    prog_parser = argparse.ArgumentParser("Arguments for qawali dataset builder")
    prog_parser.add_argument("datapath", type=str, help="Folder/directory path where qawali reference dataset will be built")
    prog_parser.add_argument("metadata", type=str, help="Json metadata file describing reference qawali dataset")
    prog_parser.add_argument("--opath",  type=str, dest="offline_path",
                             help="Folder/directory to look for qawali songs. Alternate to internet download")
    prog_parser.add_argument("--info",  dest="info", action="store_true",
                             help="Asks the program to report dataset statistics")

    prog_args = prog_parser.parse_args()

    d_path = Path(prog_args.datapath)
    m_path = Path(prog_args.metadata)
    try:
        o_path = Path(prog_args.offline_path)
        if not o_path.exists():
            logger.warning("Offline path not given or invalid, download of songs will be attemtped")
    except TypeError:
        o_path = Path("not/provided")
        logger.warning("No offline path provided")

    if not d_path.exists():
        logger.error("Target path {} does not exist".format(prog_args.datapath))
        exit(1)
    if not m_path.exists():
        logger.error("Metadata file {} does not found".format(prog_args.datapath))
        exit(1)

    logger.info("dataset construction site:{}, metadata-file:{}, offline-path:{}".format(str(d_path), str(m_path), str(o_path)))

    qds = QawalRang(d_path, m_path, o_path)

    # In case information flag is given dataset is assumed to be already constructed in the given directory
    if prog_args.info:
        qds.info()
    else:
        qds.make()
