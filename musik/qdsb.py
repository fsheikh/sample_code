# qawali dataset builder(qsdb)
# Constructs reference qawali dataset from metadata json

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

import argparse
import logging
from pathlib import Path
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

class QawaliDataSet:

   def  __init__(self, target_path, metadata_file, offline_location=None):
        logger.info("Qawali dataset construction started")


if __name__ == "__main__":
    prog_parser = argparse.ArgumentParser("Arguments for qawali dataset builder")
    prog_parser.add_argument("datapath", type=str, help="Folder/directory path where qawali reference dataset will be built")
    prog_parser.add_argument("metadata", type=str, help="Json metadata file describing reference qawali dataset")
    prog_parser.add_argument("--opath",  type=str, dest="offline_path",
                             help="Folder/directory to look for qawali songs. Alternate to internet download")

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

    qds = QawaliDataSet(d_path, m_path, o_path)
