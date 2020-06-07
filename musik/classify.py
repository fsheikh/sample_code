
# Module containing class whose objects are initialized with
# a suitable feature set extracted from a song, and then supply
# different methods heurisitcs based or machine learning to
# classify the song as a South Asian genre like Qawali, Ghazal,
# geet, kafi, thumri, etc...

import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class DesiGenreDetector:
    # Name of features expected by classifier
    # This information is determined by studying various feature-sets
    # This modules assumes that a feature set has been agreed upon
    # with user
    expQfeatures = ['energy', 'contrast', 'flatness']
    def __init__(self, featured_dict):
        logger.info("Initializing Qawali classifier")

    def isQawali(self):
        logger.info("Placeholder for holistic based algorithm")
        return False