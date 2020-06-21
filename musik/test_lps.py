# Rudimentary tests for finding longest sequence utility within
# DesiGenreDetector

import numpy as np
import os
import logging
from classify import pac_elements
from classify import DesiGenreDetector
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# TODO: Conver to unittest framework
if __name__ == "__main__":
    logger.info("Starting test for finding longest sequence within array")

    testVector = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0])
    longestSeq = pac_elements(testVector, 1.0)
    if longestSeq != (5,3):
        logger.error("Test case 1 failed")
        raise ValueError
    longestSeq = pac_elements(np.zeros(5), 1.0 )
    if longestSeq != (0,0):
        logger.error("Test case 2 failed")
        raise ValueError
    longestSeq = pac_elements(np.ones(5), 1.0)
    if longestSeq != (5,5):
        logger.error("Test case 3 failed")
        raise ValueError
    testVector = np.array([48.0, 47.0, -1.0, 5.0, 0.0, 5.0, 5.0, 99.9, 77.7])
    longestSeq = pac_elements(testVector, 5.0)
    if longestSeq != (3,2):
        logger.error("Test case 4 failed")
        raise ValueError
    testVector = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    longestSeq = pac_elements(testVector, 1.0)
    if longestSeq != (1,1):
        logger.error("Test case 5 failed")
        raise ValueError
    logger.info("Test Passed!")
else:
    logger.info("Unexpected module=%s called", __name__)