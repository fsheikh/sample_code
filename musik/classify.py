
# Module containing class whose objects are initialized with
# a suitable feature set extracted from a song, and then supply
# dectection methods utilizing either heurisitcs or machine learning to
# classify the song as a South Asian genre like Qawali, Ghazal,
# geet, kafi, thumri, etc...

from itertools import groupby
from sklearn.cluster import KMeans
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG, format='%(message)s')
logger = logging.getLogger(__name__)

# Function takes a 1-D numpy array as input and looks for the longest
# sequence of given element
def consecutive_detected_frames(framesArray, elemToMatch):
    # only works for 1-D array
    if (framesArray.size != framesArray.shape[-1]):
        logger.error("Input not a 1D array? with size=%d", framesArray.size)
        return 0

    if not np.any(framesArray == elemToMatch ):
        logger.warning("No matching element found")
        return 0

    retVal = max([sum(1 for i in g) for k,g in groupby(framesArray)])
    logger.info("Longest sequence length=%d of match=%6.4f", retVal, elemToMatch)
    return retVal


class DesiGenreDetector:
    # Name of features expected by classifier
    # This information is determined by studying various feature-sets
    # This modules assumes that a feature set has been agreed upon
    # with user
    ExpectedFeatures = ['FrameSize', 'SampleRate', 'PitchEnergy', 'SpectralContrast', 'SpectralFlatness']
    ObserveDurationSec = 3
    ClusterGroups = 2
    MidiC1 = 24
    MidiE2 = 40.0
    MidiA2 = 44.0
    def __init__(self, featureDict):
        logger.info("Initializing Qawali classifier")
        if not all(feature in DesiGenreDetector.ExpectedFeatures for feature in featureDict):
            logger.error("Features missing for detector")
            raise ValueError
        else:
            inputSizes  = (featureDict['PitchEnergy'].shape[-1], featureDict['SpectralContrast'].shape[-1],
                            featureDict['SpectralFlatness'].shape[-1])
            if not all(x==inputSizes[0] for x in inputSizes):
                logger.error("Features vectors not time aligned?")
                raise ValueError
            else:
                # We want a frame step index corresponding to Observation window
                totalSamples = inputSizes[0] * featureDict['FrameSize']
                # Number of frames that make up an observation
                self.m_observationSize = totalSamples / (DesiGenreDetector.ObserveDurationSec * featureDict['SampleRate'])
                self.m_observationFrames = inputSizes[0]/self.m_observationSize
                logger.info("Observation Frames=%d Observation Windows=%d", self.m_observationSize, self.m_observationFrames)
                self.m_features = featureDict
                # For energy based detection, we use cluster approach and need a map of pitch numbers
                # against energy measured during observation. For pitch numbers we use Midi numbers
                # starting from C1, so fill first column with midi numbers
                self.m_pitchMap = np.zeros((DesiGenreDetector.ClusterGroups, inputSizes[0]))
                self.m_pitchMap[:,0] = np.arange((DesiGenreDetector.MidiC1, DesiGenreDetector.MidiC1 + inputSizes[0]))

                # Finally we make a partial decision per feature per Frame, allocate arrays for storing these
                # partial results
                self.m_energyDecision = np.zeros(self.m_observationFrames)
                self.m_constrastDecision = np.zeros(1, self.m_observationFrames)
                self.m_flatnessDecision = np.zeros(1,self.m_observationFrames)


    def isQawali(self):
        for frameIdx in range(0,self.m_observationFrames):
            startFrame = frameIdx * self.m_observationSize
            endFrame = frameIdx * self.m_observationSize
            # Cluster pitch energy into groups, input is a 2-D feature set with first column
            # as midi number and second column containing pitch energy
            self.m_pitchMap[:,1] = np.linalg.norm(m_feature['PitchEnergy'][:,startFrame, endFrame], axis=1)
            [maxEnergy, maxEnegyPitch] = [np.max(self.m_pitchMap), np.argmax(self.m_pitchMap)]
            logger.debug("Frame=%d, maximum energy=%6.4f, corresponding pitch=%d", frameIdx, maxEnergy, maxEnergyPitch)
            # Our intention here simply is to categorize into low and high energy groups and then see if there is
            # a max energy cluster around midi 40-44
            energyClusters = KMeans(n_clusters=DesiGenreDetector.ClusterGroups, random_state=0)
            energyClusters.fit(energyPerPitch)
            # Locate cluster with maximum energy
            maxEnegyCluster = np.where(energyClusters.cluster_centers_ == np.amax(energyClusters.cluster_centers_))

            # we expect a tuple with a single 2-D array containing pitch corresponding to maximum energy
            if (len(maxEnergyCluster) != 1) and maxEnergyCluster[0] != 2:
                logger.error("Unexpected size=%d of maximum energy cluster", len(maxEnergyCluster))
            else:
                # Does maximum energy cluster include interesting midi range E2-A2
                if maxEnegyCluster[0][0] > DesiGenreDetector.MidiE2 and maxEnegyCluster[0][0] < DesiGenreDetector.MidiA2:
                    logger.info("Maximum energy pitch in frame index=%d clustered around=%6.4f lies between E2-A2", frameIdx, maxEnegyCluster[0][0])
                    self.m_energyDecision[frameIdx] = 1.0
                else:
                    logger.info("Maximum energy pitch clustered around=%6.4f outside E2-A2", maxEnegyCluster[0][0])

            # TODO: Detection based on other features

        if (consecutive_detected_frames(self.m_energyDecision, 1.0) > self.m_observationFrames/3):
            logger.info("More than one-third of observed frames are consecutively labelled positive")
            return True
        return False