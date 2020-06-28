
# Module containing class whose objects are initialized with
# a suitable feature set extracted from a song, and then supply
# dectection methods utilizing either heurisitcs or machine learning to
# classify the song as a South Asian genre like Qawali, Ghazal,
# geet, kafi, thumri, etc...

from itertools import dropwhile
from itertools import groupby
from sklearn.cluster import KMeans
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Function takes a 1-D numpy array as input and looks for the longest
# sequence of given element, returns a tuple with positive and consecutive
# positive elements
def pac_elements(framesArray, elemToMatch):
    retTuple = (0,0)
    # only works for 1-D array
    if (framesArray.size != framesArray.shape[-1]):
        logger.error("CDF:Input not a 1D array? with size=%d", framesArray.size)
        return retTuple

    if not np.any(framesArray == elemToMatch ):
        logger.warning("CDF:No matching element found")
        return retTuple

    groups = []
    for k,g in groupby(framesArray):
        groups.append(list(g))
    print(groups)
    elemNoMatchKey = lambda x : x != elemToMatch
    retTuple = (np.count_nonzero(framesArray == elemToMatch), max([sum(1 for i in dropwhile(elemNoMatchKey, g)) for k,g in groupby(framesArray)]))
    logger.info("CDF:Positive =%d, Longest sequence=%d of match=%6.4f", retTuple[0], retTuple[1], elemToMatch)
    return retTuple


class DesiGenreDetector:
    # Name of features expected by classifier
    # This information is determined by studying various feature-sets
    # This modules assumes that a feature set has been agreed upon
    # with user
    ExpectedFeatures = ['FrameSize', 'SampleRate', 'PitchEnergy', 'SpectralContrast', 'SpectralFlatness']
    ObserveDurationSec = 2
    ClusterGroups = 3
    MidiC1 = 24
    MidiC2 = 36.0
    MidiE2 = 40.0
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
                self.m_observationSize = (DesiGenreDetector.ObserveDurationSec * featureDict['SampleRate']) // featureDict['FrameSize']
                self.m_observationFrames = inputSizes[0] // self.m_observationSize
                logger.info("total samples=%d Observation windows size=%d Observation frames=%d", totalSamples, self.m_observationSize, self.m_observationFrames)
                self.m_features = featureDict
                # For energy based detection, we use cluster approach and need a map of pitch numbers
                # against energy measured during observation. For pitch numbers we use Midi numbers
                # starting from C1 and relying on frequency bins used by the feature extractor
                self.m_pitchMap = np.zeros((featureDict['PitchEnergy'].shape[0], DesiGenreDetector.ClusterGroups))
                self.m_pitchMap[:,0] = np.arange(DesiGenreDetector.MidiC1, DesiGenreDetector.MidiC1 + featureDict['PitchEnergy'].shape[0])

                # Finally we make a partial decision per feature per Frame, allocate arrays for storing these
                # partial results
                self.m_energyDecision = np.zeros(self.m_observationFrames)
                self.m_constrastDecision = np.zeros(self.m_observationFrames)
                self.m_flatnessDecision = np.zeros(self.m_observationFrames)
                self.m_overallDecision = np.zeros(self.m_observationFrames)


    def isQawali(self):
        for frameIdx in range(0, self.m_observationFrames):
            startFrame = frameIdx * self.m_observationSize
            endFrame = frameIdx * self.m_observationSize + self.m_observationSize
            # Cluster pitch energy into groups, input is a 2-D feature set with first column
            # as midi number and second column containing pitch energy
            self.m_pitchMap[:,1] = np.linalg.norm(self.m_features['PitchEnergy'][:,startFrame:endFrame], axis=1)
            [maxEnergy, maxEnergyPitch] = [np.max(self.m_pitchMap[:,1]), np.argmax(self.m_pitchMap[:,1])]
            #logger.info("Frame=%d, maximum energy=%6.4f, corresponding pitch=%d", frameIdx, maxEnergy, maxEnergyPitch + DesiGenreDetector.MidiC1)
            # Our intention here simply is to categorize into low and high energy groups and then see if there is
            # a max energy cluster around midi 40-44
            energyClusters = KMeans(n_clusters=DesiGenreDetector.ClusterGroups, random_state=0)
            energyClusters.fit(self.m_pitchMap)
            # Cluster centers is 2-D array with first column coming from input i.e. pitchMap
            # Second column is energy clusters, sort the array along this axis and pick the last
            # row corresponding to maximum energy
            sortedEnergyClusters = energyClusters.cluster_centers_[np.argsort(energyClusters.cluster_centers_[:, 1])]
            maxEnergyCluster = sortedEnergyClusters[DesiGenreDetector.ClusterGroups - 1]
            # Sanity check on extracted max energy cluster
            if (maxEnergyCluster.size != DesiGenreDetector.ClusterGroups):
                logger.error("Unexpected size=%d of maximum energy cluster", maxEnergyCluster.size)
            else:
                logger.info("max energy cluster=(%6.4f, %6.4f)", maxEnergyCluster[0], maxEnergyCluster[1])
                # Does maximum energy cluster include interesting midi range C2-E2
                if maxEnergyCluster[0] in np.arange(DesiGenreDetector.MidiC2, DesiGenreDetector.MidiE2, 0.5):
                    #logger.info("Maximum energy pitch in frame index=%d clustered around=%6.4f between C2-E2", frameIdx, maxEnergyCluster[1])
                    self.m_energyDecision[frameIdx] = 1.0
                else:
                    logger.info("Maximum energy pitch=%6.4f clustered around=%6.4f outside F2-A2", maxEnergyCluster[0], maxEnergyCluster[1])

            # Detection based on spectral contrast, feature set per observation window is
            # a 2D array with size=(subbands, self.m_observationSize), qawali classification is based
            # on observing maximum average value in highest subbad and minimum in first subband
            # TODO: If this heuristic does not work well, consider KMean or similar as well
            specContrastAverage = np.mean(self.m_features['SpectralContrast'][:,startFrame:endFrame], axis=1)
            specContrastMM = [np.argmin(specContrastAverage), np.argmax(specContrastAverage)]
            if specContrastMM[0] == 1 and specContrastMM[1] == specContrastAverage.shape[-1] - 1:
                self.m_constrastDecision[frameIdx] = 1.0
            else:
                logger.warning("Spectral contrast min-subband=%d max-subband=%d", specContrastMM[0], specContrastMM[1])

            # Detection based on spectral flatness
            flatnessAverage = np.mean(10* np.log10(self.m_features['SpectralFlatness'][:,startFrame:endFrame]))

            if int(flatnessAverage) in np.arange(-17,-21,-1):
                self.m_flatnessDecision[frameIdx] = 1.0
            else:
                logger.info("Mean flatness=%6.4f outside expected range", flatnessAverage)

            # overall is a majority decision
            self.m_overallDecision[frameIdx] = self.m_flatnessDecision[frameIdx] + self.m_constrastDecision[frameIdx] + self.m_energyDecision[frameIdx]
            if self.m_overallDecision[frameIdx] >= 2.0:
                self.m_overallDecision[frameIdx] = 1.0
            else:
                self.m_overallDecision[frameIdx] = 0.0

        logger.info("Printing pitch-energy decisions")
        print(self.m_energyDecision)
        logger.info("Printing spectral-constrast decisions")
        print(self.m_constrastDecision)
        logger.info("Printing spectral-flatness decisions")
        print(self.m_flatnessDecision)
        logger.info("Printing overall decisions")
        print(self.m_overallDecision)
        # Qawali classification heuristic: Look for at least 30% positive
        # and 15% consecutive positive frames.
        posThreshold = self.m_observationFrames / 3
        conThreshold = self.m_observationFrames / 6
        peDetection = pac_elements(self.m_overallDecision, 1.0)

        if (peDetection[0] > posThreshold and peDetection[1] > conThreshold):
            return True
        else:
            logger.info("Positive detected=%d and consecutive detected=%d do not meet heuristic", peDetection[0], peDetection[1])
        return False