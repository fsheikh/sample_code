# Neural net based supervised classifier for Qawali/songs

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

# Module implementing a pytorch neural network to train and binary classify
# Qawali songs.
from enum import Enum
import numpy as np
import logging
import torch
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Classification labels, only attempt to classify Qawali
# genre from rest of the world
class Labels(Enum):
    Qawali = 1
    NonQawali = 10

# Mode of feature loading, we can be loading training/validation/evaluation
# input features and corresponding output expected vectors
class LoadMode(Enum):
    TI = 'train-in'
    TO = 'train-out'
    EI = 'eval-in'
    EO = 'eval-out'
    VI = 'validate-in'
    VO = 'validate-out'

# Simple linear network to see if a state-less classifer
# works for genre detection, answer is no, now here for posterity
class QawaliNet(torch.nn.Module):
    Layer1Out = 32
    Layer2Out = 16
    Layer3Out = 2
    def __init__(self, numFeatures):
        super(QawaliNet, self).__init__()
        self.m_layer1 = torch.nn.Linear(numFeatures, QawaliNet.Layer1Out)
        self.m_active = torch.nn.ReLU()
        self.m_layer2 = torch.nn.Linear(QawaliNet.Layer1Out, QawaliNet.Layer2Out)
        self.m_layer3 = torch.nn.Linear(QawaliNet.Layer2Out, QawaliNet.Layer3Out)

    def forward(self, xTensor):
        l1 = self.m_layer1(xTensor)
        l1Out = self.m_active(l1)
        l2 = self.m_layer2(l1Out)
        l2Out = self.m_active(l2)
        l3  = self.m_layer3(l2Out)
        l3Out = self.m_active(l3)
        # If input dimensions are B x M x N (where N is size of feature vector)
        # l3Out has shape B x M x 2, we are only interested in what comes of
        # the last dimension since that maps on the two genre we are interested in
        yPredicted = torch.nn.functional.log_softmax(l3Out[-1], dim=1)

        return yPredicted

# Simple one-layer LSTM to apply on time-sequence
# of audio features suitable for Qawali recognition
class QawaliLSTM(torch.nn.Module):
    HiddenDim = 128
    OutDim = 2 # only interested in classiying two caterogies, Qawali and the rest
    def __init__(self, numFeatures):
        super(QawaliLSTM, self).__init__()

        self.m_lstm = torch.nn.LSTM(numFeatures, QawaliLSTM.HiddenDim,2)
        self.m_linear = torch.nn.Linear(QawaliLSTM.HiddenDim, QawaliLSTM.OutDim)

    def forward(self, input):
        lstmOut, hiddenOut = self.m_lstm(input)
        lstmAffine = self.m_linear(lstmOut[-1])
        estimatedGenres = torch.nn.functional.log_softmax(lstmAffine, dim=1)
        return estimatedGenres

class QawaliClassifier:
    FeatureDataFile = 'features.dat'
    def __init__(self, trainingSize, evalSize, numFeatures):
        self.m_model = QawaliLSTM(numFeatures)
        self.m_T = trainingSize
        self.m_E = evalSize
        #self.m_criterion = torch.nn.NLLLoss(torch.Tensor([1.0, 0.083]), reduction='mean')
        self.m_criterion = torch.nn.NLLLoss(reduction='mean')
        self.m_optimizer = torch.optim.Adam(self.m_model.parameters(), lr=1e-2)
        self.m_N = numFeatures
        self.m_dataMap = {}
        self.m_dataMap[LoadMode.TI] = torch.empty([self.m_T, 1, self.m_N], dtype=torch.float32)
        self.m_dataMap[LoadMode.EI] = torch.empty([self.m_E, 1, self.m_N], dtype=torch.float32)
        # Attempting binary classification, construction is in the form of one-hot vector
        # First column represents interesting label (Qawali), second column other genre
        self.m_dataMap[LoadMode.TO] = torch.zeros([self.m_T, 2], dtype=torch.long)
        self.m_dataMap[LoadMode.EO] = torch.zeros([self.m_T, 2], dtype=torch.long)
        self.m_loadCursor = 0

    # Given a dictionary of features per song, converts them into
    # tensors for later training/validation/evaluation
    def load(self, featureDict, genre, mode=LoadMode.TI):

        if mode != LoadMode.TI and mode != LoadMode.EI:
            logger.error("Only training/evaluation input modes are supported, outputs are deduced from inputs")
            raise ValueError

        pe = torch.from_numpy(featureDict['PitchEnergy'])
        sc = torch.from_numpy(featureDict['SpectralContrast'])

        combined = torch.cat((pe, sc), dim=0)
        timeAxisSize = combined.size()[1]
        logger.info("Features=%d x Time instances=%d", combined.size()[0], combined.size()[1])
        if (combined.size()[0] != self.m_N):
            logger.error("Unexpected number of features=%d", combined.size()[0])
            raise RuntimeError
        # Normalize/pre-processing
        combinedNorm = abs(combined - combined.mean() / combined.std())

        # This is the first set of feature vectors (features for first training/eval sample)
        # Reason for re-assignment instead of just assigning is to do a check on dimensions
        if self.m_dataMap[mode].size()[1] == 1:
            self.m_dataMap[mode] = combinedNorm.reshape(1, timeAxisSize, self.m_N)
        else:
            self.m_dataMap[mode] = torch.cat((self.m_dataMap[mode], combinedNorm.reshape(1,timeAxisSize,self.m_N)), dim=0)

        # Output is a 2D array where first column indicates index of song within training/evaluation data
        # Second column is the class label, since we have only two classes we can use 0/1.
        # This means output tensor will have 1 in the 2nd column only if the song represented by first column
        # is a Qawali. Output Mode is to index into the data map, if we are loading evaluation features, corresponding
        # output vector should be filled, we already know these are the only two possible combination for now
        label = Labels.Qawali if (genre == 'Q') else Labels.NonQawali
        outMode = LoadMode.EO if (mode == LoadMode.EI) else LoadMode.TO
        self.m_dataMap[outMode][self.m_loadCursor, 0] = self.m_loadCursor
        self.m_dataMap[outMode][self.m_loadCursor, 1] = label.value

        self.m_loadCursor = self.m_loadCursor + 1

    def save_and_plot(self):
        logger.info("Saving loaded feature data to %s", QawaliClassifier.FeatureDataFile)
        with open(QawaliClassifier.FeatureDataFile, 'wb+') as dataFile:
            torch.save(self.m_dataMap, dataFile)

        # Plot feature vectors for input training sample
        timeFoldingSize = self.m_dataMap[LoadMode.TI].size()[1]
        fig = plt.figure(figsize=(10,6))
        plt.title('Feature view as tensors')
        plt.imshow(self.m_dataMap[LoadMode.TI].reshape(timeFoldingSize, self.m_T * self.m_N).numpy())
        plt.colorbar()
        fig.savefig('feature-data.png')
        plt.close()
        logger.info("Saved training data tensor")

    def reload_from_disk(self):
        logger.info("Loading feature data from %s", QawaliClassifier.FeatureDataFile)
        with open(QawaliClassifier.FeatureDataFile, 'rb') as f:
            self.m_dataMap = torch.load(f)

    # Given already loaded training data, uses it to learn Y (genre)
    # given feature vector X for each song.
    def train(self):
       self.m_model.train()
       self.m_model.hidden = torch.zeros(1,1,QawaliLSTM.HiddenDim)
       logger.info("Starting training!")
       trainX = self.m_dataMap[LoadMode.TI].float()
       trainY = self.m_dataMap[LoadMode.TO].float()
       # In order to process in batches of size M, one can divide the range parameter by M
       # As long as the training data-set is not huge, training with a single batch is okay
       for trainIdx in range(int(self.m_T)):

            # For batch process change the scalar multiplier accordingly
            m = 1
            xSlice = trainX[m * trainIdx: m * (trainIdx + 1),:,:]
            # Got a slice of training data with dimensions m x Time x Features
            yHat = self.m_model(xSlice.permute(1,0,2))

            # Expected output is in terms of class indices
            yTarget = torch.max(trainY[m * trainIdx: m * (trainIdx+1),:],1)[1]
            # Below formulation is for target output when using MSE loss function
            #loss = self.m_criterion(yHat, expY[:,trainIdx].reshape(timeSamples,1).float())
            loss = self.m_criterion(yHat, yTarget)
            logger.info("Index=%d Loss=%6.4f", trainIdx, loss.item())

            self.m_optimizer.zero_grad()
            loss.backward()
            self.m_optimizer.step()

    # Given input features map, uses the trained model to predict
    # the genre, returns true/false if result of prediction matches
    # given ground-truth
    def classify(self, inFeatures, genre):
        self.m_model.eval()
        logger.info("Started evaluation mode!")
        evalX = self.m_dataMap[LoadMode.EI].float()
        evalY = self.m_dataMap[LoadMode.EO].float()

        for evalIdx in range(int(self.m_E)):
            # single batch processing
            m = 1
            xSlice = evalX[m * evalIdx : m * (evalIdx + 1),:,:]
            outputPrediction = self.m_model(xSlice.permute(1,0,2))
            logger.info("Song evaluated at index=%d", evalIdx)
            print(outputPrediction)
            logger.info("\n****\n")
        #if genre == 'Q' and torch.eq(outputPrediction, torch.zeros(timeAxisSize).long()):
        #    logger.info("Qawali Matched!")
        #    return True
        #elif genre != 'Q' and tensor.eq(outputPrediction, torch.zeros(timeAxisSize).long()):
        #    logger.info("Non Qawali predicted")
        #    return True
        #else:
        #    logger.error("Genre mismatch from Qawali learner")
        #    return False

