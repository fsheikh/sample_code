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

import numpy as np
import logging
import torch
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


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
    TrainDataFile = 'training.dat'
    def __init__(self, trainingSize, numFeatures):
        self.m_model = QawaliLSTM(numFeatures)
        self.m_T = trainingSize
        self.m_criterion = torch.nn.NLLLoss(torch.Tensor([1.0, 0.083]), reduction='mean')
        self.m_optimizer = torch.optim.Adam(self.m_model.parameters(), lr=1e-2)
        self.m_N = numFeatures
        self.m_trainX = torch.empty([self.m_T, 1, self.m_N], dtype=torch.float32)
        # Attempting binary classification, construction is in the form of one-hot vector
        # First column represents interesting label (Qawali), second column other genre
        self.m_trainY = torch.zeros([self.m_T, 2], dtype=torch.long)
        self.m_loadCursor = 0
    # Given a dictionary of features, converts them into
    # tensors for later training/evaluation
    def load(self, featureDict, genre):
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
        # This is the first set of feature vectors (first training sample)
        if self.m_trainX.size()[1] == 1:
            self.m_trainX = combinedNorm.reshape(1, timeAxisSize, self.m_N)
        else:
            self.m_trainX = torch.cat((self.m_trainX, combinedNorm.reshape(1,timeAxisSize,self.m_N)), dim=0)
        if genre == 'Q':
            self.m_trainY[self.m_loadCursor, 0] = 1
        else:
            self.m_trainY[self.m_loadCursor,1] = 1
        self.m_loadCursor = self.m_loadCursor + 1

    def save_and_plot(self):
        fig = plt.figure(figsize=(10,6))
        logger.info("Saving training data to %s", QawaliClassifier.TrainDataFile)
        plt.title('Tensor view: Training data')
        with open(QawaliClassifier.TrainDataFile, 'wb+') as trainFile:
            torch.save({"train-in": self.m_trainX, "train-out": self.m_trainY}, trainFile)
        timeFoldingSize = self.m_trainX.size()[1]
        plt.imshow(self.m_trainX.reshape(timeFoldingSize, self.m_T * self.m_N).numpy())
        plt.colorbar()
        fig.savefig('tensor-training.png')
        plt.close()
        logger.info("Saved training data tensor")

    def reload_from_disk(self):
        logger.info("Loading training data from %s", QawaliClassifier.TrainDataFile)
        with open(QawaliClassifier.TrainDataFile, 'rb') as trainFile:
            fullData = torch.load(trainFile)
            self.m_trainX = fullData["train-in"]
            self.m_trainY = fullData["train-out"]

    # Given input feature tensor x, trains the model
    # to learn it as y (float value)
    def train(self):
       self.m_model.train()
       self.m_model.hidden = torch.zeros(1,1,QawaliLSTM.HiddenDim)
       logger.info("Starting training!")
       # We expect a double tensor of size TrainingSize x Time x FeatureSize
       self.m_trainX = self.m_trainX.float()
       # To avoid later automatic broadcasting, in case of using MSE Loss
       #timeSamples = self.m_trainX.size()[1]
       #expY = torch.ones([timeSamples, 1], dtype=torch.float32) * self.m_trainY.reshape(1, self.m_batch)
       # We process in batches of two
       for trainIdx in range(int(self.m_T)):

            self.m_model.zero_grad()
            # Process two training samples in one batch
            xSlice = self.m_trainX[1 * trainIdx: 1 * (trainIdx + 1),:,:]
            # Got a slice of training data with dimensions 2 x Time x Features
            yHat = self.m_model(xSlice.permute(1,0,2))

            # Batch to class indices in case of using CrossEntropy loss function
            yTarget = torch.max(self.m_trainY[1 * trainIdx: 1 * (trainIdx+1),:],1)[1]
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

        # Make tensor out of features
        pe = torch.from_numpy(inFeatures['PitchEnergy'])
        sc = torch.from_numpy(inFeatures['SpectralContrast'])

        combined = torch.cat((pe, sc), dim=0)
        timeAxisSize = combined.size()[1]
        logger.info("Features=%d x Time instances=%d", combined.size()[0], combined.size()[1])
        if (combined.size()[0] != self.m_N):
            logger.error("Unexpected number of features=%d", combined.size()[0])
            raise RuntimeError

        normalizedEval = abs(combined - combined.mean() / combined.std())
        normalizedEval = normalizedEval.float()
        outputPrediction = self.m_model(normalizedEval.float().reshape(timeAxisSize, 1, self.m_N))

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
