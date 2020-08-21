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

class QawaliNet(torch.nn.Module):
    Layer1Out = 32
    Layer2Out = 8
    def __init__(self, numFeatures):
        super(QawaliNet, self).__init__()
        self.m_layer1 = torch.nn.Linear(numFeatures, QawaliNet.Layer1Out)
        self.m_active = torch.nn.ReLU()
        self.m_layer2 = torch.nn.Linear(QawaliNet.Layer1Out, QawaliNet.Layer2Out)
        self.m_layer3 = torch.nn.Linear(QawaliNet.Layer2Out, 2)
        self.m_sig = torch.nn.Sigmoid()

    def forward(self, xTensor):
        l1 = self.m_layer1(xTensor)
        l1Out = self.m_active(l1)
        l2 = self.m_layer2(l1Out)
        l2Out = self.m_active(l2)
        l3  = self.m_layer3(l2Out)
        yPredicted = l3

        return yPredicted


class QawaliClassifier:
    TrainDataFile = 'training.dat'
    def __init__(self, batchSize, numFeatures):
        self.m_model = QawaliNet(numFeatures)
        self.m_criterion = torch.nn.CrossEntropyLoss(weight= torch.Tensor([1.0, 0.1]), reduction='mean')
        self.m_optimizer = torch.optim.Adam(self.m_model.parameters(), lr=1e-2)
        self.m_batch = batchSize
        self.m_N = numFeatures
        self.m_trainX = torch.empty([self.m_batch, 1, numFeatures], dtype=torch.float32)
        self.m_trainY = torch.empty([self.m_batch, 1], dtype=torch.float32)
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
        # This is the first batch of data being stored
        if self.m_trainX.size()[1] == 1:
            self.m_trainX = combinedNorm.reshape(1, timeAxisSize, self.m_N)
        else:
            self.m_trainX = torch.cat((self.m_trainX, combinedNorm.reshape(1,timeAxisSize,self.m_N)), dim=0)
        if genre == 'Q':
            self.m_trainY[self.m_loadCursor] = torch.Tensor([2.0])
        else:
            self.m_trainY[self.m_loadCursor] = torch.Tensor([1.0])
        self.m_loadCursor = self.m_loadCursor + 1

    def save_and_plot(self):
        fig = plt.figure(figsize=(10,6))
        logger.info("Saving training data to %s", QawaliClassifier.TrainDataFile)
        plt.title('Tensor view: Training data')
        with open(QawaliClassifier.TrainDataFile, 'wb+') as trainFile:
            torch.save({"train-in": self.m_trainX, "train-out": self.m_trainY}, trainFile)
        timeFoldingSize = self.m_trainX.size()[1]
        plt.imshow(self.m_trainX.reshape(timeFoldingSize, self.m_batch * self.m_N).numpy())
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
       logger.info("Starting training!")
       self.m_trainX = self.m_trainX.float()
       timeSamples = self.m_trainX.size()[1]
       # To avoid later automatic broadcasting, in case of using MSE Loss
       #self.m_trainY = torch.ones([timeSamples, 1], dtype=torch.float32) * self.m_trainY.reshape(1, self.m_batch)
       for trainIdx in range(self.m_batch):
            yHat = self.m_model(self.m_trainX[trainIdx,:,:])

            # Batch to class indices in case of using CrossEntropy loss function
            if (self.m_trainY[trainIdx].item() == 2.0):
                logger.info("Target class: Qawali!")
                yTarget = torch.zeros(timeSamples).long()
            else:
                logger.info("Target class: Not-A-Qawali!")
                yTarget = torch.ones(timeSamples).long()

            # Uncomment below formulation of target output when using MSE loss function
            #loss = self.m_criterion(yHat, self.m_trainY[:,trainIdx].reshape(timeSamples,1).float())
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
        outputPrediction = self.m_model(normalizedEval.reshape(1,timeAxisSize, self.m_N))

        print(outputPrediction)
        #if genre == 'Q' and torch.eq(outputPrediction, torch.zeros(timeAxisSize).long()):
        #    logger.info("Qawali Matched!")
        #    return True
        #elif genre != 'Q' and tensor.eq(outputPrediction, torch.zeros(timeAxisSize).long()):
        #    logger.info("Non Qawali predicted")
        #    return True
        #else:
        #    logger.error("Genre mismatch from Qawali learner")
        #    return False
