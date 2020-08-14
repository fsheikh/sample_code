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
    Layer1Out = 64
    Layer2Out = 16
    def __init__(self, numFeatures):
        super(QawaliNet, self).__init__()
        self.m_layer1 = torch.nn.Linear(numFeatures, QawaliNet.Layer1Out)
        self.m_active = torch.nn.ReLU()
        self.m_layer2 = torch.nn.Linear(QawaliNet.Layer1Out, QawaliNet.Layer2Out)
        self.m_layer3 = torch.nn.Linear(QawaliNet.Layer2Out, 1)
        self.m_sig = torch.nn.Sigmoid()

    def forward(self, xTensor):
        l1 = self.m_layer1(xTensor)
        l1Out = self.m_active(l1)
        l2 = self.m_layer2(l1Out)
        l2Out = self.m_active(l2)
        l3  = self.m_layer3(l2Out)
        yPredicted = self.m_sig(l3)

        return yPredicted


class QawaliClassifier:
    def __init__(self, numFeatures):
        self.m_model = QawaliNet(numFeatures)
        self.m_criterion = torch.nn.MSELoss(reduction='sum')
        self.m_optimizer = torch.optim.SGD(self.m_model.parameters(), lr=1e-2)
        self.m_N = numFeatures
        self.m_trainX = torch.tensor([], dtype=torch.float)
        self.m_trainY = torch.tensor([], dtype=torch.float)
    # Given a dictionary of features, converts them into
    # tensors for later training/evaluation
    def load(self, featureDict, genre):
        pe = torch.from_numpy(featureDict['PitchEnergy'])
        sc = torch.from_numpy(featureDict['SpectralContrast'])

        combined = torch.cat((pe, sc), dim=0)
        if (combined.size()[0] != self.m_N):
            logger.error("Unexpected number of features=%d", combined.size()[0])
            raise RuntimeError

        combinedNorm = abs(combined - combined.mean() / combined.std())
        if self.m_trainX.size()[0] == 0:
            self.m_trainX = combinedNorm.reshape(1, self.m_N)
        else:
            self.m_trainX = torch.cat((self.m_trainX, combinedNorm.reshape(1,self.m_N)), dim=0)
        if genre == 'Q':
            self.m_trainY = torch.cat((self.m_trainY, torch.Tensor([1.0])), dim=0)
        else:
            self.m_trainY = torch.cat((self.m_trainY, torch.Tensor([0.0])), dim=0)

    def display(self):
        fig = plt.figure(figsize=(6,4))
        plt.title('Tensor view: Training data')
        plt.imshow(self.m_trainX.numpy())
        plt.colorbar()
        fig.savefig('tensor-training.png')
        plt.close()
        logger.info("Saved training data tensor")

    # Given input feature tensor x, trains the model
    # to learn it as y (float value)
    def train(self):
       logger.info("Starting training!")
       for trainIdx in range(self.m_N):
           yHat = self.m_model(self.m_trainX[trainIdx,:].reshape(1,self.m_N).float())

           loss = self.m_criterion(yHat, self.m_trainY[trainIdx])
           logger.info("Index=%d Loss=%6.4f", trainIdx, loss.item())

           self.m_optimizer.zero_grad()
           loss.backward()
           self.m_optimizer.step()

    # Given input feature tensor x, retuns true if
    # output of model matched tensor y, otherwise false
    def classify(self, xTensor, yTensor):
        logger.info("To be implemented")