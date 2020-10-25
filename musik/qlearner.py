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
    Qawali = 0
    NonQawali = 1

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
    EnableDebug = False
    def __init__(self, trainingSize, evalSize, numFeatures):
        self.m_model = QawaliLSTM(numFeatures)
        self.m_T = trainingSize
        self.m_E = evalSize
        self.m_criterion = torch.nn.NLLLoss(torch.Tensor([1.0, 0.167]), reduction='mean')
        self.m_optimizer = torch.optim.Adam(self.m_model.parameters(), lr=1e-5)
        self.m_N = numFeatures
        self.m_dataMap = {}
        self.m_dataMap[LoadMode.TI] = torch.empty([1, 1, self.m_N], dtype=torch.float32)
        self.m_dataMap[LoadMode.EI] = torch.empty([1, 1, self.m_N], dtype=torch.float32)
        # Attempting binary classification, construction is in the form of one-hot vector
        # First column represents interesting label (Qawali), second column other genre
        self.m_dataMap[LoadMode.TO] = torch.zeros([self.m_T, 2], dtype=torch.long)
        self.m_dataMap[LoadMode.EO] = torch.zeros([self.m_E, 2], dtype=torch.long)
        self.m_loadCursor = 0

    # Given a dictionary of features per song, converts them into
    # tensors for later training/validation/evaluation
    def load_one(self, featureDict, genre, mode=LoadMode.TI):

        if mode != LoadMode.TI and mode != LoadMode.EI:
            logger.error("Only training/evaluation input modes are supported, outputs are deduced from inputs")
            raise ValueError

        pe = torch.from_numpy(featureDict['PitchEnergy'])
        #sc = torch.from_numpy(featureDict['SpectralContrast'])

        #combined = torch.cat((pe, sc), dim=0)
        combined = pe
        timeAxisSize = combined.size()[1]
        logger.info("Features=%d x Time instances=%d", combined.size()[0], combined.size()[1])
        assert(combined.size()[0] == self.m_N)
        # Normalize/pre-processing
        combinedNorm = abs(combined - combined.mean()) / combined.std()
        # FIXME: There is no real need of reshaping feature vectors, reshape here
        # and then subsequent pemutation change will probably return wrong results
        # Consider toy example below:
        # >>> catData = torch.cat((tData,tData),dim=0)
        #    >>> catData
        #   tensor([[[-3., -7.,  2.],
        #            [-4.,  6.,  9.]],
        #
        #            [[-3., -7.,  2.],
        #            [-4.,  6.,  9.]]])
        #    >>> catData.shape
        #    torch.Size([2, 2, 3])
        #    >>> catNorm=torch.norm(catData[0,:,:],None,1)
        #    >>> catNorm
        #    tensor([ 7.8740, 11.5326])
        #    >>> npNorm
        #    array([ 7.87400787, 11.53256259])
        #    >>> data
        #    array([[-3., -7.,  2.],
        #        [-4.,  6.,  9.]])
        # It might be better to fix the neural network than adapt the input dimensions which
        # made it kind of work but no meaningful results were produced.
        # Reason for re-assignment is that we don't know both dimensions of input feature
        # vector in advance. Convert 2D feature vector to 3D by using the tensor utility function
        if (torch.numel(self.m_dataMap[mode]) == self.m_N):
            self.m_dataMap[mode] = combinedNorm.unsqueeze(0)
        else:
            self.m_dataMap[mode] = torch.cat((self.m_dataMap[mode],combinedNorm.unsqueeze(0)), dim=0)

        # Output is a 2D array of one-hot vectors with first column for Qawali songs
        # and second for everything else, this means that first column will contain
        # a 1.0 value only if the features collected above correspond to a qawali item
        label = Labels.Qawali if (genre == 'Q') else Labels.NonQawali
        outMode = LoadMode.EO if (mode == LoadMode.EI) else LoadMode.TO
        self.m_dataMap[outMode][self.m_loadCursor, label.value] = 1.0

        self.m_loadCursor = self.m_loadCursor + 1

    def load_complete(self, mode):
        logger.info("Loading of one set completed, reset internal load counter")
        self.m_loadCursor = 0
        if mode == LoadMode.TI:
            assert(self.m_dataMap[mode].size()[0] == self.m_T)
        elif mode == LoadMode.EI:
            assert(self.m_dataMap[mode].size()[0] == self.m_E)
        else:
            logger.info("Invalid mode for marking data loading completion %d", mode)

    def save_and_plot(self):
        logger.info("Saving loaded feature data to %s", QawaliClassifier.FeatureDataFile)
        with open(QawaliClassifier.FeatureDataFile, 'wb+') as dataFile:
            torch.save(self.m_dataMap, dataFile)

        # Plot feature vectors for input training sample, we can't plot all
        # 2D feature vectors, to have a sanity check we plot six vectors, three
        # each from training and evaluation sets
        fig = plt.figure(figsize=(10,6))
        plt.title('Feature view as tensors')
        plt.subplot(3,2,1)
        plt.imshow(self.m_dataMap[LoadMode.TI][0,:,:])
        plt.subplot(3,2,2)
        plt.imshow(self.m_dataMap[LoadMode.TI][int(self.m_T - 20),:,:])
        plt.subplot(3,2,3)
        plt.imshow(self.m_dataMap[LoadMode.TI][int(self.m_T-1),:,:])
        plt.subplot(3,2,4)
        plt.imshow(self.m_dataMap[LoadMode.EI][0,:,:])
        plt.subplot(3,2,5)
        plt.imshow(self.m_dataMap[LoadMode.EI][int(self.m_E/2),:,:])
        plt.subplot(3,2,6)
        plt.imshow(self.m_dataMap[LoadMode.EI][int(self.m_E-1),:,:])
        fig.savefig('feature-data.png')
        plt.close()
        logger.info("Saved feature vectors data tensor")

    def reload_from_disk(self):
        logger.info("Loading feature data from %s", QawaliClassifier.FeatureDataFile)
        with open(QawaliClassifier.FeatureDataFile, 'rb') as f:
            self.m_dataMap = torch.load(f)

    # Given already loaded training data, uses it to learn Y (genre)
    # given feature vector X for each song.
    def train(self, batch_size=1):
       self.m_model.train()
       self.m_model.hidden = torch.zeros(1,1,QawaliLSTM.HiddenDim)
       logger.info("Starting training!")
       trainX = self.m_dataMap[LoadMode.TI].float()
       trainY = self.m_dataMap[LoadMode.TO].float()
       # In order to process in batches of size M, one can divide the range parameter by M
       # As long as the training data-set is not huge, training with a single batch is okay
       for trainIdx in range(int(self.m_T/batch_size)):
            # Expected output is in terms of class indices
            yTarget = torch.max(trainY[batch_size * trainIdx: batch_size * (trainIdx+1),:],1)[1]
            print(yTarget)
            if (torch.min(yTarget).item()==0):
                logger.info("Batch contains Qawali")
            else:
                logger.info("Batch does not contain Qawali")

            xSlice = trainX[batch_size * trainIdx: batch_size * (trainIdx + 1),:,:]
            if QawaliClassifier.EnableDebug:
                # Data inspection, mean, variance, l2-norm, along the time-axis
                xSliceMean = torch.mean(xSlice,2)
                xSliceVar = torch.var(xSlice,2)
                xSliceNorm = torch.norm(xSlice,None,2)
                (maxMean, maxMIdx) = torch.max(xSliceMean,1)
                (minMean, minMIdx) = torch.min(xSliceMean,1)
                (maxNorm, maxNIdx) = torch.max(xSliceNorm,1)
                (minNorm, minNIdx) = torch.min(xSliceNorm,1)
                (maxVar, maxVIdx) = torch.max(xSliceVar,1)
                (minVar, minVIdx) = torch.min(xSliceVar,1)
                logger.info("Training--->Mean: max(%6.4f, %6.4f) min(%6.4f, %6.4f) Variance: max(%6.4f, %6.4f) min (%6.4f, %6.4f)\
                            Norm: max(%6.4f, %6.4f) min(%6.4f, %6.4f)", maxMean, maxMIdx, minMean, minMIdx, maxVar, maxVIdx,
                            minVar, minVIdx, maxNorm, maxNIdx, minNorm, minNIdx)
            # Got a slice of training data with dimensions m x Time x Features
            yHat = self.m_model(xSlice.permute(2,0,1))
            logger.info("Estimated output while training")
            print(yHat)

            # Below formulation is for target output when using MSE loss function
            #loss = self.m_criterion(yHat, expY[:,trainIdx].reshape(timeSamples,1).float())
            loss = self.m_criterion(yHat, yTarget)
            logger.info("Index=%d Loss=%6.4f", trainIdx, loss.item())

            self.m_optimizer.zero_grad()
            loss.backward()
            self.m_optimizer.step()
            logger.info("\nTraining iteration=%d ending\n", trainIdx)

    # Given input features map, uses the trained model to predict
    # the genre, returns true/false if result of prediction matches
    # given ground-truth
    def classify(self, batch_size=1):
        self.m_model.eval()
        with torch.no_grad():
            logger.info("Started evaluation mode!")
            evalX = self.m_dataMap[LoadMode.EI].float()
            evalY = self.m_dataMap[LoadMode.EO].float()
            s_count = 0;
            f_count = 0;

            for evalIdx in range(int(self.m_E)):
                self.m_model.hidden = torch.zeros(1,1,QawaliLSTM.HiddenDim)
                songIsQawali = False
                if self.m_dataMap[LoadMode.EO][evalIdx,0] == 1:
                    logger.info("Ground Truth for song[%d] is Qawali", evalIdx)
                    songIsQawali = True
                else:
                    logger.info("Ground Truth for song[%d] is Non-Qawali", evalIdx)

                xSlice = evalX[batch_size * evalIdx : batch_size * (evalIdx + 1),:,:]
                if QawaliClassifier.EnableDebug:
                    xSliceMean = torch.mean(xSlice,2)
                    xSliceVar = torch.var(xSlice,2)
                    xSliceNorm = torch.norm(xSlice,None,2)
                    (maxMean, maxMIdx) = torch.max(xSliceMean,1)
                    (minMean, minMIdx) = torch.min(xSliceMean,1)
                    (maxNorm, maxNIdx) = torch.max(xSliceNorm,1)
                    (minNorm, minNIdx) = torch.min(xSliceNorm,1)
                    (maxVar, maxVIdx) = torch.max(xSliceVar,1)
                    (minVar, minVIdx) = torch.min(xSliceVar,1)
                    logger.info("Evaluation-->Mean: max(%6.4f, %6.4f) min(%6.4f, %6.4f) Variance: max(%6.4f, %6.4f) min (%6.4f, %6.4f)\
                            Norm: max(%6.4f, %6.4f) min(%6.4f, %6.4f)", maxMean, maxMIdx, minMean, minMIdx, maxVar, maxVIdx,
                            minVar, minVIdx, maxNorm, maxNIdx, minNorm, minNIdx)
                outputPrediction = self.m_model(xSlice.permute(2,0,1))
                print(outputPrediction)
                (decision, descIdx) = torch.max(outputPrediction,1)
                if (songIsQawali and descIdx == 0.0):
                    s_count = s_count + 1
                elif (not songIsQawali and descIdx == 1.0):
                    s_count = s_count + 1
                else:
                    f_count = f_count + 1

            logger.info("Success=%d, Failure=%d, Total=%d", s_count, f_count, evalIdx)
