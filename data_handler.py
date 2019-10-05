import numpy as np
import sys
import os
from scipy import misc

class DataHandler:
    def __init__(self, dataVersion, dataRoot, utilRoot, numTrImg, numValImg, numValLoc, batchSize, patchSize, dispRange):
        if dataVersion == 'kitti2012':
            self.numChannels = 1
        elif dataVersion == 'kitti2015':
            self.numChannels = 3
        else:
            sys.exit('DataVersion incorreto')

        self.batchSize = batchSize
        self.patchSize = patchSize
        self.dispRange = dispRange
        self.halfPatch = patchSize // 2
        self.halfRange = dispRange // 2

        self.trPtr = 0
        self.currEpoch = 0

        self.fileIds = np.fromfile(os.path.join(utilRoot, 'myPerm.bin'), '<f4')
        self.trLoc = np.fromfile(('%s/tr_%d_%d_%d.bin')%(utilRoot, numTrImg, self.halfPatch, self.halfRange), '<f4').reshape(-1, 5).astype(int)

        if numValImg == 0:
            self.valLoc = self.trLoc
            print('Validando no treino')
        else:
            self.valLoc = np.fromfile(('%s/val_%d_%d_%d.bin')%(utilRoot, numValImg, self.halfPatch, self.halfRange), '<f4').reshape(-1, 5).astype(int)

        print(('training locations: %d -- valuation locations: %d')%(self.trLoc.shape[0], self.valLoc.shape[0]))

        for i in range(2, 5):
            self.trLoc[:, i] -= 1
            self.valLoc[:, i] -= 1

        self.lData = {}
        self.rData = {}

        self.trPerm = np.arange(self.trLoc.shape[0])
        self.valPerm = np.arange(self.valLoc.shape[0])
        np.random.shuffle(self.trPerm)
        np.random.shuffle(self.valPerm)

        for i in range(numTrImg+numValImg):
            fn = self.fileIds[i]
            if dataVersion == 'kitti2015':
                lImg = misc.imread(('%s/image_2/%06d_10.png')%(dataRoot, fn))
                rImg = misc.imread(('%s/image_3/%06d_10.png')%(dataRoot, fn))
            elif dataVersion == 'kitti2012':
                lImg = misc.imread(('%s/image_0/%06d_10.png')%(dataRoot, fn))
                rImg = misc.imread(('%s/image_1/%06d_10.png')%(dataRoot, fn))

            lImg = (lImg - lImg.mean())/lImg.std()
            rImg = (rImg - rImg.mean())/rImg.std()

            self.lData[fn] = lImg.reshape(lImg.shape[0], lImg.shape[1], self.numChannels)
            self.rData[fn] = rImg.reshape(rImg.shape[0], rImg.shape[1], self.numChannels)

        self.batchLeft = np.zeros((self.batchSize, self.patchSize, self.patchSize, self.numChannels))
        self.batchRight = np.zeros((self.batchSize, self.patchSize, self.patchSize+self.dispRange-1, self.numChannels))
        self.batchLabel = np.zeros((self.batchSize, dispRange))
        dist = [0.05, 0.2, 0.5, 0.2, 0.05]
        halfDist = len(dist)//2
        count = 0
        for i in range(dispRange //2 - halfDist, dispRange //2 + halfDist + 1):
            self.batchLabel[:, i] = dist[count]
            count += 1

        self.valLeft = np.zeros((numValLoc, self.patchSize, self.patchSize, self.numChannels))
        self.valRight = np.zeros((numValLoc, self.patchSize, self.patchSize+self.dispRange-1, self.numChannels))
        self.valLabel = np.zeros((numValLoc, dispRange))
        count = 0
        for i in range(dispRange //2 - halfDist, dispRange //2 + halfDist + 1):
            self.valLabel[:, i] = dist[count]
            count += 1

        for i in range(numValLoc):
            imgId, locType, centerX, centerY, rightCenterX = self.valLoc[self.valPerm[i], 0], self.valLoc[self.valPerm[i], 1], self.valLoc[self.valPerm[i], 2], self.valLoc[self.valPerm[i], 3], self.valLoc[self.valPerm[i], 4]
            rightCenterY = centerY
#            print('%d e %d'%(centerY-self.halfPatch, centerY+self.halfPatch+1))
            self.valLeft[i] = self.lData[imgId][(centerY-self.halfPatch):(centerY+self.halfPatch+1), (centerX-self.halfPatch):(centerX+self.halfPatch+1), :]
            if locType == 1:
                self.valRight[i] = self.rData[imgId][rightCenterY-self.halfPatch : rightCenterY+self.halfPatch+1, rightCenterX-self.halfPatch-self.halfRange : rightCenterX+self.halfPatch+self.halfRange+1, :]
            elif locType == 2:
                self.valRight[i] = np.tranpose(self.rData[imgId][rightCenterY-self.halfPatch-self.halfRange : rightCenterY+self.halfPatch+self.halfRange+1, rightCenterX-self.halfPatch : rightCenterX+self.halfPatch+1, :], (1, 0, 2))

        print('Validation created: num(%d)'% numValLoc)

    def NextBatch(self):
        for idx in range(self.batchSize):
            i = self.trPtr + (idx+1)
            if i > self.trPerm.shape[0]:
                i = 1
                self.trPrt = -(idx+1)+1 # me parece besta mas é so -idx
                self.currEpoch += 1
                print('....epoch id: '+self.currEpoch+' done....\n')
            i -= 1

            imgId, locType, centerX, centerY, rightCenterX = self.trLoc[self.trPerm[i], 0], self.trLoc[self.trPerm[i], 1], self.trLoc[self.trPerm[i], 2], self.trLoc[self.trPerm[i], 3], self.trLoc[self.trPerm[i], 4]
            rightCenterY = centerY

            if locType == 1:
                self.batchLeft[idx] = self.lData[imgId][(centerY-self.halfPatch):(centerY+self.halfPatch+1), (centerX-self.halfPatch):(centerX+self.halfPatch+1), :]
                self.batchRight[idx] = self.rData[imgId][rightCenterY-self.halfPatch : rightCenterY+self.halfPatch+1, rightCenterX-self.halfPatch-self.halfRange : rightCenterX+self.halfPatch+self.halfRange+1, :]
            elif locType == 2:
                self.batchLeft[idx] = np.transpose(self.lData[imgId][(centerY-self.halfPatch):(centerY+self.halfPatch+1), (centerX-self.halfPatch):(centerX+self.halfPatch+1), :], (1, 0, 2))
                self.batchRight[idx] = np.transpose(self.rData[imgId][rigthCenterY-self.halfPatch : rightCenterY+self.halfPatch+1, rightCenterX-self.halfPatch-self.halfRange : rightCenterX+self.halfPatch+self.halfRange+1, :], (1, 0, 2))

        self.trPtr = self.trPtr + self.batchSize
        return self.batchLeft, self.batchRight, self.batchLabel

    def Evaluate(self):
        return self.valLeft, self.valRight, self.valLabel

if __name__ == '__main__':
    dh = DataHandler(dataVersion='kitti2015', dataRoot=os.path.expanduser('~/JP/training'), utilRoot=os.path.expanduser('~/JP/cvpr16_stereo_public/preprocess/debug_15'), numTrImg=160, numValImg=40, numValLoc=100, batchSize=128, patchSize=37, dispRange=201) #patchSize=18 dispRage=100 numValImg=40
    bLeft, bRight, bLabels = dh.NextBatch()
    print(bLeft.shape)
    print(bLabels[:5])
