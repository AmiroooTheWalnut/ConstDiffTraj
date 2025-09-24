import numpy as np
import matplotlib.pyplot as plt

class JSD:
    def __init__(self, numGrids, realTraj, generatedTrajs, forbiddenMap, scale=None):
        if scale!=None:
            realTrajGrid = self.trajToGrid_scale(numGrids, realTraj, scale)
        else:
            realTrajGrid = self.trajToGrid(numGrids, realTraj)
        totalReal = np.sum(realTrajGrid)
        if scale != None:
            genTrajGrid = self.trajToGrid_scale(numGrids, generatedTrajs, scale)
        else:
            genTrajGrid = self.trajToGrid(numGrids, generatedTrajs)
        totalGen = np.sum(genTrajGrid)

        #plt.imshow(forbiddenMap, cmap="cool",
        #           extent=(0 - (1 / numGrids) / 2, 1 + (1 / numGrids) / 2, 0 - (1 / numGrids) / 2, 1 + (1 / numGrids) / 2),
        #           origin='lower')

        #plt.imshow(realTrajGrid, cmap="Oranges",
        #           extent=(0 - (1 / numGrids) / 2, 1 + (1 / numGrids) / 2, 0 - (1 / numGrids) / 2, 1 + (1 / numGrids) / 2),
        #           origin='lower',alpha=0.5)
        #plt.title('RealTraj Dist with Forbiddens')
        #plt.show()

        #plt.imshow(forbiddenMap, cmap="cool",
        #           extent=(
        #           0 - (1 / numGrids) / 2, 1 + (1 / numGrids) / 2, 0 - (1 / numGrids) / 2, 1 + (1 / numGrids) / 2),
        #           origin='lower')

        #plt.imshow(genTrajGrid, cmap="GnBu",
        #           extent=(
        #           0 - (1 / numGrids) / 2, 1 + (1 / numGrids) / 2, 0 - (1 / numGrids) / 2, 1 + (1 / numGrids) / 2),
        #           origin='lower',alpha=0.5)
        #plt.title('GenTraj Dist with Forbiddens')
        #plt.show()

        #plt.imshow(genTrajGrid, cmap="GnBu",
        #           extent=(
        #               0 - (1 / numGrids) / 2, 1 + (1 / numGrids) / 2, 0 - (1 / numGrids) / 2, 1 + (1 / numGrids) / 2),
        #           origin='lower', alpha=0.5)
        #plt.title('GenTraj Dist')
        #plt.show()

        #plt.imshow(realTrajGrid, cmap="GnBu",
        #           extent=(
        #               0 - (1 / numGrids) / 2, 1 + (1 / numGrids) / 2, 0 - (1 / numGrids) / 2, 1 + (1 / numGrids) / 2),
        #           origin='lower', alpha=0.5)
        #plt.title('RealTraj Dist')
        #plt.show()


        self.JSDValue=np.zeros(1)
        for i in range(genTrajGrid.shape[0]):
            for j in range(genTrajGrid.shape[1]):
                P=(realTrajGrid[i,j]/totalReal)
                Q=(genTrajGrid[i,j]/totalGen)
                M=(P+Q)/2
                if M>0:
                    firstTerm=0
                    secondTerm=0
                    if P>0:
                        firstTerm=P*np.log(P/M)
                    if Q>0:
                        secondTerm=Q*np.log(Q/M)
                    self.JSDValue = self.JSDValue+(1/2)*firstTerm+(1/2)*secondTerm

    def trajToGrid(self, numGrids, generatedTrajs):
        gridData = np.zeros((numGrids, numGrids))
        for i in range(generatedTrajs.shape[0]):
            for j in range(generatedTrajs.shape[1]):
                gridX = np.floor(generatedTrajs[i, j, 0] * numGrids)
                gridY = np.floor(generatedTrajs[i, j, 1] * numGrids)
                if gridX>numGrids-1:
                    gridX=numGrids-1
                if gridX<0:
                    gridX=0
                if gridY>numGrids-1:
                    gridY=numGrids-1
                if gridY<0:
                    gridY=0
                gridData[int(gridX),int(gridY)] = gridData[int(gridX),int(gridY)]+1
        return gridData

    def trajToGrid_scale(self, numGrids, generatedTrajs, scale):
        gridData = np.zeros((numGrids, numGrids))
        for i in range(generatedTrajs.shape[0]):
            for j in range(generatedTrajs.shape[1]):
                gridX = np.floor((generatedTrajs[i, j, 0]/scale) * numGrids)
                gridY = np.floor((generatedTrajs[i, j, 1]/scale) * numGrids)
                if gridX>numGrids-1:
                    gridX=numGrids-1
                if gridX<0:
                    gridX=0
                if gridY>numGrids-1:
                    gridY=numGrids-1
                if gridY<0:
                    gridY=0
                gridData[int(gridX),int(gridY)] = gridData[int(gridX),int(gridY)]+1
        return gridData

class JSD_SingleB:
    def __init__(self, numGrids, realTraj, generatedTrajs, forbiddenMap, scale=None):
        if scale!=None:
            realTrajGrid = self.trajToGrid_scale(numGrids, realTraj,scale=scale)
        else:
            realTrajGrid = self.trajToGrid(numGrids, realTraj)

        totalReal = np.sum(realTrajGrid)
        if scale != None:
            genTrajGrid = self.trajToGrid_scale(numGrids, generatedTrajs,scale=scale)
        else:
            genTrajGrid = self.trajToGrid(numGrids, generatedTrajs)

        totalGen = np.sum(genTrajGrid)

        self.minBValue=10000
        minJSDValueLocal=100
        for b in np.arange(0.1,7,0.2):
            JSDValueLocal = np.zeros(1)
            #genTrajGridAdj = np.pow(genTrajGrid,b)

            tempAdj=np.pow((genTrajGrid / totalGen), b)
            genTrajGridAdjusted=tempAdj/np.sum(tempAdj)

            #plt.imshow(genTrajGridAdjusted, cmap="GnBu",
            #           extent=(
            #               0 - (1 / numGrids) / 2, 1 + (1 / numGrids) / 2, 0 - (1 / numGrids) / 2,
            #               1 + (1 / numGrids) / 2),
            #           origin='lower', alpha=0.5)
            #plt.title('GenTraj Dist')
            #plt.show()
            #print(b)

            for i in range(genTrajGrid.shape[0]):
                for j in range(genTrajGrid.shape[1]):
                    P = (realTrajGrid[i, j] / totalReal)
                    Q = genTrajGridAdjusted[i, j]
                    M = (P + Q) / 2
                    if M > 0:
                        firstTerm = 0
                        secondTerm = 0
                        if P > 0:
                            firstTerm = P * np.log(P / M)
                        if Q > 0:
                            secondTerm = Q * np.log(Q / M)
                        JSDValueLocal = JSDValueLocal + (1 / 2) * firstTerm + (1 / 2) * secondTerm
            if JSDValueLocal<minJSDValueLocal:
                self.minBValue=b
                minJSDValueLocal=JSDValueLocal
            print(JSDValueLocal)

        #genTrajGridAdj = np.pow(genTrajGrid, self.minBValue)

        tempAdj = np.pow((genTrajGrid / totalGen), self.minBValue)
        genTrajGridAdjusted = tempAdj / np.sum(tempAdj)

        self.JSDValue=np.zeros(1)
        for i in range(genTrajGrid.shape[0]):
            for j in range(genTrajGrid.shape[1]):
                P=(realTrajGrid[i,j]/totalReal)
                Q=genTrajGridAdjusted[i,j]
                M=(P+Q)/2
                if M>0:
                    firstTerm=0
                    secondTerm=0
                    if P>0:
                        firstTerm=P*np.log(P/M)
                    if Q>0:
                        secondTerm=Q*np.log(Q/M)
                    self.JSDValue = self.JSDValue+(1/2)*firstTerm+(1/2)*secondTerm

        #plt.imshow(forbiddenMap, cmap="cool",
        #           extent=(
        #           0 - (1 / numGrids) / 2, 1 + (1 / numGrids) / 2, 0 - (1 / numGrids) / 2, 1 + (1 / numGrids) / 2),
        #           origin='lower')

        #plt.imshow(realTrajGrid, cmap="Oranges",
        #           extent=(
        #           0 - (1 / numGrids) / 2, 1 + (1 / numGrids) / 2, 0 - (1 / numGrids) / 2, 1 + (1 / numGrids) / 2),
        #           origin='lower', alpha=0.5)
        #plt.title('RealTraj Dist with Forbiddens')
        #plt.show()

        #plt.imshow(forbiddenMap, cmap="cool",
        #           extent=(
        #               0 - (1 / numGrids) / 2, 1 + (1 / numGrids) / 2, 0 - (1 / numGrids) / 2, 1 + (1 / numGrids) / 2),
        #           origin='lower')

        #plt.imshow(genTrajGrid, cmap="GnBu",
        #           extent=(
        #               0 - (1 / numGrids) / 2, 1 + (1 / numGrids) / 2, 0 - (1 / numGrids) / 2, 1 + (1 / numGrids) / 2),
        #           origin='lower', alpha=0.5)
        #plt.title('GenTraj Dist with Forbiddens')
        #plt.show()

        #plt.imshow(genTrajGrid, cmap="GnBu",
        #           extent=(
        #               0 - (1 / numGrids) / 2, 1 + (1 / numGrids) / 2, 0 - (1 / numGrids) / 2, 1 + (1 / numGrids) / 2),
        #           origin='lower', alpha=0.5)
        #plt.title('GenTraj Dist')
        #plt.show()

        #plt.imshow(realTrajGrid, cmap="GnBu",
        #           extent=(
        #               0 - (1 / numGrids) / 2, 1 + (1 / numGrids) / 2, 0 - (1 / numGrids) / 2, 1 + (1 / numGrids) / 2),
        #           origin='lower', alpha=0.5)
        #plt.title('RealTraj Dist')
        #plt.show()

    def trajToGrid(self, numGrids, generatedTrajs):
        gridData = np.zeros((numGrids, numGrids))
        for i in range(generatedTrajs.shape[0]):
            for j in range(generatedTrajs.shape[1]):
                gridX = np.floor(generatedTrajs[i, j, 0] * numGrids)
                gridY = np.floor(generatedTrajs[i, j, 1] * numGrids)
                if gridX>numGrids-1:
                    gridX=numGrids-1
                if gridX<0:
                    gridX=0
                if gridY>numGrids-1:
                    gridY=numGrids-1
                if gridY<0:
                    gridY=0
                gridData[int(gridX),int(gridY)] = gridData[int(gridX),int(gridY)]+1
        return gridData

    def trajToGrid_scale(self, numGrids, generatedTrajs, scale):
        gridData = np.zeros((numGrids, numGrids))
        for i in range(generatedTrajs.shape[0]):
            for j in range(generatedTrajs.shape[1]):
                gridX = np.floor((generatedTrajs[i, j, 0]/scale) * numGrids)
                gridY = np.floor((generatedTrajs[i, j, 1]/scale) * numGrids)
                if gridX>numGrids-1:
                    gridX=numGrids-1
                if gridX<0:
                    gridX=0
                if gridY>numGrids-1:
                    gridY=numGrids-1
                if gridY<0:
                    gridY=0
                gridData[int(gridX),int(gridY)] = gridData[int(gridX),int(gridY)]+1
        return gridData