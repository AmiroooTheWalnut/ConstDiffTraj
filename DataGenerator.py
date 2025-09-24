import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

numTrajectories=5
longestTrajectory=40

genData=np.zeros((numTrajectories,longestTrajectory,2))-1
np.random.seed(0)

nGrid=10

selectedOrNotSelected=np.zeros((nGrid,nGrid))
serializedSelected=[]
for i in range(nGrid):
    for j in range(nGrid):
        if np.random.rand()<0.5:
            selectedOrNotSelected[i,j] = 1
            serializedSelected.append({i*nGrid+j: (i,j)})

for i in range(numTrajectories):
    trajLength = np.random.randint(longestTrajectory-2)+2
    startIndex = np.random.randint(len(serializedSelected))
    oneFrameRaw = np.zeros(trajLength)-1
    oneFrameRaw[0] = startIndex
    for j in range(1,trajLength):
        listOfNeighbourIndices=[]
        x = list(serializedSelected[int(oneFrameRaw[j - 1])].values())[0][0]
        y = list(serializedSelected[int(oneFrameRaw[j - 1])].values())[0][1]
        if x-1>-1 and y-1>-1:
            if selectedOrNotSelected[x - 1][y - 1] == 1:
                listOfNeighbourIndices.append((x - 1, y - 1))
        if x-1>-1:
            if selectedOrNotSelected[x - 1][y] == 1:
                listOfNeighbourIndices.append((x - 1, y))
        if x-1>-1 and y+1<nGrid:
            if selectedOrNotSelected[x - 1][y+1] == 1:
                listOfNeighbourIndices.append((x - 1, y+1))


        if y-1>-1:
            if selectedOrNotSelected[x][y - 1] == 1:
                listOfNeighbourIndices.append((x, y - 1))
        if y+1<nGrid:
            if selectedOrNotSelected[x][y+1] == 1:
                listOfNeighbourIndices.append((x, y+1))


        if x+1<nGrid and y-1>-1:
            if selectedOrNotSelected[x + 1][y - 1] == 1:
                listOfNeighbourIndices.append((x + 1, y - 1))
        if x+1<nGrid:
            if selectedOrNotSelected[x + 1][y] == 1:
                listOfNeighbourIndices.append((x + 1, y))
        if x+1<nGrid and y+1<nGrid:
            if selectedOrNotSelected[x + 1][y+1] == 1:
                listOfNeighbourIndices.append((x + 1, y+1))

        if len(listOfNeighbourIndices) == 0:
            continue
        else:
            selectedNeighbor = np.random.randint(len(listOfNeighbourIndices))
            for m in range(len(serializedSelected)):
                if list(serializedSelected[m].values())[0][0] == listOfNeighbourIndices[selectedNeighbor][0] and \
                        list(serializedSelected[m].values())[0][1] == listOfNeighbourIndices[selectedNeighbor][1]:
                    oneFrameRaw[j]=m
                    break
    # print('!!!')
    for h in range(len(oneFrameRaw)):
        x = list(serializedSelected[int(oneFrameRaw[h])].values())[0][0] / nGrid
        y = list(serializedSelected[int(oneFrameRaw[h])].values())[0][1] / nGrid
        genData[i, h, 0] = x
        genData[i, h, 1] = y
    # print('!!!')

cmap_name = 'viridis'  # Example: Use the 'viridis' colormap
cmap = cm.get_cmap(cmap_name,numTrajectories)
for i in range(numTrajectories):
    for h in range(1,longestTrajectory):
        if genData[i,h,0]>0:
            color = cmap(i)
            plt.plot([genData[i,h-1,0],genData[i,h,0]], [genData[i,h-1,1],genData[i,h,1]],color=color, marker='o')

plt.show()

# print('!!!')