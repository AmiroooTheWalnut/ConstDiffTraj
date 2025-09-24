# Version 5 with initial condition with 4 inputs
from sympy.abc import alpha
from torch.optim.lr_scheduler import ExponentialLR

import DataGeneratorFcn
import V12FancySchedule_weightOnly as MinimalisticModel
from tqdm.auto import trange
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from datetime import datetime
import math
import torch
import torch.nn as nn
import torch.optim as optim
from QualityMeasure import JSD, JSD_SingleB
from torchviz import make_dot
from torchview import draw_graph
import PIL.Image as Image

from LossVisualizer import LossVisualizer

print(torch.cuda.is_available())

modelVersion="V12AdvancedWeighting_80_fast_automated"
isTrainModel=False
continueTrain=True
isChangeWeights=True
isRunOnCPU=False
isCompleteRandomCities=False
weightingStepSize=0.1
beforeStepsFixed=150
numOptIterates=250
extraStepsFixed=150
initialLR=0.00001

def visualiza(selectedOrNotSelected,numInstances, input, axis=None, auxSelectedOrNotSelected=None):
    cmap_name = 'viridis'  # Example: Use the 'viridis' colormap
    cmap = cm.get_cmap(cmap_name, numInstances)
    for i in range(numInstances):
        for h in range(1, trajectoryLength):
            if input[i, h, 0] > -1:
                color = cmap(i)
                if axis==None:
                    plt.plot([input[i, h - 1, 0], input[i, h, 0]], [input[i, h - 1, 1], input[i, h, 1]], color=color,
                             marker='', zorder=2, alpha=0.5)
                else:
                    axis.plot([input[i, h - 1, 0], input[i, h, 0]], [input[i, h - 1, 1], input[i, h, 1]], color=color,
                             marker='', zorder=2, alpha=0.5)

    if auxSelectedOrNotSelected is not None:
        selectedOrNotSelected=selectedOrNotSelected-1*auxSelectedOrNotSelected
    if axis == None:
        plt.imshow(selectedOrNotSelected.transpose(), cmap="cool",
                    extent=(0 - (1 / nGrid) / 2, 1 + (1 / nGrid) / 2, 0 - (1 / nGrid) / 2, 1 + (1 / nGrid) / 2),
                    origin='lower', zorder=1, alpha=0.99)
    else:
        axis.imshow(selectedOrNotSelected.transpose(), cmap="cool",
                   extent=(0 - (1 / nGrid) / 2, 1 + (1 / nGrid) / 2, 0 - (1 / nGrid) / 2, 1 + (1 / nGrid) / 2),
                   origin='lower', zorder=1, alpha=0.99)
    # plt.grid(True)
    if axis == None:
        plt.show()

numTrajectories=4000
trajectoryLength=12

numCities = 1
numCitiesTrain = 1
rng = np.random.default_rng(0)
seeds = rng.integers(low=0, high=100, size=numCities)
# seeds[0] = 0#FOR DEBUGGING!
datas=[]
selectedOrNotSelecteds=[]
serializedSelected2DMatrixs=[]

[data, nGrid, selectedOrNotSelected, serializedSelected2DMatrix]=DataGeneratorFcn.generateSyntheticDataFixedLengthInputImage("testStreets2.png", numTrajectories=numTrajectories,
                                                                trajectoryLength=trajectoryLength, numGrid=40,
                                                                seed=123, visualize=False)

if isRunOnCPU == False:
    serializedSelected2DMatrix = torch.from_numpy(serializedSelected2DMatrix).cuda()
else:
    serializedSelected2DMatrix = torch.from_numpy(serializedSelected2DMatrix)

time = datetime.now()#MAIN
np.random.seed(time.minute + time.hour + time.microsecond)#MAIN
np.random.shuffle(data)#MAIN

datas.append(data)
selectedOrNotSelecteds.append(selectedOrNotSelected)
serializedSelected2DMatrixs.append(serializedSelected2DMatrix)

for c in range(numCities-1):
    # [data, nGrid, selectedOrNotSelected,
     # serializedSelected2DMatrix] = DataGeneratorFcn.generateSyntheticDataFixedLength(numTrajectories=numTrajectories,
     #                                                                                 trajectoryLength=trajectoryLength,
     #                                                                                 numGrid=40, seed=seeds[c],
     #                                                                                 visualize=False)

    [data, nGrid, selectedOrNotSelected, serializedSelected2DMatrix, _]=DataGeneratorFcn.generateSyntheticDataFixedLength(numTrajectories=numTrajectories,
                                                                trajectoryLength=trajectoryLength, numGrid=40,
                                                                seed=seeds[c], visualize=False)

    if isRunOnCPU == False:
        serializedSelected2DMatrix = torch.from_numpy(serializedSelected2DMatrix).cuda()
    else:
        serializedSelected2DMatrix = torch.from_numpy(serializedSelected2DMatrix)

    time = datetime.now()#MAIN
    np.random.seed(time.minute + time.hour + time.microsecond)#MAIN
    np.random.shuffle(data)#MAIN

    datas.append(data)
    selectedOrNotSelecteds.append(selectedOrNotSelected)
    serializedSelected2DMatrixs.append(serializedSelected2DMatrix)


#[data,nGrid,selectedOrNotSelected]=DataGeneratorFcn.generateSyntheticDataFixedLength(numTrajectories=numTrajectories,trajectoryLength=trajectoryLength,numGrid=40,seed=3,visualize=False)
# DataGeneratorFcn.generateSyntheticDataVariableLength(numTrajectories=100,longestTrajectory=80,numGrid=50,seed=3,visualize=True)

timesteps = 16
BATCH_SIZE = 200

# if isRunOnCPU==False:
#     serializedSelected2DMatrix = torch.from_numpy(serializedSelected2DMatrix).cuda()
# else:
#     serializedSelected2DMatrix = torch.from_numpy(serializedSelected2DMatrix)

model = MinimalisticModel.SimpleNN(serializedSelected2DMatrix,lenForbidden=serializedSelected2DMatrix.shape[1],maxLengthSize=trajectoryLength,temporalFeatureSize=2)

total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total number of trainable parameters: {total_trainable_params}")

# Visualizing the graph
# Test input for visualization
in1=torch.randn(BATCH_SIZE, trajectoryLength, 2)
in2=torch.randn(BATCH_SIZE,1)
cond1=torch.randn(BATCH_SIZE,1)
cond2=torch.randn(BATCH_SIZE,1)
# Forward pass
if isRunOnCPU==False:
    model=model.cuda()
    in1=in1.cuda()
    in2=in2.cuda()
    cond1 = cond1.cuda()
    cond2 = cond2.cuda()
y = model(in1, in2, cond1, cond2)

if isRunOnCPU==False:
    graph = draw_graph(model, input_data=(in1, in2, cond1, cond2), expand_nested=True, device="cuda")
else:
    graph = draw_graph(model, input_data=(in1, in2, cond1, cond2), expand_nested=True, device="cpu")

graph.visual_graph.render('model_pytorch'+modelVersion, format="png")

# Loss and optimizer
if "LOSS" in modelVersion:
    criterion = MinimalisticModel.CustomLoss(serializedSelected2DMatrix,serializedSelected2DMatrix.shape[1],trajectoryLength,timesteps)
    # criterion.requires_grad_(True)
else:
    criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(),lr=initialLR)
# optimizer = optim.NAdam(model.parameters(),lr=0.00002)

my_file = Path('model'+modelVersion+'.pytorch')
if my_file.is_file() and continueTrain==True:
    try:
        model.load_state_dict(torch.load('model' + modelVersion + '.pytorch', weights_only=True))
        print("MODEL LOADED!")
    except:
        print("FAILED TO LOAD WEGHTS!")
if isRunOnCPU==False:
    model.cuda()

    # model = tf.keras.models.load_model('model'+modelVersion+'.pytorch')

def train_one(x_img,opt,cr,learningScheduleTime, isAdvancedWeighting=True, isAdvancedExponent=False):
    global isChangeWeights
    x_ts = MinimalisticModel.generate_ts(timesteps, len(x_img), learningScheduleTime, isChangeWeights, isAdvancedWeighting=isAdvancedWeighting)
    # x_ts = MinimalisticModel.generate_ts(timesteps, timesteps)

    # idx=237
    # for h in range(1, 12):
    #     plt.plot([x_img[idx, h - 1, 0], x_img[idx, h, 0]], [x_img[idx, h - 1, 1], x_img[idx, h, 1]], marker='', zorder=2, alpha=0.5, color='b')
    # # plt.show()

    x_a, x_b = MinimalisticModel.forward_noise(timesteps, x_img, x_ts, learningScheduleTime, isChangeWeights,isVizualize=False,isAdvancedWeighting=isAdvancedExponent)
    x_a = torch.from_numpy(x_a).to(torch.float32)
    x_ts = torch.from_numpy(x_ts).to(torch.float32)
    x_b = torch.from_numpy(x_b).to(torch.float32)


    # fig, axs = plt.subplots(nrows=2, ncols=(int)(timesteps / 2))
    # for i in range(x_a.shape[0]):
    #     row = (int)(math.floor(i/(timesteps / 2)))
    #     col = (int)(i % (timesteps / 2))
    #     visualiza(selectedOrNotSelected, 16, x_a, axis=axs[row][col])
    #     axs[row][col].title.set_text("Time: "+str(i))
    # plt.show()
    #
    # fig1, axs1 = plt.subplots(nrows=1, ncols=(int)(timesteps / 1))
    # for i in range(x_a.shape[0]):
    #     row = (int)(math.floor(i / (timesteps / 2)))
    #     col = (int)(i % (timesteps / 2))
    #     visualiza(selectedOrNotSelected, 16, x_a, axis=axs[row][col])
    #     axs1[row][col].title.set_text("Time: " + str(i))
    # plt.show()

    x_ts = x_ts.unsqueeze(1)
    cond1 = torch.unsqueeze(torch.from_numpy(x_img[:,0,0]),dim=1).to(torch.float32)
    cond2 = torch.unsqueeze(torch.from_numpy(x_img[:,0,1]),dim=1).to(torch.float32)
    # opt.zero_grad()
    if isRunOnCPU == False:
        outputs = model(x_a.cuda(), x_ts.cuda(), cond1.cuda(), cond2.cuda())
        if "LOSS" in modelVersion:
            loss = cr(outputs, x_b.cuda(), x_ts.cuda())
        else:
            loss = cr(outputs, x_b.cuda())
    else:
        outputs = model(x_a, x_ts, cond1, cond2)
        if "LOSS" in modelVersion:
            loss = cr(outputs, x_b, x_ts)
        else:
            loss = cr(outputs, x_b)
    # loss.requires_grad = True
    loss.backward()
    # opt.step()
    # visualiza(x_a.shape[0], x_a)
    # visualiza(x_b.shape[0], x_b)
    return loss

def train(X_trains, selectedOrNotSelecteds, serializedSelected2DMatrixs, BATCH_SIZE,optimizer_input,cri, offsetWeighting=0, maxWeightingStep=0.1, numIterates=30, extraIterates=0, beforeStepsFixed=0, isVisualizeLoss=False, isAdvancedWeighting=True, isAdvancedExponent=False):
    allLoss=np.zeros(numIterates+extraIterates+beforeStepsFixed)
    bar = trange(numIterates+extraIterates+beforeStepsFixed)
    total = 10
    scheduler1 = ExponentialLR(optimizer_input, gamma=0.998)
    if isVisualizeLoss==True:
        lv=LossVisualizer(numIterates+extraIterates+beforeStepsFixed)
    optimResetCounter=0
    usingLR = initialLR
    for i in bar:
        if i > beforeStepsFixed+numIterates:
            adjustedScheduleValue=offsetWeighting+maxWeightingStep
        elif i<beforeStepsFixed:
            adjustedScheduleValue=0
        elif i>beforeStepsFixed:
            adjustedScheduleValue=math.pow(offsetWeighting+((i-beforeStepsFixed) / numIterates)*maxWeightingStep,1.2)
        # print("adjustedScheduleValue:")
        # print(adjustedScheduleValue)
        if optimResetCounter>numIterates*0.34:
            usingLR=usingLR*0.9
            optimizer_input = optim.Adam(model.parameters(), lr=usingLR)
            scheduler1 = ExponentialLR(optimizer_input, gamma=0.998)
            optimResetCounter=0
            print("OPTIMIZER RESET")
        optimResetCounter=optimResetCounter+1
        epoch_loss = 0.0
        # optimizer_input.zero_grad()
        for j in range(total):
            optimizer_input.zero_grad()
            if isCompleteRandomCities==True:
                time = datetime.now()
                seedVal=time.minute + time.hour + time.microsecond
                [data, nGrid, selectedOrNotSelected,
                 serializedSelected2DMatrix,_] = DataGeneratorFcn.generateSyntheticDataFixedLength(
                    numTrajectories=numTrajectories,
                    trajectoryLength=trajectoryLength,
                    numGrid=40, seed=seedVal,
                    visualize=False)
                # randIndex = np.random.randint(len(X_trains))
                X_train = data
                # model.sforbiddenSerialMap = selectedOrNotSelecteds[d]
                model.psl.lenForbidden = serializedSelected2DMatrix.shape[1]
                model.pslSum.lenForbidden = serializedSelected2DMatrix.shape[1]
                if isRunOnCPU == False:
                    serializedSelected2DMatrix = torch.from_numpy(serializedSelected2DMatrix).cuda()
                else:
                    serializedSelected2DMatrix = torch.from_numpy(serializedSelected2DMatrix)
                model.psl.B = serializedSelected2DMatrix
                model.pslSum.B = serializedSelected2DMatrix
                cri.forbiddens = serializedSelected2DMatrix
                cri.lenForbidden = serializedSelected2DMatrix.shape[1]
                # model.lenForbidden = serializedSelected2DMatrixs[d].shape[1]
                np.random.seed(time.minute + time.hour + time.microsecond)  # MAIN
                x_img = X_train[np.random.randint(len(X_train), size=BATCH_SIZE)]
                loss = train_one(x_img, optimizer_input, cri, adjustedScheduleValue,isAdvancedWeighting=isAdvancedWeighting,isAdvancedExponent=isAdvancedExponent)
                epoch_loss += loss.item() * 1
            else:
                for d in range(numCitiesTrain):
                    randIndex = np.random.randint(len(X_trains))
                    X_train = X_trains[randIndex]
                    # model.sforbiddenSerialMap = selectedOrNotSelecteds[d]
                    model.psl.lenForbidden = serializedSelected2DMatrixs[d].shape[1]
                    model.psl.B = serializedSelected2DMatrixs[d]
                    model.pslSum.lenForbidden = serializedSelected2DMatrixs[d].shape[1]
                    model.pslSum.B = serializedSelected2DMatrixs[d]
                    cri.forbiddens=serializedSelected2DMatrixs[d]
                    cri.lenForbidden=serializedSelected2DMatrixs[d].shape[1]

                    # model.lenForbidden = serializedSelected2DMatrixs[d].shape[1]
                    x_img = X_train[np.random.randint(len(X_train), size=BATCH_SIZE)]
                    loss = train_one(x_img, optimizer_input, cri, adjustedScheduleValue,isAdvancedWeighting=isAdvancedWeighting,isAdvancedExponent=isAdvancedExponent)
                    # optimizer_input.step()
                    epoch_loss += loss.item() * 1

            optimizer_input.step()
            #pg = (j / total) * 100
        # optimizer_input.step()
        if i % 3 == 0:
            bar.set_description(f'loss: {epoch_loss/total:.5f}, lr: {scheduler1.get_last_lr()[0]:.5f}')
        scheduler1.step()
        allLoss[i]=epoch_loss/total
        if isVisualizeLoss == True:
            lv.values[i] = epoch_loss/total
    model.eval()
    torch.save(model.state_dict(), 'model' + modelVersion + '.pytorch')
    print("MODEL SAVED!")
    if isVisualizeLoss == True:
        lv.visualize(saveFileName='Loss_pytorch_' + modelVersion + '.png',
                     titleText="Pytorch loss value over iterations. Model " + modelVersion + ".")
    return allLoss

if isTrainModel==True:
    allLossValues=[]
    repeatTimes=2
    maxOffsetMovement=0.0
    for r in range(repeatTimes):
        offsetValues = np.arange((r/repeatTimes)*maxOffsetMovement, 1, weightingStepSize)
        for s in range(offsetValues.shape[0]):
            stepSize=np.minimum(weightingStepSize,1-offsetValues[s])
            lossValuesRes=train(datas, selectedOrNotSelecteds, serializedSelected2DMatrixs, BATCH_SIZE, optimizer, criterion, offsetWeighting=offsetValues[s], maxWeightingStep=stepSize, numIterates=numOptIterates, extraIterates=extraStepsFixed, beforeStepsFixed=beforeStepsFixed, isVisualizeLoss=False, isAdvancedWeighting=True, isAdvancedExponent=False)
            allLossValues.append(lossValuesRes)
    resultAllLoss = np.concatenate(allLossValues)
    plt.plot(resultAllLoss)
    plt.title("All loss values during training")
    plt.savefig("V12LossValues.png")
    plt.show()


def predict(serializedSelected2DMatrix, trajectoryLength,numTraj=10):
    x = np.random.normal(loc=0.5,scale=0.5,size=(numTraj, trajectoryLength, 2))
    # x = np.random.normal(loc=0.5,scale=0.33,size=(numTraj, trajectoryLength, 2))
    # x = np.random.uniform(low=0, high=1, size=(numTraj, trajectoryLength, 2))

    for idx in range(x.shape[0]):
        for h in range(1, 12):
            plt.plot([x[idx, h - 1, 0], x[idx, h, 0]], [x[idx, h - 1, 1], x[idx, h, 1]], marker='',
                     zorder=2, alpha=0.5, color='g')
    plt.show()

    x = torch.from_numpy(x).to(torch.float32)
    model.psl.lenForbidden = serializedSelected2DMatrix.shape[1]
    model.pslSum.lenForbidden = serializedSelected2DMatrix.shape[1]
    # if isRunOnCPU == False:
    #     serializedSelected2DMatrix = torch.from_numpy(serializedSelected2DMatrix).cuda()
    # else:
    #     serializedSelected2DMatrix = torch.from_numpy(serializedSelected2DMatrix)
    model.psl.B = serializedSelected2DMatrix
    model.pslSum.B = serializedSelected2DMatrix

    cmap_name = 'jet'  # Example: Use the 'jet' colormap
    cmap = cm.get_cmap(cmap_name, timesteps)

    # fig, axs = plt.subplots(nrows=1, ncols=(int)(timesteps/1))
    with torch.no_grad():
        indices = np.random.choice(serializedSelected2DMatrix.shape[1], numTraj, replace=True)
        samples = serializedSelected2DMatrix[:,indices]
        samples=torch.transpose(samples,0,1)
        cond1 = torch.unsqueeze(samples[:, 0], dim=1)
        cond2 = torch.unsqueeze(samples[:, 1], dim=1)
        for i in trange(timesteps):
            color = cmap(i)
            resX = x.cpu().detach().numpy()
            for idx in range(x.shape[0]):
                for h in range(1, 12):
                    plt.plot([resX[idx, h - 1, 0], resX[idx, h, 0]], [resX[idx, h - 1, 1], resX[idx, h, 1]], marker='',
                             zorder=2, alpha=0.5, color=color)

            ## colVal=i%(int)(timesteps/2)
            ## rowVal=math.floor(i/(int)(timesteps/2))
            # cond1 = torch.unsqueeze(x[:, 0, 0], dim=1)
            # cond2 = torch.unsqueeze(x[:, 0, 1], dim=1)
            t = i
            x_ts = np.full((numTraj), t)
            x_ts = torch.from_numpy(x_ts).to(torch.float32)
            x_ts = x_ts.unsqueeze(1)
            if isRunOnCPU == False:
                x = model(x.cuda(),x_ts.cuda(), cond1.cuda(), cond2.cuda())
            else:
                x = model(x, x_ts, cond1, cond2)
    #         visualiza(selectedOrNotSelected, numTraj, x.cpu().detach().numpy(), axis=axs[i])
    #         axs[i].title.set_text("Time: "+str(i))
    plt.show()
    return x

numPredicts = 1000
for c in range(numCities):
    pred = predict(serializedSelected2DMatrixs[c], trajectoryLength,numTraj=numPredicts)

    pred = pred.cpu().detach().numpy()

    JSDValue = JSD(nGrid,datas[c],pred,selectedOrNotSelecteds[c])

    print("JSDValue")
    print(JSDValue.JSDValue)

    JSDValue_SingleB = JSD_SingleB(nGrid,datas[c],pred,selectedOrNotSelecteds[c])

    print("JSDValue_singleB")
    print(JSDValue_SingleB.JSDValue)
    print("B value")
    print(JSDValue_SingleB.minBValue)

    visualiza(selectedOrNotSelecteds[c],numPredicts, pred)

mainSelectedOrNotSelected = selectedOrNotSelecteds[0]

# FANCY TEST! CHANGE THE CITY AND TEST!!!
# [dataT,nGrid,selectedOrNotSelectedT,serializedSelected2DMatrixT,_]=DataGeneratorFcn.generateSyntheticDataFixedLength(numTrajectories=numTrajectories,trajectoryLength=trajectoryLength,numGrid=40,seed=1,visualize=False)

[data, nGrid, selectedOrNotSelectedT, serializedSelected2DMatrixT]=DataGeneratorFcn.generateSyntheticDataFixedLengthInputImage("testStreets3.png", numTrajectories=numTrajectories,
                                                                trajectoryLength=trajectoryLength, numGrid=40,
                                                                seed=123, visualize=False)

# serializedSelected2DMatrixT=serializedSelected2DMatrixT[:,0:302]
if isRunOnCPU==False:
    serializedSelected2DMatrixT=torch.from_numpy(serializedSelected2DMatrixT).cuda()
else:
    serializedSelected2DMatrixT = torch.from_numpy(serializedSelected2DMatrixT)
model.psl.B=serializedSelected2DMatrixT
model.pslSum.B=serializedSelected2DMatrixT

numPredicts = 800
pred = predict(serializedSelected2DMatrixT, trajectoryLength,numTraj=numPredicts)

pred = pred.cpu().detach().numpy()

JSDValue = JSD(nGrid,data,pred,selectedOrNotSelectedT)

print("JSDValue other city: ")
print(JSDValue.JSDValue)

JSDValue_SingleB = JSD_SingleB(nGrid,data,pred,selectedOrNotSelected)

print("JSDValue_singleB other city: ")
print(JSDValue_SingleB.JSDValue)
print("B value other city: ")
print(JSDValue_SingleB.minBValue)

visualiza(selectedOrNotSelectedT,numPredicts, pred, auxSelectedOrNotSelected=mainSelectedOrNotSelected)

#torch.save(model.state_dict(), 'model'+modelVersion+'.pytorch')

print("!!!")
