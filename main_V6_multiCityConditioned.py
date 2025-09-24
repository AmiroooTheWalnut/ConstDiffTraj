# Version 5 with initial condition with 4 inputs
from sympy.abc import alpha
from torch.optim.lr_scheduler import ExponentialLR

import DataGeneratorFcn
import V10ConditionalUNet as MinimalisticModel
import V10_onlyForb as MinimalisticModel_unseen
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

modelVersionKnown="V10ConditionalUnet"
modelVersionUnknown="V10_onlyForb"
# modelVersionTotal="V10_mixed"
isTrainModel=True
continueTrain=True
isRunOnCPU=False
isCompleteRandomCities=False
numOptIterates=3200
initialLR=0.000001

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
        selectedOrNotSelected=selectedOrNotSelected+2*auxSelectedOrNotSelected
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

numKnownCities = 1
numUnknownCities = 3
rng = np.random.default_rng(0)
seeds = rng.integers(low=0, high=100, size=numUnknownCities)
# seeds[0] = 0#FOR DEBUGGING!
datasKnown=[]
selectedOrNotSelectedsKnown=[]
serializedSelected2DMatrixsKnown=[]
datasUnknown=[]
selectedOrNotSelectedsUnknown=[]
serializedSelected2DMatrixsUnknown=[]
serializedSelectedListUnknown=[]

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

datasKnown.append(data)
selectedOrNotSelectedsKnown.append(selectedOrNotSelected)
serializedSelected2DMatrixsKnown.append(serializedSelected2DMatrix)

for c in range(numUnknownCities):
    # [data, nGrid, selectedOrNotSelected,
     # serializedSelected2DMatrix] = DataGeneratorFcn.generateSyntheticDataFixedLength(numTrajectories=numTrajectories,
     #                                                                                 trajectoryLength=trajectoryLength,
     #                                                                                 numGrid=40, seed=seeds[c],
     #                                                                                 visualize=False)

    [data, nGrid, selectedOrNotSelected, serializedSelected2DMatrix,serializedSelectedList]=DataGeneratorFcn.generateSyntheticDataFixedLength(numTrajectories=numTrajectories,
                                                                trajectoryLength=trajectoryLength, numGrid=40,
                                                                seed=seeds[c], visualize=False)

    if isRunOnCPU == False:
        serializedSelected2DMatrix = torch.from_numpy(serializedSelected2DMatrix).cuda()
    else:
        serializedSelected2DMatrix = torch.from_numpy(serializedSelected2DMatrix)

    time = datetime.now()#MAIN
    np.random.seed(time.minute + time.hour + time.microsecond)#MAIN
    np.random.shuffle(data)#MAIN

    datasUnknown.append(data)
    selectedOrNotSelectedsUnknown.append(selectedOrNotSelected)
    serializedSelected2DMatrixsUnknown.append(serializedSelected2DMatrix)
    serializedSelectedListUnknown.append(serializedSelectedList)


#[data,nGrid,selectedOrNotSelected]=DataGeneratorFcn.generateSyntheticDataFixedLength(numTrajectories=numTrajectories,trajectoryLength=trajectoryLength,numGrid=40,seed=3,visualize=False)
# DataGeneratorFcn.generateSyntheticDataVariableLength(numTrajectories=100,longestTrajectory=80,numGrid=50,seed=3,visualize=True)

timesteps = 16
BATCH_SIZE = 140

# if isRunOnCPU==False:
#     serializedSelected2DMatrix = torch.from_numpy(serializedSelected2DMatrix).cuda()
# else:
#     serializedSelected2DMatrix = torch.from_numpy(serializedSelected2DMatrix)

modelKnown = MinimalisticModel.SimpleNN(serializedSelected2DMatrix,lenForbidden=serializedSelected2DMatrix.shape[1],maxLengthSize=trajectoryLength,temporalFeatureSize=2)
modelUnknown = MinimalisticModel.SimpleNN(serializedSelected2DMatrix,lenForbidden=serializedSelected2DMatrix.shape[1],maxLengthSize=trajectoryLength,temporalFeatureSize=2)

# Visualizing the graph
# Test input for visualization
in1=torch.randn(BATCH_SIZE, trajectoryLength, 2)
in2=torch.randn(BATCH_SIZE,1)
cond1=torch.randn(BATCH_SIZE,1)
cond2=torch.randn(BATCH_SIZE,1)
# Forward pass
if isRunOnCPU==False:
    modelKnown=modelKnown.cuda()
    in1=in1.cuda()
    in2=in2.cuda()
    cond1 = cond1.cuda()
    cond2 = cond2.cuda()
y = modelKnown(in1, in2, cond1, cond2)
if isRunOnCPU==False:
    modelUnknown=modelUnknown.cuda()
    in1=in1.cuda()
    in2=in2.cuda()
    cond1 = cond1.cuda()
    cond2 = cond2.cuda()
y = modelUnknown(in1, in2, cond1, cond2)

if isRunOnCPU==False:
    graph = draw_graph(modelKnown, input_data=(in1, in2, cond1, cond2), expand_nested=True, device="cuda")
else:
    graph = draw_graph(modelKnown, input_data=(in1, in2, cond1, cond2), expand_nested=True, device="cpu")

graph.visual_graph.render('model_pytorch'+modelVersionKnown, format="png")

if isRunOnCPU==False:
    graph = draw_graph(modelUnknown, input_data=(in1, in2, cond1, cond2), expand_nested=True, device="cuda")
else:
    graph = draw_graph(modelUnknown, input_data=(in1, in2, cond1, cond2), expand_nested=True, device="cpu")

graph.visual_graph.render('model_pytorch'+modelVersionUnknown, format="png")

# # Open and display the image using Matplotlib
# img = Image.open('model_pytorch'+modelVersion+".png")
# plt.figure(figsize=(10, 10))
# plt.imshow(img)
# plt.axis("off")  # Hide axes
# plt.show()
#
# # Visualize execution graph
# dot = make_dot(y, params=dict(model.named_parameters()))
# dot.format = "png"
# dot.render('model_pytorch_alt'+modelVersion)
# dot.view()

# Loss and optimizer
if "LOSS" in modelVersionKnown:
    criterion = MinimalisticModel.CustomLoss(serializedSelected2DMatrix,serializedSelected2DMatrix.shape[1],trajectoryLength,timesteps)
    # criterion.requires_grad_(True)
else:
    criterion = nn.L1Loss()
if "LOSS" in modelVersionUnknown:
    criterion = MinimalisticModel.CustomLoss(serializedSelected2DMatrix,serializedSelected2DMatrix.shape[1],trajectoryLength,timesteps)
    # criterion.requires_grad_(True)
else:
    criterion = nn.L1Loss()
optimizerKnown = optim.Adam(modelKnown.parameters(),lr=initialLR)
optimizerUnknown = optim.Adam(modelUnknown.parameters(),lr=initialLR)
# optimizer = optim.NAdam(model.parameters(),lr=0.00002)

my_file = Path('model'+modelVersionKnown+'.pytorch')
if my_file.is_file() and continueTrain==True:
    try:
        modelKnown.load_state_dict(torch.load('model' + modelVersionKnown + '.pytorch', weights_only=True))
    except:
        print("FAILED TO LOAD WEGHTS!")
if isRunOnCPU==False:
    modelKnown.cuda()

    # model = tf.keras.models.load_model('model'+modelVersion+'.pytorch')

def train_one(x_img,opt,cr,model_in):
    x_ts = MinimalisticModel.generate_ts(timesteps, len(x_img))
    # x_ts = MinimalisticModel.generate_ts(timesteps, timesteps)

    # idx=237
    # for h in range(1, 12):
    #     plt.plot([x_img[idx, h - 1, 0], x_img[idx, h, 0]], [x_img[idx, h - 1, 1], x_img[idx, h, 1]], marker='', zorder=2, alpha=0.5, color='b')
    # # plt.show()

    x_a, x_b = MinimalisticModel.forward_noise(timesteps, x_img, x_ts)
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
        outputs = model_in(x_a.cuda(), x_ts.cuda(), cond1.cuda(), cond2.cuda())
        if "LOSS" in modelVersionKnown:
            loss = cr(outputs, x_b.cuda(), x_ts.cuda())
        else:
            loss = cr(outputs, x_b.cuda())
    else:
        outputs = model_in(x_a, x_ts, cond1, cond2)
        if "LOSS" in modelVersionKnown:
            loss = cr(outputs, x_b, x_ts)
        else:
            loss = cr(outputs, x_b)
    # loss.requires_grad = True
    loss.backward()
    # opt.step()
    # visualiza(x_a.shape[0], x_a)
    # visualiza(x_b.shape[0], x_b)
    return loss

def train(X_trainsKnown, selectedOrNotSelectedsKnown, serializedSelected2DMatrixsKnown,
          X_trainsUnknown, selectedOrNotSelectedsUnknown, serializedSelected2DMatrixsUnknown,
          BATCH_SIZE,optimizerKnown_in, optimizerUnknown_in,cri, numIterates=30, isVisualizeLoss=False):
    bar = trange(numIterates)
    total = 15
    schedulerKnown = ExponentialLR(optimizerKnown_in, gamma=0.9985)
    schedulerUnknown = ExponentialLR(optimizerUnknown_in, gamma=0.9985)
    if isVisualizeLoss==True:
        lv_k=LossVisualizer(numIterates)
        lv_uk = LossVisualizer(numIterates)
    optimResetCounter=0
    usingLR = initialLR
    for i in bar:
        if optimResetCounter>numIterates*0.34:
            usingLR=usingLR*0.8
            optimizerKnown_in = optim.Adam(modelKnown.parameters(), lr=usingLR)
            schedulerKnown = ExponentialLR(optimizerKnown_in, gamma=0.9985)
            optimizerUnknown_in = optim.Adam(modelKnown.parameters(), lr=usingLR)
            schedulerUnknown = ExponentialLR(optimizerUnknown_in, gamma=0.9985)
            optimResetCounter=0
            print("OPTIMIZER RESET")
        optimResetCounter=optimResetCounter+1
        epoch_loss_k = 0.0
        epoch_loss_uk = 0.0
        optimizerKnown_in.zero_grad()
        optimizerUnknown_in.zero_grad()
        for j in range(total):
            if isCompleteRandomCities==True:
                time = datetime.now()
                seedVal=time.minute + time.hour + time.microsecond
                [data, nGrid, selectedOrNotSelected,
                 serializedSelected2DMatrix] = DataGeneratorFcn.generateSyntheticDataFixedLength(
                    numTrajectories=numTrajectories,
                    trajectoryLength=trajectoryLength,
                    numGrid=40, seed=seedVal,
                    visualize=False)
                # randIndex = np.random.randint(len(X_trains))
                X_train = data
                # model.sforbiddenSerialMap = selectedOrNotSelecteds[d]
                modelUnknown.psl.lenForbidden = serializedSelected2DMatrix.shape[1]
                modelUnknown.pslSum.lenForbidden = serializedSelected2DMatrix.shape[1]
                if isRunOnCPU == False:
                    serializedSelected2DMatrix = torch.from_numpy(serializedSelected2DMatrix).cuda()
                else:
                    serializedSelected2DMatrix = torch.from_numpy(serializedSelected2DMatrix)
                modelUnknown.psl.B = serializedSelected2DMatrix
                modelUnknown.pslSum.B = serializedSelected2DMatrix
                cri.forbiddens = serializedSelected2DMatrix
                cri.lenForbidden = serializedSelected2DMatrix.shape[1]
                # model.lenForbidden = serializedSelected2DMatrixs[d].shape[1]
                np.random.seed(time.minute + time.hour + time.microsecond)  # MAIN
                x_img = X_train[np.random.randint(len(X_train), size=BATCH_SIZE)]
                loss = train_one(x_img, optimizerUnknown_in, cri,modelUnknown)
                epoch_loss_uk += loss.item() * 1
                optimizerUnknown_in.step()
            else:
                for d in range(len(X_trainsKnown)):
                    randIndex = np.random.randint(len(X_trainsKnown))
                    X_train = X_trainsKnown[randIndex]
                    # model.sforbiddenSerialMap = selectedOrNotSelecteds[d]
                    modelKnown.psl.lenForbidden = serializedSelected2DMatrixsKnown[d].shape[1]
                    modelKnown.psl.B = serializedSelected2DMatrixsKnown[d]
                    modelKnown.pslSum.lenForbidden = serializedSelected2DMatrixsKnown[d].shape[1]
                    modelKnown.pslSum.B = serializedSelected2DMatrixsKnown[d]
                    cri.forbiddens = serializedSelected2DMatrixsKnown[d]
                    cri.lenForbidden = serializedSelected2DMatrixsKnown[d].shape[1]
                    # model.lenForbidden = serializedSelected2DMatrixs[d].shape[1]
                    x_img = X_train[np.random.randint(len(X_train), size=BATCH_SIZE)]
                    loss = train_one(x_img, optimizerKnown_in, cri, modelKnown)
                    epoch_loss_k += loss.item() * 1
                for d in range(len(X_trainsUnknown)):
                    # randIndex = np.random.randint(len(X_trainsUnknown))
                    # X_train = X_trainsUnknown[randIndex]
                    # # model.sforbiddenSerialMap = selectedOrNotSelecteds[d]
                    data=DataGeneratorFcn.genOnlyTrajFixedLengthRandom(40,numTrajectories,
                        trajectoryLength,serializedSelectedListUnknown[d],selectedOrNotSelectedsUnknown[d])
                    X_train = data
                    modelUnknown.psl.lenForbidden = serializedSelected2DMatrixsUnknown[d].shape[1]
                    modelUnknown.psl.B = serializedSelected2DMatrixsUnknown[d]
                    modelUnknown.pslSum.lenForbidden = serializedSelected2DMatrixsUnknown[d].shape[1]
                    modelUnknown.pslSum.B = serializedSelected2DMatrixsUnknown[d]
                    cri.forbiddens = serializedSelected2DMatrixsUnknown[d]
                    cri.lenForbidden = serializedSelected2DMatrixsUnknown[d].shape[1]
                    # # model.lenForbidden = serializedSelected2DMatrixs[d].shape[1]
                    x_img = X_train[np.random.randint(len(X_train), size=BATCH_SIZE)]
                    loss = train_one(x_img, optimizerUnknown_in, cri, modelUnknown)
                    epoch_loss_uk += loss.item() * 1

                optimizerKnown_in.step()
                optimizerUnknown_in.step()


            #pg = (j / total) * 100
        if i % 3 == 0:
            bar.set_description(f'loss K: {epoch_loss_k/total:.5f}, lr K: {schedulerKnown.get_last_lr()[0]:.5f},loss UK: {epoch_loss_uk/total:.5f}, lr UK: {schedulerUnknown.get_last_lr()[0]:.5f}')
        schedulerKnown.step()
        schedulerUnknown.step()
        if isVisualizeLoss == True:
            lv_k.values[i] = epoch_loss_k/total
        if isVisualizeLoss == True:
            lv_uk.values[i] = epoch_loss_uk/total
    if isVisualizeLoss == True:
        lv_k.visualize(saveFileName='Loss_pytorch_' + modelVersionKnown + '.png',
                     titleText="Pytorch loss value over iterations. Model " + modelVersionKnown + ".")
    if isVisualizeLoss == True:
        lv_uk.visualize(saveFileName='Loss_pytorch_' + modelVersionUnknown + '.png',
                     titleText="Pytorch loss value over iterations. Model " + modelVersionUnknown + ".")

if isTrainModel==True:
    train(datasKnown, selectedOrNotSelectedsKnown, serializedSelected2DMatrixsKnown,
          datasUnknown, selectedOrNotSelectedsUnknown, serializedSelected2DMatrixsUnknown,
          BATCH_SIZE, optimizerKnown, optimizerUnknown, criterion, numIterates=numOptIterates, isVisualizeLoss=True)
    modelKnown.eval()
    torch.save(modelKnown.state_dict(), 'model' + modelVersionKnown + '.pytorch')
    modelUnknown.eval()
    torch.save(modelUnknown.state_dict(), 'model' + modelVersionKnown + '.pytorch')


def predict(serializedSelected2DMatrix, trajectoryLength,model,numTraj=10):
    x = np.random.normal(loc=0.5,size=(numTraj, trajectoryLength, 2))
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

numPredicts = 100
for c in range(numKnownCities):
    pred = predict(serializedSelected2DMatrixsKnown[c], trajectoryLength,modelKnown,numTraj=numPredicts)

    pred = pred.cpu().detach().numpy()

    JSDValue = JSD(nGrid,data,pred,selectedOrNotSelectedsKnown[c])

    print("JSDValue")
    print(JSDValue.JSDValue)

    JSDValue_SingleB = JSD_SingleB(nGrid,data,pred,selectedOrNotSelectedsKnown[c])

    print("JSDValue_singleB")
    print(JSDValue_SingleB.JSDValue)
    print("B value")
    print(JSDValue_SingleB.minBValue)

    visualiza(selectedOrNotSelectedsKnown[c],numPredicts, pred)

for c in range(numUnknownCities):
    pred = predict(serializedSelected2DMatrixsUnknown[c], trajectoryLength,modelUnknown,numTraj=numPredicts)

    pred = pred.cpu().detach().numpy()

    JSDValue = JSD(nGrid,data,pred,selectedOrNotSelectedsUnknown[c])

    print("JSDValue")
    print(JSDValue.JSDValue)

    JSDValue_SingleB = JSD_SingleB(nGrid,data,pred,selectedOrNotSelectedsUnknown[c])

    print("JSDValue_singleB")
    print(JSDValue_SingleB.JSDValue)
    print("B value")
    print(JSDValue_SingleB.minBValue)

    visualiza(selectedOrNotSelectedsUnknown[c],numPredicts, pred)

mainSelectedOrNotSelected = selectedOrNotSelectedsKnown[0]

# FANCY TEST! CHANGE THE CITY AND TEST!!!
[dataT,nGrid,selectedOrNotSelectedT,serializedSelected2DMatrixT]=DataGeneratorFcn.generateSyntheticDataFixedLength(numTrajectories=numTrajectories,trajectoryLength=trajectoryLength,numGrid=40,seed=1,visualize=False)
# serializedSelected2DMatrixT=serializedSelected2DMatrixT[:,0:302]
if isRunOnCPU==False:
    serializedSelected2DMatrixT=torch.from_numpy(serializedSelected2DMatrixT).cuda()
else:
    serializedSelected2DMatrixT = torch.from_numpy(serializedSelected2DMatrixT)
modelUnknown.psl.B=serializedSelected2DMatrixT
modelUnknown.pslSum.B=serializedSelected2DMatrixT

numPredicts = 100
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
