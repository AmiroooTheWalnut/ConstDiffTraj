# Version 5 with initial condition with 2 inputs and condition is on the first step of the sequences
from torch.optim.lr_scheduler import ExponentialLR

import DataGeneratorFcn
import V5ConditionedModelSimple as MinimalisticModel
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

modelVersion="V5ConditionalModelSimple"
continueTrain=True
isRunOnCPU=False
numOptIterates=280

def visualiza(selectedOrNotSelected,numInstances, input, axis=None):
    cmap_name = 'viridis'  # Example: Use the 'viridis' colormap
    cmap = cm.get_cmap(cmap_name, numInstances)
    for i in range(numInstances):
        for h in range(1, trajectoryLength):
            if input[i, h, 0] > -1:
                color = cmap(i)
                if axis==None:
                    plt.plot([input[i, h - 1, 0], input[i, h, 0]], [input[i, h - 1, 1], input[i, h, 1]], color=color,
                             marker='o')
                else:
                    axis.plot([input[i, h - 1, 0], input[i, h, 0]], [input[i, h - 1, 1], input[i, h, 1]], color=color,
                             marker='o')
    if axis == None:
        plt.imshow(selectedOrNotSelected.transpose(), cmap="cool",
                    extent=(0 - (1 / nGrid) / 2, 1 + (1 / nGrid) / 2, 0 - (1 / nGrid) / 2, 1 + (1 / nGrid) / 2),
                    origin='lower')
    else:
        axis.imshow(selectedOrNotSelected.transpose(), cmap="cool",
                   extent=(0 - (1 / nGrid) / 2, 1 + (1 / nGrid) / 2, 0 - (1 / nGrid) / 2, 1 + (1 / nGrid) / 2),
                   origin='lower')
    # plt.grid(True)
    if axis == None:
        plt.show()

numTrajectories=200
trajectoryLength=8

[data,nGrid,selectedOrNotSelected,serializedSelected2DMatrix]=DataGeneratorFcn.generateSyntheticDataFixedLength(numTrajectories=numTrajectories,trajectoryLength=trajectoryLength,numGrid=40,seed=3,visualize=False)
#[data,nGrid,selectedOrNotSelected]=DataGeneratorFcn.generateSyntheticDataFixedLength(numTrajectories=numTrajectories,trajectoryLength=trajectoryLength,numGrid=40,seed=3,visualize=False)
# DataGeneratorFcn.generateSyntheticDataVariableLength(numTrajectories=100,longestTrajectory=80,numGrid=50,seed=3,visualize=True)

time=datetime.now()
np.random.seed(time.minute+time.hour+time.microsecond)

np.random.shuffle(data)

timesteps = 16
BATCH_SIZE = 32

if isRunOnCPU==False:
    serializedSelected2DMatrix=torch.from_numpy(serializedSelected2DMatrix).cuda()
else:
    serializedSelected2DMatrix = torch.from_numpy(serializedSelected2DMatrix)

model = MinimalisticModel.SimpleNN(serializedSelected2DMatrix,lenForbidden=serializedSelected2DMatrix.shape[1],maxLengthSize=trajectoryLength,temporalFeatureSize=2)

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
y = model(in1, in2)

if isRunOnCPU==False:
    graph = draw_graph(model, input_data=(in1, in2), expand_nested=True, device="cuda")
else:
    graph = draw_graph(model, input_data=(in1, in2), expand_nested=True, device="cpu")

graph.visual_graph.render('model_pytorch'+modelVersion, format="png")

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
if "LOSS" in modelVersion:
    criterion = MinimalisticModel.CustomLoss(serializedSelected2DMatrix,serializedSelected2DMatrix.shape[1],trajectoryLength,timesteps)
else:
    criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(),lr=0.00001)

my_file = Path('model'+modelVersion+'.pytorch')
if my_file.is_file() and continueTrain==True:
    try:
        model.load_state_dict(torch.load('model' + modelVersion + '.pytorch', weights_only=True))
    except:
        print("FAILED TO LOAD WEGHTS!")
if isRunOnCPU==False:
    model.cuda()

    # model = tf.keras.models.load_model('model'+modelVersion+'.pytorch')

def train_one(x_img,opt,cr):
    x_ts = MinimalisticModel.generate_ts(timesteps, len(x_img))
    x_a, x_b = MinimalisticModel.forward_noise(timesteps, x_img, x_ts)
    x_a = torch.from_numpy(x_a).to(torch.float32)
    x_ts = torch.from_numpy(x_ts).to(torch.float32)
    x_b = torch.from_numpy(x_b).to(torch.float32)
    x_ts = x_ts.unsqueeze(1)
    #cond1 = torch.unsqueeze(torch.from_numpy(x_img[:,0,0]),dim=1).to(torch.float32)
    #cond2 = torch.unsqueeze(torch.from_numpy(x_img[:,0,1]),dim=1).to(torch.float32)
    opt.zero_grad()
    if isRunOnCPU == False:
        outputs = model(x_a.cuda(), x_ts.cuda())
        if "LOSS" in modelVersion:
            loss = cr(outputs, x_b.cuda(), x_ts.cuda())
        else:
            loss = cr(outputs, x_b.cuda())
    else:
        outputs = model(x_a, x_ts)
        if "LOSS" in modelVersion:
            loss = cr(outputs, x_b, x_ts)
        else:
            loss = cr(outputs, x_b)
    loss.backward()
    opt.step()
    # visualiza(x_a.shape[0], x_a)
    # visualiza(x_b.shape[0], x_b)
    return loss

def train(X_train, BATCH_SIZE,optim,cri, numIterates=30, isVisualizeLoss=False):
    bar = trange(numIterates)
    total = 250
    scheduler1 = ExponentialLR(optimizer, gamma=0.995)
    if isVisualizeLoss==True:
        lv=LossVisualizer(numIterates)
    for i in bar:
        epoch_loss = 0.0
        for j in range(total):
            x_img = X_train[np.random.randint(len(X_train), size=BATCH_SIZE)]
            loss = train_one(x_img,optim,cri)
            epoch_loss += loss.item() * 1
            #pg = (j / total) * 100
        if i % 3 == 0:
            bar.set_description(f'loss: {epoch_loss/total:.5f}, lr: {scheduler1.get_last_lr()[0]:.5f}')
        scheduler1.step()
        if isVisualizeLoss == True:
            lv.values[i] = epoch_loss/total
    if isVisualizeLoss == True:
        lv.visualize(saveFileName='Loss_pytorch_' + modelVersion + '.png',
                     titleText="Pytorch loss value over iterations. Model " + modelVersion + ".")

train(data,BATCH_SIZE,optimizer,criterion,numIterates=numOptIterates,isVisualizeLoss=False)

model.eval()

def predict(trajectoryLength,numTraj=10):
    x = np.random.normal(size=(numTraj, trajectoryLength, 2))
    x = torch.from_numpy(x).to(torch.float32)
    # fig, axs = plt.subplots(nrows=2, ncols=(int)(timesteps/2))
    with torch.no_grad():
        indices = np.random.choice(serializedSelected2DMatrix.shape[1], numTraj, replace=True)
        samples = serializedSelected2DMatrix[:, indices]
        samples = torch.transpose(samples, 0, 1)
        cond1 = torch.unsqueeze(samples[:, 0], dim=1)
        cond2 = torch.unsqueeze(samples[:, 1], dim=1)
        for i in trange(timesteps):
            # colVal=i%(int)(timesteps/2)
            # rowVal=math.floor(i/(int)(timesteps/2))
            #cond1 = torch.unsqueeze(x[:, 0, 0], dim=1)
            #cond2 = torch.unsqueeze(x[:, 0, 1], dim=1)
            t = i
            x_ts = np.full((numTraj), t)
            x_ts = torch.from_numpy(x_ts).to(torch.float32)
            x_ts = x_ts.unsqueeze(1)
            x[:, 0, 0] = cond1[:, 0]
            x[:, 0, 1] = cond2[:, 0]
            if isRunOnCPU == False:
                x = model(x.cuda(),x_ts.cuda())
            else:
                x = model(x, x_ts)
            # visualiza(numTraj, x, axis=axs[rowVal,colVal])
            # axs[rowVal,colVal].title.set_text("Time: "+str(i))
    # plt.show()
    return x

torch.save(model.state_dict(), 'model'+modelVersion+'.pytorch')

numPredicts = 2000
pred = predict(trajectoryLength,numTraj=numPredicts)

pred = pred.cpu().detach().numpy()

JSDValue = JSD(nGrid,data,pred,selectedOrNotSelected)

print("JSDValue")
print(JSDValue.JSDValue)

JSDValue_SingleB = JSD_SingleB(nGrid,data,pred,selectedOrNotSelected)

print("JSDValue_singleB")
print(JSDValue_SingleB.JSDValue)
print("B value")
print(JSDValue_SingleB.minBValue)

visualiza(selectedOrNotSelected,numPredicts, pred)

# # FANCY TEST! CHANGE THE CITY AND TEST!!!
# [dataT,nGrid,selectedOrNotSelectedT,serializedSelected2DMatrixT]=DataGeneratorFcn.generateSyntheticDataFixedLength(numTrajectories=numTrajectories,trajectoryLength=trajectoryLength,numGrid=40,seed=0,visualize=False)
# serializedSelected2DMatrixT=serializedSelected2DMatrixT[:,0:302]
# if isRunOnCPU==False:
#     serializedSelected2DMatrixT=torch.from_numpy(serializedSelected2DMatrixT).cuda()
# else:
#     serializedSelected2DMatrixT = torch.from_numpy(serializedSelected2DMatrixT)
# model.psl.B=serializedSelected2DMatrixT
#
# numPredicts = 2000
# pred = predict(trajectoryLength,numTraj=numPredicts)
#
# pred = pred.cpu().detach().numpy()
#
# JSDValue = JSD(nGrid,data,pred,selectedOrNotSelectedT)
#
# print("JSDValue")
# print(JSDValue.JSDValue)
#
# visualiza(selectedOrNotSelectedT,numPredicts, pred)

#torch.save(model.state_dict(), 'model'+modelVersion+'.pytorch')

print("!!!")
