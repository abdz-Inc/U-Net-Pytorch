import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import gc
import torchvision.transforms as transforms
import os
from tqdm import tqdm
import torch.optim as optim
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torch.nn.functional as F

from dataset import UnetDataset
from unet import Unet
from loss import DiceLoss


#transformation
myTrans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((512,512))
])


batch = 8
device = torch.device('cuda')
unet = Unet(3, 13).to(device)
seg = UnetDataset('path_to_input','path_to_segmentation',transform =myTrans)
loader = DataLoader(seg, batch_size = batch, pin_memory = True)

lr = 0.0001
n_epochs = 1
lossfn = DiceLoss()
optimizer = optim.Adam(unet.parameters(), lr = lr)

def train():
    loop = tqdm(loader)
    running_loss = 0

    for b_id, (x, y) in enumerate(loop):
        x = x.to(device)
        y = y.to(device)
        pred = unet(x)
        loss = lossfn(pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_val = loss.item()
        running_loss += loss_val

        loop.set_postfix_str(f'batch : {b_id}, loss: {loss_val}')

    print(f"Epoch : {epoch}, Epoch loss : {running_loss/len(loader)}")
    

def train_model():

    for epoch in range(n_epochs):

        train()

        state_dict = {'model':unet.state_dict(), 'optim':optimizer.state_dict()}

        with open('unet-model.pth.tar','wb') as f:
            torch.save(state_dict, f)