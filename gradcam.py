#!/usr/bin/env python
# coding: utf-8

# In[1]:


from inspect import getsource
import math
import json
from PIL import Image
from tqdm.notebook import tqdm

import numpy as np

import torch
from torch import nn
from torchvision import models
from torchvision import transforms

import matplotlib.pyplot as plt
plt.style.use('ggplot')



class Flatten(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return torch.flatten(x, 1)
    
    
model = models.resnet50(pretrained=True)
model.eval()
layers = [
    model.conv1,
    model.bn1,
    model.relu,
    model.maxpool,
    model.layer1,
    model.layer2,
    model.layer3,
    model.layer4,
    model.avgpool,
    Flatten(),
    model.fc
]


img_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
])

normalize = transforms.Compose([
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    ),
])

upsample = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(224),
    transforms.ToTensor(),
])


avg = nn.AdaptiveAvgPool2d(1)
relu = nn.ReLU()
softmax = nn.Softmax(dim=1)


def compute_gradcam(image, class_index=0, layers=layers, layer_index=7, normalize_gradcam=True):
    x = normalize(image).unsqueeze(0)
    act = nn.Sequential(*layers[:layer_index+1])(x)
    act = act.detach().requires_grad_(True)
    pred = softmax(
        nn.Sequential(*layers[layer_index+1:])(act)
    )
    # argmax = pred.argmax(dim=1)
    if act.grad is not None:
        act.grad.data.fill_(0)

        ## the two-liner:
    pred[:,class_index].sum().backward(retain_graph=True)
    gradcam = relu(act * avg(act.grad)).sum(dim=1)

    gradcam = gradcam.detach()
    if normalize_gradcam:
        vmax = gradcam.max()
        vmin = 0
        gradcam = (gradcam-vmin)/(vmax-vmin)
    return gradcam




# In[2]:


if __name__ == '__main__':
    image = Image.open('cat-dog.jpg')
    transformed_image = img_transform(image)

    gradcam = compute_gradcam(transformed_image, 1)
    # plt.imshow(transformed_image.permute(1,2,0))
    plt.imshow(gradcam[0]);
    # plt.imshow(transformed_image.permute(1,2,0) * upsample(gradcam[0]).permute(1,2,0), alpha=1)
    plt.show()

