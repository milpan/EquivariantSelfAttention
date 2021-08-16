import os

import e2cnn

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import PIL

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import configparser as ConfigParser
from tqdm.notebook import trange, tqdm
import utils
# Ipmport various network architectures
from networks import AGRadGalNet, DNSteerableLeNet, DNSteerableAGRadGalNet #e2cnn module only works in python3.7+
# Import various data classes
from datasets import FRDEEPF
from datasets import MiraBest_full, MBFRConfident, MBFRUncertain, MBHybrid
from datasets import MingoLoTSS, MLFR, MLFRTest

#Ensure the GPU is being used for the calculation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Function to generate uniform normal weights for the last fully connected layer
def GenerateRandomWeights(m):
        classname = m.__class__.__name__
        # for every Linear layer in a model..
        if classname.find('Linear') == 3:
            # apply a uniform distribution to the weights and a bias=0
            m.weight.data.uniform_(0.0, 1.0)
            m.bias.data.fill_(0)

# Make the math imports to calculate metrics related to the Fisher Information, such as the Jacobian/Hessian
def Jacobian(y, x, create_graph=False):                                                               
    jac = []                                                                                          
    flat_y = y.reshape(-1).cpu()                                                                            
    grad_y = torch.zeros_like(flat_y)                                                                 
    for i in range(len(flat_y)):                                                                      
        grad_y[i] = 1.  
        grad_x, = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True, create_graph=create_graph, allow_unused=True)
        print(grad_x)
        jac.append(grad_x.reshape(x.shape))                                                           
        grad_y[i] = 0.                                                                                
    return torch.stack(jac).reshape(y.shape + x.shape);

def FisherCalc(y, x):
    jacobian = Jacobian(y, x, create_graph=True)
    jacobian = jacobian.cpu().detach().numpy()
    fisher = torch.from_numpy(np.outer(jacobian, jacobian)).type(torch.DoubleTensor)
    return fisher;


#---MAIN FISHER INITALISATION---#
'''
model --> PyTorch Compatiable Model
data --> PyTorch/SKLearn Compatiable DataLoader Function
n_iterations --> Number of realisations of the fisher matrix
'''
def InitaliseFisher(model, data, n_iterations):
    #First we need to obtain the number of weights in the last fully connected layer and freeze the weights in every other layer
    print("Calculating " + n_iterations + " realisations of the Fisher Matrix")
    Fisher_Matrix = torch.zeros((n_iterations, n_weights, n_weights)).to(device)
    for i in trange(n_iterations):
        #Freeze ALL Layers
        for params in model.parameters():
            params.requires_grad = False;
        #Unfreeze Last Layer and Obtain the Number of Weights in that layer
        model.parameters()[-1].requires_grad = True
        weights = model.parameters()[-1]
        #net.classifier.weights
        n_weights = weights.size[0]
        #Iterate across the dataloader
        for x_n, y_n in data:
            x_n, y_n = x_n.to(device), y_n.to(device)
            #Generate random weights for the last fully connected layer
            model.apply(GenerateRandomWeights)
            newWeights = model.fc3.weights
            #Pass the data through the model and obtain both output and softmaxed probabiities
            #!!!May need to softmax manually if models outputs arent already being softmaxxed!!!
            f_n, pi_n = model(x_n)
            Fisher_Matrix += FisherCalc(f_n, newWeights)
        return ;

def Hessian(y, x):
    hessian = Jacobian(Jacobian(y, x, create_graph=False), x)
    return hessian;

#Functions to evaluate the Fisher Metrics, such as Eigenvalue and Rank
def calc_eig(fisher):
    eigval = torch.eig(fisher, eigenvectors=False,  out=None).eigenvalues[:,0]
    return eigval

def calc_rank(fisher):
    Rank = []
    with torch.no_grad():
        rank = torch.matrix_rank(fishercpu).item()
        Rank.append(rank)
        return Rank;
    
def normalise(Fishers):
    num_samples = len(Fishers)
    TrF_integral = (1 / num_samples) * np.sum(np.array([torch.trace(F) for F in Fishers]))
    return [((12) / TrF_integral) * F for F in Fishers]