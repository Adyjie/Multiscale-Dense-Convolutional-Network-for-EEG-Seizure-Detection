# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 10:13:37 2020

@author: ady-yu
"""
#import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import os
import os.path as osp
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from collections import OrderedDict
from torchsummary import summary
import scipy.io as scio

#%%
class _DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, bn_size):
        super(_DenseLayer, self).__init__()
        # expand channels
        self.norm1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(in_channels, bn_size*growth_rate, kernel_size=1, stride=1, bias=False)

        # path1 with kernel_size = 3
        self.norm2 = nn.BatchNorm1d(bn_size*growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(bn_size*growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)

        # path2 with kernel_size = 5
        self.norm3 = nn.BatchNorm1d(bn_size*growth_rate)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv1d(bn_size*growth_rate, growth_rate, kernel_size=5, stride=1, padding=2, bias=False)

        # path4 with kernel_size = 7
        self.norm4 = nn.BatchNorm1d(bn_size*growth_rate)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv1d(bn_size*growth_rate, growth_rate, kernel_size=7, stride=1, padding=3, bias=False)
        
        # compress
        self.norm5 = nn.BatchNorm1d(3*growth_rate)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv1d(3*growth_rate, growth_rate, kernel_size=1, stride=1, padding=0, bias=False)
        
    def forward(self, x):
        x1 = self.norm1(x)
        x1 = self.relu1(x1)
        x1 = self.conv1(x1)

        x_2 = self.norm2(x1)
        x_2 = self.relu2(x_2)
        x_2 = self.conv2(x_2)
        
        x_3 = self.norm3(x1)
        x_3 = self.relu3(x_3)
        x_3 = self.conv3(x_3)
        
        x_4 = self.norm4(x1)
        x_4 = self.relu4(x_4)
        x_4 = self.conv4(x_4)
        
        x_con = torch.cat([x_2, x_3, x_4], 1)
        compress = self.norm5(x_con)
        compress = self.relu5(compress)
        compress = self.conv5(compress)

        compress = torch.cat([x, compress], 1)

        return compress
        
        
class _Transition(nn.Sequential):
      def __init__(self, in_channels, out_channels):
          super(_Transition, self).__init__()
          self.add_module('norm', nn.BatchNorm1d(in_channels))
          self.add_module('relu', nn.ReLU(inplace=True))
          self.add_module('conv', nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False))
          self.add_module('avgpool', nn.AvgPool1d(kernel_size=2, stride=2))
          
class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, in_channels, bn_size, growth_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            self.add_module('denselayer%d' % (i+1),
                            _DenseLayer(in_channels+growth_rate*i, growth_rate, bn_size))        

class casAM(nn.Module):
    def __init__(self, num_layers, in_channels, bn_size, growth_rate):
        super(casAM, self).__init__()


            
class DenseNet_BC(nn.Module):
    def __init__(self, growth_rate=16, block_config=(8,8,8,8,8), bn_size=4, compression=0.5, num_classes=2):
        super(DenseNet_BC, self).__init__()
        # 初始卷积filter为： growth_rate*2
        num_init_feature = 2 * growth_rate
        self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv1d(21, num_init_feature, kernel_size=7, stride=1, padding=3, bias=False)),
                ('norm0', nn.BatchNorm1d(num_init_feature)),
                ('relu0', nn.ReLU(inplace=True)),
                ]))
        num_feature = num_init_feature
        for i, num_layers in enumerate(block_config):
            self.features.add_module('denseblock %d' % (i+1),
                                     _DenseBlock(num_layers, num_feature, bn_size, growth_rate))
            num_feature +=  growth_rate * num_layers
            if (i != len(block_config)-1):
                self.features.add_module('trainsition %d' % (i+1),
                                         _Transition(num_feature, int(num_feature * compression)))
                num_feature = int(num_feature * compression)
        
        self.features.add_module('normE', nn.BatchNorm1d(num_feature))
        self.features.add_module('reluE', nn.ReLU(inplace=True))
        self.features.add_module('avg_poolE', nn.AdaptiveAvgPool1d(1))
        
        self.classifier = nn.Linear(num_feature, 1)
        self.act = nn.Sigmoid()
        
    def forward(self, x):
        features = self.features(x)
        features = features.view(-1, 1, features.size(1))
        output = self.classifier(features)
        output = self.act(output)
        return output
    
def densenet():
    return DenseNet_BC(growth_rate=16, block_config=(8,8,8,8,8), bn_size=4, compression=0.5, num_classes=2)
    
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#net = densenet().to(device)
#summary(net, (21,256))
#print(net)


#%%

def train(epoch, model, lossFunction, optimizer, device, trainloader):
    
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0    # accumulate every batch loss in a epoch
    total = 0
    TP,TN,FN,FP = 0.0,0.0,0.0,0.0
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)     # load data to gpu device
        inputs, targets = Variable(inputs), Variable(targets)
        optimizer.zero_grad()                    # clear gradients of all optimized torch.Tensors
        outputs = model(inputs)                  # forward propagation return the value of softmax function
        
        loss = lossFunction(outputs, targets)    # compute loss
        loss.backward()                          # compute gradient of loss over parameters 
        optimizer.step()                         # update parameters with gradient descent 

        train_loss += loss.data.item()           # accumulate every batch loss in a epoch
        pre = torch.round(outputs)
        total += targets.size(0)
        pre1 = pre[:,0,0]

        TP += ((pre1==1) & (targets==1)).sum()
        TN += ((pre1==0) & (targets==0)).sum()
        FN += ((pre1==0) & (targets==1)).sum()
        FP += ((pre1==1) & (targets==0)).sum()

        TP = TP.to(torch.float64)
        TN = TN.to(torch.float64)
        FN = FN.to(torch.float64)
        FP = FP.to(torch.float64)

    print('Train loss: %.4f, Train Acc: %.4f%%, Train Sen: %.4f%%, Train Spe: %.4f%%, (%d/%d)'
          % (train_loss/(batch_idx+1), 100.0*(TP+TN)/(TP+TN+FP+FN), 100.0*TP/(TP+FN), 100.0*TN/(TN+FP), (TP+TN), total))
        


best_acc = float('-inf')  # best_acc
best_loss = float('inf')

def test(model, lossFunction, optimizer, device, testloader):

    model.eval()       #enter test mode
    test_loss = 0      # accumulate every batch loss in a epoch
    total = 0
    TP,TN,FN,FP = 0.0,0.0,0.0,0.0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = lossFunction(outputs, targets) #compute loss

            test_loss += loss.data.item() # accumulate every batch loss in a epoch
            pre = torch.round(outputs)
            total += targets.size(0)          
            pre1 = pre[:,0,0]
            
            TP += ((pre1==1) & (targets==1)).sum()
            TN += ((pre1==0) & (targets==0)).sum()
            FN += ((pre1==0) & (targets==1)).sum()
            FP += ((pre1==1) & (targets==0)).sum()
            
            TP = TP.to(torch.float64)
            TN = TN.to(torch.float64)
            FN = FN.to(torch.float64)
            FP = FP.to(torch.float64)            

        # print loss and acc
        print('Test loss: %.4f, Test Acc: %.4f%%, Test Sen: %.4f%%, Test Spe: %.4f%%, (%d/%d)'
              % (test_loss/(batch_idx+1), 100.*(TP+TN)/(TP+TN+FP+FN), 100.0*TP/(TP+FN), 100.0*TN/(TN+FP), (TP+TN), total))
        
    # Save checkpoint
    global best_acc
    global best_loss
    valAcc = 100.*(TP+TN)/(TP+TN+FP+FN)
    test_loss /= len(targets)
    if test_loss < best_loss:
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(model.state_dict(), './checkpoint/tugLoss.pth')
        best_loss = test_loss
        best_acc = valAcc
        print(' Model Saved !')
        print("=======================")
        print('*')
    else:
        print('Pass ')
        print("=======================")
        print('*')        

         
def data_loader():
    
    train_experimentdata = scio.loadmat('train_experimentdata_tug2.mat')
    test_experimentdata = scio.loadmat('valid_experimentdata_tug2.mat')
    train_label = scio.loadmat('train_label_tug2.mat')
    test_label = scio.loadmat('valid_label_tug2.mat')
    
    # Convert to array
    train_experimentdata = train_experimentdata['train_experimentdata_tug2']
    test_experimentdata = test_experimentdata['valid_experimentdata_tug2']
    train_label = train_label['train_label_tug2']
    test_label = test_label['valid_label_tug2']
    
    # 对应
    X_train = torch.Tensor(train_experimentdata)  
    X_test = torch.Tensor(test_experimentdata)
    Y_train = torch.Tensor(train_label[:,0]) 
    Y_test = torch.Tensor(test_label[:,0])
 
    # 维度变换
    X_train = X_train.permute(0,2,1)
    X_test = X_test.permute(0,2,1) 

    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, Y_test)
    trainloader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    testloader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)
    return trainloader, testloader        



def run(model, num_epochs):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
#   define the loss function and optimizer
    lr=0.005
    lossFunction = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    trainloader, testloader = data_loader()
    for epoch in range(num_epochs):
        train(epoch, model, lossFunction, optimizer, device, trainloader)
        test(model, lossFunction, device, testloader)
        if (epoch + 1) % 100 == 0 :
            lr = lr * 0.6
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
# start training and testing
model = densenet()
# num_epochs is adjustable
run(model, num_epochs=500)
print(best_acc)
print('tug1torch')          
        
        

        
        
        
        
        
        