#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from torch import nn
import torch
import torch.nn.functional as F

# MNIST/EMNIST
class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        if args.dataset == 'mnist':
            outdim = 50
        if args.dataset == 'emnist':
            outdim = 512
        self.classifier = nn.Sequential(
                    nn.Linear(3380,outdim), # mnist:(3380,50);emnist(3380,512)
                    nn.ReLU(),  
                    nn.Dropout(p=0.5),
                    nn.Linear(outdim, args.num_classes),  #mnist:(50,10);emnist(512,62)
            )

    def forward(self, x, return_all_layers):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        logits = self.classifier(x.view(x.shape[0],-1))
        log_softmax = F.log_softmax(logits, dim=1)

        if return_all_layers:
            return [logits, log_softmax]
        else:
            return log_softmax

# CIFAR-10
class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) 
        self.dropout = nn.Dropout(p=0.5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        
        self.classifier = nn.Sequential(
                    nn.Linear(400, 120),
                    nn.ReLU(),  
                    nn.Linear(120, 84),
                    nn.ReLU(),  
                    nn.Linear(84, args.num_classes),
            )

    def forward(self, x, return_all_layers):
        x = self.pool(F.relu(self.dropout(self.conv1(x))))
        x = self.pool(F.relu(self.dropout(self.conv2(x))))

        logits = self.classifier(x.view(x.shape[0],-1))
        log_softmax = F.log_softmax(logits, dim=1)
        if return_all_layers:
            return [logits, log_softmax]
        else:
            return log_softmax

# CIFAR-100 VGG-19
class VGG19(torch.nn.Module):
    def __init__(self, args):
        super(VGG19, self).__init__()
        self.args = args
        
        # calculate same padding:
        # (w - k + 2*p)/s + 1 = o
        # => p = (s(o-1) - w + k)/2
        
        self.block_1 = nn.Sequential(
                nn.Conv2d(in_channels=self.args.num_channels,
                          out_channels=64,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          # (1(32-1)- 32 + 3)/2 = 1
                          padding=1), 
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(in_channels=64,
                          out_channels=64,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2),
                             stride=(2, 2))
        )
        
        self.block_2 = nn.Sequential(
                nn.Conv2d(in_channels=64,
                          out_channels=128,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(in_channels=128,
                          out_channels=128,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2),
                             stride=(2, 2))
        )
        
        self.block_3 = nn.Sequential(        
                nn.Conv2d(in_channels=128,
                          out_channels=256,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(in_channels=256,
                          out_channels=256,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),        
                nn.Conv2d(in_channels=256,
                          out_channels=256,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(in_channels=256,
                          out_channels=256,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2),
                             stride=(2, 2))
        )
        
          
        self.block_4 = nn.Sequential(   
                nn.Conv2d(in_channels=256,
                          out_channels=512,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),        
                nn.Conv2d(in_channels=512,
                          out_channels=512,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),        
                nn.Conv2d(in_channels=512,
                          out_channels=512,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(in_channels=512,
                          out_channels=512,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),   
                nn.MaxPool2d(kernel_size=(2, 2),
                             stride=(2, 2))
        )
        
        self.block_5 = nn.Sequential(
                nn.Conv2d(in_channels=512,
                          out_channels=512,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),            
                nn.Conv2d(in_channels=512,
                          out_channels=512,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),            
                nn.Conv2d(in_channels=512,
                          out_channels=512,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(in_channels=512,
                          out_channels=512,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),   
                nn.MaxPool2d(kernel_size=(2, 2),
                             stride=(2, 2))             
        )
        if args.dataset == 'cifar100':
            indim = 512
        if args.pro == 'True':
            self.pre_projection = nn.Sequential(
                nn.Linear(indim, 4096),
                nn.ReLU(),   
                nn.Linear(4096, 4096),
                nn.ReLU()
            )
            self.projection = nn.Linear(4096, 256) 
            self.classifier = nn.Linear(256, args.num_classes)
        else:
            self.classifier = nn.Sequential(
                    nn.Linear(indim, 4096),
                    nn.ReLU(),   
                    nn.Linear(4096, 4096),
                    nn.ReLU(),
                    nn.Linear(4096, args.num_classes)
            )
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                #n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                #m.weight.data.normal_(0, np.sqrt(2. / n))
                m.weight.detach().normal_(0, 0.05)
                if m.bias is not None:
                    m.bias.detach().zero_()
            elif isinstance(m, torch.nn.Linear):
                m.weight.detach().normal_(0, 0.05)
                m.bias.detach().detach().zero_()
            elif isinstance(m,nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def forward(self, x, return_all_layers=False):

        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)

        if self.args.pro == 'True':
            prepro = self.pre_projection(x.view(x.shape[0],-1))
            pro = self.projection(prepro)
            logits = self.classifier(pro)
        else:
            logits = self.classifier(x.view(x.shape[0],-1))
        log_softmax = F.log_softmax(logits, dim=1)
    
        if return_all_layers:
            if self.args.pro == 'True':
                return [pro, log_softmax]
            else: 
                return [logits, log_softmax]
        else:
            return log_softmax