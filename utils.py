#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
import numpy as np
from torchvision import datasets, transforms
from sampling import *

def get_dataset(args): 
   
    if args.dataset == 'mnist':
        label_rate = 0.18
    if args.dataset == 'emnist':
        label_rate = 0.16
    if args.dataset == 'cifar10':
        label_rate = 0.24
    if args.dataset == 'cifar100':
        label_rate = 0.30 
        
    if args.dataset == 'cifar' or args.dataset =='cifar100':
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        if args.dataset == 'cifar':
            data_dir = './data/cifar/' 
            train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=apply_transform) 
            test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=apply_transform)
        else:
            data_dir = './data/cifar100/'
            train_dataset = datasets.CIFAR100(data_dir, train=True, download=True, transform=train_transform)
            test_dataset = datasets.CIFAR100(data_dir, train=False, download=True, transform=test_transform)
    
        if args.iid: 
            user_groups, user_groups_un = cifar_iid(train_dataset, args.num_users, label_rate)
        else: 
            if args.dataset == 'cifar':
                user_groups, user_groups_un = cifar_noniid(train_dataset, args.num_users, args.num_classes,label_rate)
            if args.dataset == 'cifar100':
                user_groups, user_groups_un = cifar100_noniid(train_dataset, args.num_users, args.num_classes,label_rate, args.alpha)

    if args.dataset == 'mnist' or args.dataset == 'emnist':
        if args.dataset == 'mnist':
            data_dir = './data/mnist'
        if args.dataset == 'emnist':
            data_dir = './data/emnist' 
        apply_transform = transforms.Compose([
            transforms.Resize([64, 64]),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        if args.dataset != 'emnist':
            train_dataset = datasets.MNIST(data_dir, train=True, download=False,transform=apply_transform)
            test_dataset = datasets.MNIST(data_dir, train=False, download=False,transform=apply_transform)
        else:
            train_dataset = datasets.EMNIST(data_dir,split='byclass', train=True, download=False,transform=apply_transform)
            test_dataset = datasets.EMNIST(data_dir,split = 'byclass', train=False, download=False,transform=apply_transform)
        if args.iid:
            user_groups, user_groups_un = mnist_iid(train_dataset, args.num_users,label_rate)
        else:
            if args.dataset == 'mnist':
                user_groups, user_groups_un  = mnist_noniid(train_dataset, args.num_users,label_rate)
            if args.dataset == 'emnist':
                user_groups, user_groups_un  = emnist_noniid(train_dataset, args.num_users,62,label_rate)
    return train_dataset, test_dataset, user_groups, user_groups_un

def average_weights(w,aggregation_weights): 
    w_avg = copy.deepcopy(w[0]) 
    for key in w_avg.keys(): 
        w_avg[key] = torch.mul(w_avg[key],aggregation_weights[0])
        for i in range(1, len(w)): 
            w_avg[key] += torch.mul(w[i][key],aggregation_weights[i])  
    return w_avg

def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Projection layer  : {args.pro}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning Rate  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')
    
    print(f'    GPU  : {args.gpu}\n')
    
    print('    Federated parameters:')
    print(f'    Dataset  : {args.dataset}')
    print(f'    Method  : {args.method}')
    print(f'    localoss  : {args.localloss}')
    print(f'    ablation  : {args.abla}')
    
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
        print(f'    NonIID Degree   : {args.alpha}')
    if args.altype=='RCS':
        print('    RCS')
    else:
        print('    ACS')
    print(f'    Seed  : {args.seed}')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return
