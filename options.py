#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser() 

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=200, 
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=10,
                        help="number of clients: K")
    parser.add_argument('--frac', type=float, default=1, 
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=10, 
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=1e-3, # cifar100=1e-2; cifar10/mnist/emnist=1e-3
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='SGD weightdecay (default: 0.0005)')

    # model arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name') 
    parser.add_argument('--pro', type=str, default='True', help='projection layer') 
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to \
                        use for convolution')
    parser.add_argument('--num_channels', type=int, default=1, help="number of channels of imgs") # mnist/emnist=1, cifar10/cifar100=3
    parser.add_argument('--norm', type=str, default='batch_norm',
                        help="batch_norm, layer_norm, or None") 
    parser.add_argument('--num_filters', type=int, default=32, 
                        help="number of filters for conv nets -- 32 for \
                        mini-imagenet, 64 for omiglot.")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than \
                        strided convolutions")
    # dataset arguments
    parser.add_argument('--dataset', type=str, default='cifar', help="name of dataset") # 'cifar' 'cifar100' 'emnist'
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes") # emnist=62 ,cifar100 = 100
    parser.add_argument('--alpha', type=float, default=1e-2, help="cifar100 noniid degree")
    parser.add_argument('--iid', type=int, default=0, help='Default set to IID. Set to 0 for non-IID.')
    
    # training arguments
    parser.add_argument('--gpu', type=str, default="cuda:6", help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--optimizer', type=str, default='sgd', help="type of optimizer")
    parser.add_argument('--seed', type=int, default=1, help='random seed') 
    parser.add_argument('--verbose', type=int, default=1, help='verbose') 
    
    # FAL-CHASe arguments
    parser.add_argument('--method', type=str, default='chase-subset', help='al method') 
    parser.add_argument('--localloss', type=str, default='CE-CS', help='the loss of local training') # 'CE':only classification loss, 'CE-CS': + alignment loss
    parser.add_argument('--weightcs', type=float, default=1, help="alignment loss weight") # cifar10/cifar100=1, mnist=0.1, emnist=0.0/0.1
    parser.add_argument('--passive_rate', type=float, default=[0.20,0.60,0.20], help='passive rate')
    parser.add_argument('--w', type=int, default=2, help='the window size of shadow value')
    parser.add_argument('--Ta', type=float, default=0.4, help='the ratio of awake') 
    parser.add_argument('--set', type=str, default='True', help='subset sampling') 
    parser.add_argument('--abla', type=str, default='Tloc', help='ablation study') # 'Tloc','Tloc-ab'
    parser.add_argument('--des', type=str, default='SAL', help='ablation study')   #'S-AL' 'F-AL'
    parser.add_argument('--altype', type=str, default='RCS', help='AL Settings') # 'ACS' 'RCS'
    args = parser.parse_args()
    if args.altype == 'RCS':
        parser.add_argument('--anota_number', type=int, default=[5, 7, 10], help='annotation number') 
        parser.add_argument('--anota_cycle', type=int, default=[5,3,1], help='annotation cycle') 
        parser.add_argument('--anota_acc', type=float, default=[1,1,1], help='annotation accuracy')
    else:
       
        parser.add_argument('--anota_number', type=int, default=[10, 10, 10], help='annotation number') 
        parser.add_argument('--anota_cycle', type=int, default=[1,1,1], help='annotation cycle') 
        parser.add_argument('--anota_acc', type=int, default=[1,1,1], help='annotation accuracy')
    args = parser.parse_args()
    return args
