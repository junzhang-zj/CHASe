#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import numpy as np
import json
import os
from torchvision import datasets, transforms

def get_dirs(main_dir):
    list_dirs = []
    for root, dirs, files in os.walk(main_dir):
        for file in files:
            list_dirs.append(os.path.join(root, file))
    return list_dirs

def mnist_iid(dataset, num_users,label_rate):
    num_items = int(len(dataset)/num_users)
    num_label = int(label_rate*num_items) 
    num_unlabel = int((1-label_rate)*num_items)
    users_label, users_unlabel, all_idxs = {}, {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        if label_rate<1:
            users_label[i] = set(np.random.choice(all_idxs, num_label,replace=False))
            all_idxs = list(set(all_idxs) - users_label[i])
            users_unlabel[i] = set(np.random.choice(all_idxs, num_unlabel, replace=False))
            all_idxs = list(set(all_idxs) - users_unlabel[i])
        else: 
            users_label[i] = set(np.random.choice(all_idxs, num_label,replace=False))

    if label_rate<1:
        return users_label, users_unlabel
    else:
        return users_label

def cifar_iid(dataset, num_users, label_rate):
    num_items = int(len(dataset)/num_users)
    num_label = int(label_rate*num_items) 
    num_unlabel = int((1-label_rate)*num_items)
    users_label, users_unlabel, all_idxs = {}, {}, [i for i in range(len(dataset))] 
    for i in range(num_users): 
        if label_rate<1:
            users_label[i] = set(np.random.choice(all_idxs, num_label,replace=False))
            all_idxs = list(set(all_idxs) - users_label[i])
            users_unlabel[i] = set(np.random.choice(all_idxs, num_unlabel,replace=False))
            all_idxs = list(set(all_idxs) -users_unlabel[i])
        else:
            users_label[i] = set(np.random.choice(all_idxs, num_label,replace=False))
    if label_rate<1:
        return users_label, users_unlabel
    else:
        return users_label

def mnist_noniid(dataset, num_users, label_rate):
    noniid_degree = 2
    num_imgs = int((len(dataset)/num_users)/noniid_degree) 
    num_shards = int(num_users*noniid_degree) 
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    delta_shards = int(num_shards/10)
    labels = dataset.targets.numpy()
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    users_label, users_unlabel= {}, {}

    for i in range(num_users):
    
        rand_class = np.random.choice(10, noniid_degree, replace=False) 
        rand_delta = np.random.choice(delta_shards, noniid_degree)
        rand_set = set(rand_class*delta_shards + rand_delta)
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0).astype(int)
        if label_rate<1:
            users_label[i] = np.random.choice(dict_users[i], int(len(dict_users[i])*label_rate), replace=False)
            users_unlabel[i] = set(dict_users[i]) - set(users_label[i])
    
    if label_rate<1:
        return users_label, users_unlabel
    else:
        return dict_users

def emnist_noniid(dataset, num_users,num_classes, label_rate):
    noniid_degree = 10
    num_imgs = int((len(dataset)/num_users)/noniid_degree) 
    num_shards = int(num_users*noniid_degree) 
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    delta_shards = int(num_shards/num_classes)
    labels = dataset.targets.numpy()

    idxs_labels = np.vstack((idxs, labels[0:len(idxs)]))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    users_label, users_unlabel= {}, {}

    for i in range(num_users):
        rand_class = np.random.choice(num_classes, noniid_degree, replace=False) 
        rand_delta = np.random.choice(delta_shards, noniid_degree)
        rand_set = set(rand_class*delta_shards + rand_delta)
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0).astype(int)
        if label_rate<1:
            users_label[i] = np.random.choice(dict_users[i], int(len(dict_users[i])*label_rate), replace=False)
            users_unlabel[i] = set(dict_users[i]) - set(users_label[i])
    
    if label_rate<1:
        return users_label, users_unlabel
    else:
        return dict_users

def cifar_noniid(dataset, num_users, num_classes, label_rate):
  
    noniid_degree = 5
    num_imgs = int((len(dataset)/num_users)/noniid_degree) 
    num_shards = int(num_users*noniid_degree) 
    delta_shards = int(num_shards/num_classes) 

    idx_shard = [i for i in range(num_shards)] 
    dict_users = {i: np.array([]) for i in range(num_users)} 
    idxs = np.arange(num_shards*num_imgs) 
    labels = np.array(dataset.targets)

    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    users_label, users_unlabel= {}, {}
    for i in range(num_users):
        rand_class = np.random.choice(num_classes, noniid_degree, replace=False)
        rand_delta = np.random.choice(delta_shards, noniid_degree)
        rand_set = set(rand_class*delta_shards + rand_delta)
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0).astype(int)
        if label_rate<1:
            users_label[i] = np.random.choice(dict_users[i], int(len(dict_users[i])*label_rate), replace=False)
            users_unlabel[i] = set(dict_users[i]) - set(users_label[i])
    
    if label_rate<1:
        return users_label, users_unlabel
    else:
        return dict_users

def cifar100_noniid(dataset,num_users,num_classes,label_rate,alpha): 
    y_train = np.array(dataset.targets)
    min_size = 0
    K = num_classes
    N = len(y_train) 
    net_dataidx_map = {} 
    users_label, users_unlabel= {}, {}
    while min_size < 10:
        idx_batch = [[] for _ in range(num_users)] 
        for k in range(K):
            idx_k = np.where(y_train == k)[0] 
            np.random.shuffle(idx_k) 
            proportions = np.random.dirichlet(np.repeat(alpha, num_users))
            proportions = np.array([p * (len(idx_j) < N / num_users) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(num_users):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]
        if label_rate<1:
                users_label[j] = np.random.choice(net_dataidx_map[j], int(len(net_dataidx_map[j])*label_rate), replace=False)
                users_unlabel[j] = set(net_dataidx_map[j]) - set(users_label[j])

    if label_rate<1:
        return users_label, users_unlabel
    else:
        return net_dataidx_map
