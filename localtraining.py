#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import numpy as np
from numpy.core.numeric import indices
import torch,copy,math,time
from torch import nn
from torch.utils.data import DataLoader, Dataset

class DatasetSplit(Dataset):

    def __init__(self, dataset, idxs, cl_id, args):
        self.args = args
        self.cl_id = cl_id 
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self): 
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        
        return torch.as_tensor(image), torch.as_tensor(label)
        
class LocalUpdate(object):
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.device = torch.device(args.gpu if torch.cuda.is_available() else "cpu") 
        self.criterion = nn.NLLLoss()
        self.contrastive_criterion = nn.CrossEntropyLoss().to(self.args.gpu)
        self.cos =  nn.CosineSimilarity(dim=-1)
            
    def update_weights(self, id, model, idxs_tra, idxs_un, dataset, globalmodel=None,old_weight=None,var_labels=None,round=None):
        if self.args.optimizer == 'sgd': 
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        if self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=0)
            
        if self.args.method == 'chase-subset':
            localvar_time = 0
            local_fluc = [[]for i in range(self.args.local_ep)]
            scores = [[]for i in range(self.args.local_ep)]
            fluc_number = np.array([0 for i in range(len(idxs_un))])
        
        if self.args.localloss =='CE-CS' and round>0:
            oldmodel = copy.deepcopy(model)
            oldmodel.load_state_dict(old_weight)
            
            model.train()
            globalmodel.eval()
            oldmodel.eval()
            
            fake_labels = fluc_number.copy()
            fake_labels_mean = np.mean(fake_labels)
            fake_labels[fluc_number>=fake_labels_mean] = 0
            fake_labels[fluc_number<fake_labels_mean] = 1
            fake_labels = torch.tensor(fake_labels)
        
        epoch_loss = []
        trainloader = DataLoader(DatasetSplit(dataset, idxs_tra, id, self.args), batch_size=self.args.local_bs, pin_memory=False, shuffle=True)
        for iter in range(self.args.local_ep):
            batch_loss,batch_contraloss,batch_loss12  = [],[],[]
            model.train()

            if round == 0 or round == None or self.args.localloss == 'CE' or len(idxs_un)<=self.args.local_bs :
                for _, (images, labels) in enumerate(trainloader):
                    images, labels = images.to(self.device), labels.to(self.device)
                    log_probs = model(images, return_all_layers = False) 
                    loss = self.criterion(log_probs, labels)
                    
                    model.zero_grad()
                    loss.backward()
                    optimizer.step()

                    batch_loss.append(loss.item())
                epoch_loss.append(sum(batch_loss)/len(batch_loss))
                print('Client',id,iter,'epoch的loss:',epoch_loss[-1])
            else:
                print('Calibrate DB')
                for _, (images, labels) in enumerate(trainloader):
                    images, labels = images.to(self.device), labels.to(self.device)
                    # classification loss
                    log_probs = model(images, return_all_layers = False) 
                    loss1 = self.criterion(log_probs, labels)
                    
                    random_idx = np.random.choice(len(idxs_un), size=images.shape[0], replace=False)
                    batch_un_idx = idxs_un[random_idx]
                    unlabeldataloader = DataLoader(DatasetSplit(dataset, batch_un_idx, id, self.args), batch_size=len(batch_un_idx), pin_memory=False, shuffle=False)
                    for _, (images_un,_) in enumerate(unlabeldataloader):
                        labels_un = fake_labels[random_idx]
                        images_un, labels_un = images_un.to(self.device), labels_un.to(self.device)
                    
                        # alignment loss
                        pro1,_ = model(images_un, return_all_layers = True)
                        pro2,_ = globalmodel(images_un, return_all_layers = True)
                        pro3,_ = oldmodel(images_un, return_all_layers = True)

                        cos2 = self.cos(pro1,pro2).reshape(-1,1)
                        cos3 = self.cos(pro1,pro3).reshape(-1,1)
                        logits = torch.cat((cos2, cos3), dim=1)
                        logits /= 0.5 # temperature \tau =0.5

                        loss2 = self.contrastive_criterion(logits, labels_un)
                    
                    loss = loss1 + self.args.weightcs*loss2
                    
                    model.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    batch_loss12.append(loss.item())
                    batch_loss.append(loss1.item())
                    batch_contraloss.append(loss2.item())
                epoch_loss.append(sum(batch_loss)/len(batch_loss))
                print('Client',id,iter,'total loss of epoch:',np.mean(batch_loss12),'classfication  loss:',epoch_loss[-1],'alignment loss:',np.mean(batch_contraloss))
            
            if self.args.method == 'chase-subset':
                localvar_start = time.time()
                if (self.args.abla[:4] == 'Tglo') and (iter+1) == self.args.local_ep:
                    scores[iter],local_fluc[iter] = self.unlabel_inference(id, model, idxs_un, dataset,scores[iter],local_fluc[iter])
                if self.args.abla[:4] == 'Tloc':
                    scores[iter],local_fluc[iter] = self.unlabel_inference(id, model, idxs_un, dataset,scores[iter],local_fluc[iter])
                localvar_end = time.time()
                localvar_time += localvar_end - localvar_start
        local_loss = sum(epoch_loss) / len(epoch_loss) 
        
        if self.args.method == 'chase-subset':
            if self.args.abla[:4] != 'Tglo':
                local_fluc=np.array(local_fluc).T
                for i in range(len(idxs_un)):
                    for j in range(self.args.local_ep):
                        if j > 0 and local_fluc[i][j] != local_fluc[i][j-1]:
                            fluc_number[i] = fluc_number[i]+1
            return model.state_dict(), local_loss, fluc_number, scores[-1], localvar_time
    
    def unlabel_inference(self, id, model,un_idxs, dataset, scores=None, num_fluc=None):
        unlabeldataloader = DataLoader(DatasetSplit(dataset, un_idxs, id, self.args), 
                                        batch_size=128, pin_memory=False, shuffle=False)
        model.eval()
        for _, (images, _) in enumerate(unlabeldataloader):
            images = images.to(self.device)
            log_probs = model(images, return_all_layers=False)
            probs = torch.exp(log_probs)
            probs[torch.where(probs==0)] = min(probs[torch.where(probs!=0)]) 
            max_prob_idx = torch.max(log_probs,dim=1).indices
            if scores != None: scores += list((np.array(-(probs*torch.log2(probs)).sum(1).cpu().detach())))
            if num_fluc != None: num_fluc += max_prob_idx.cpu().numpy().tolist()
        return np.array(scores), np.array(num_fluc)
    
    def inference(self, id, model, dataset, idxs_test):
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        testloader = DataLoader(DatasetSplit(dataset, idxs_test, id, self.args), batch_size=int(len(idxs_test)/10), shuffle=False)
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(self.device), labels.to(self.device)
            log_probs = model(images, return_all_layers = False)
            batch_loss = self.criterion(log_probs, labels)
            loss += batch_loss.item()
            _, pred_labels = torch.max(log_probs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels) 
        loss = loss/batch_idx
        accuracy = correct/total 
        return accuracy, loss
    
def test_inference(args, model, test_dataset):
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")
    criterion = nn.NLLLoss() 
    testloader = DataLoader(test_dataset, batch_size=128, shuffle=False) 
    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)
        log_probs = model(images, return_all_layers = False)
        batch_loss = criterion(log_probs, labels)
        loss += batch_loss.item()
        _, pred_labels = torch.max(log_probs, 1) 
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)
    loss = loss/batch_idx
    accuracy = correct/total
    return accuracy, loss
