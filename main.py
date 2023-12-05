import copy
import random, math
import time
import pickle
import numpy as np
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
plt.switch_backend("agg")

from operator import itemgetter

import os, sys
sys.path.append(os.path.dirname(__file__))
sys.path.append("")

import torch
from torch import nn

from tensorboardX import SummaryWriter

from options import args_parser
from localtraining import LocalUpdate, test_inference
from models import *
from utils import get_dataset, average_weights, exp_details
from select_strategy import CHASe_EV

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def init_model(model):
    for name, w in model.named_parameters():
        if name.endswith("weight"):
            nn.init.xavier_normal_(w, gain=1)
        else:
            nn.init.constant_(w, 0.01)
    model.to(device)

def text_save(filename, data):
    file = open(filename,'a')
    for i in range(len(data)):
        s = str(data[i]).replace('[','').replace(']','')
        s = s.replace("'",'').replace(',','') +'\n'
        file.write(s)
    file.close()
    print("Save File")

def plot(filename, xlabel, ylabel, data):
    plt.figure()
    plt.plot(range(len(data)), data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig('./save/figure/{}_{}.png'.format(sample_strategy, filename))

if __name__ == '__main__':
    start_time = time.time()
    path_project = os.path.abspath('..') 
    logger = SummaryWriter('./logs')
    args = args_parser()
    exp_details(args)
    device = torch.device(args.gpu if torch.cuda.is_available() else "cpu") 
    setup_seed(args.seed)

    train_dataset, test_dataset, user_groups, user_groups_un = get_dataset(args)
    local_anotnumber = [[] for _ in range(len(user_groups.keys()))]
    local_anotcycle = copy.deepcopy(local_anotnumber)
    local_anotaccur = copy.deepcopy(local_anotnumber)
    idxs_tra, idxs_val, idxs_test = {},{},{}
    client_idx = np.array(list(user_groups.keys()))
    client_hier = (np.array(args.num_users)*args.passive_rate).astype(int)
    passive_client = np.random.choice(client_idx, client_hier[0], replace=False)
    client_idx = np.setxor1d(client_idx, passive_client)
    ordinary_client = np.random.choice(client_idx, client_hier[1], replace=False)
    active_client = np.setxor1d(client_idx, ordinary_client)
    mask_al, mask_roll,mask_gloinfer = {},{},{}

    for idx in (user_groups.keys()):
        user_groups[idx] = list(user_groups[idx])
        user_groups_un[idx] = list(user_groups_un[idx])
        mask_al[idx] = np.ones(len(user_groups_un[idx]),dtype=bool)
        mask_roll[idx] = np.ones(len(user_groups_un[idx]),dtype=bool)

        idxs_tra[idx] = user_groups[idx][:int(len(user_groups[idx])-1000)]
        idxs_test[idx] = user_groups[idx][-1000:]
  
        if idx in passive_client:
            local_anotnumber[idx] = args.anota_number[0]
            local_anotcycle[idx] = args.anota_cycle[0]
            local_anotaccur[idx] = args.anota_acc[0]
        elif idx in ordinary_client:
            local_anotnumber[idx] = args.anota_number[1]
            local_anotcycle[idx] = args.anota_cycle[1]
            local_anotaccur[idx] = args.anota_acc[1]
        else:
            local_anotnumber[idx] = args.anota_number[2]
            local_anotcycle[idx] = args.anota_cycle[2]
            local_anotaccur[idx] = args.anota_acc[2]

    if args.model == 'cnn':
        if args.dataset == 'mnist' or args.dataset == 'emnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)
        elif args.dataset == 'cifar100':
            global_model = VGG19(args=args)
    else:
        exit('Error: unrecognized model')
    
    if args.dataset == 'cifar100':
        global_model.to(device)
    else:
        init_model(global_model) 
            
    global_model.train()
    global_weights = global_model.state_dict()
    print(global_model)

    tra_loss_list, tra_acc_list, test_acc_list, test_loss_list, local_tra_acc, local_tra_loss, var_list = [], [], [], [], [], [], []
    score_aggre, score_local, num_lofl, idxs_un, npr_un, unlabel_index, idxs_dormant = {},{},{},{},{},{},{}
    print_every,sample_time,inference_time,localvar_times = 1,0,0,0
    local_var_form = [{} for i in range(args.num_users)]
    old_weights = {}
    
    if args.method == 'chase-subset':
        AL = CHASe_EV(args=args)
        sample_strategy = '{}500_{}_{}_IID{}{}_Seed{}_Round{}_Abla{}_LocalLoss{}_CSweight{}_BS{}_Des{}_NoFAM'.format(args.method, args.dataset, args.altype, args.iid,args.alpha, args.seed, args.epochs,args.abla,args.localloss,args.weightcs,args.local_bs,args.des)
        
    for epoch in trange(args.epochs): 
        print(f'\n | Global Training Round : {epoch} |\n') 
        
        local_weights, local_losses, aggregation_weights = [], [], []
        sum_selectdata = 0
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        
        global_model.train()
        
        # local training
        for idx in idxs_users: 
            local_model = LocalUpdate(args=args, logger=logger)
            localmodel = copy.deepcopy(global_model)
    
            if args.method == 'chase-subset':
                if epoch==0: 
                    idxs_un[idx] = np.array(user_groups_un[idx])
                    idxs_dormant[idx] = np.array([],dtype=int)
                # Subset sampling
                if len(idxs_un[idx])>500 and epoch > 0 and args.set == 'True':
                    unlabel_index[idx] = np.random.choice(np.arange(len(idxs_un[idx])), size=500, replace=False)
                else:
                    unlabel_index[idx] = np.arange(len(idxs_un[idx]))
                
                if args.localloss == 'CE-CS':
                    if epoch==0: 
                        old_weights[idx] = copy.deepcopy(global_weights)
                        var_labels = None
                    else:
                        var_labels = np.array(itemgetter(*idxs_un[idx][unlabel_index[idx]])(local_var_form[idx])) 
                    w,loss,num_lofl[idx],score_local[idx],localvar_time = local_model.update_weights(idx, localmodel, idxs_tra[idx], idxs_un[idx][unlabel_index[idx]], train_dataset,globalmodel=global_model,old_weight=old_weights[idx],var_labels=var_labels,round=epoch)
                    old_weights[idx] = copy.deepcopy(w)
                    if epoch==0: 
                        local_var_form[idx] = dict(zip(idxs_un[idx][unlabel_index[idx]],num_lofl[idx]))
                    else:
                        for k,item in enumerate(idxs_un[idx][unlabel_index[idx]]):
                            local_var_form[idx][item] = num_lofl[idx][k]
    
                if args.localloss == 'CE':
                    if epoch==0: var_labels = None
                    else:
                        var_labels = np.array(itemgetter(*idxs_un[idx][unlabel_index[idx]])(local_var_form[idx]))

                    w,loss,num_lofl[idx],score_local[idx],localvar_time = local_model.update_weights(idx, localmodel, idxs_tra[idx], idxs_un[idx][unlabel_index[idx]], train_dataset)
                    if epoch==0: 
                        local_var_form[idx] = dict(zip(idxs_un[idx][unlabel_index[idx]],num_lofl[idx]))
                    else:
                        for k,item in enumerate(idxs_un[idx][unlabel_index[idx]]):
                            local_var_form[idx][item] = num_lofl[idx][k]
                            
                localvar_times +=  localvar_time
                print("local training time ：", localvar_times)
            
            # save local updates 
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            if  args.method == 'chase-subset':
                npr_un.setdefault(idx,[]).append(len(idxs_un[idx]))
                npr_un.setdefault(idx,[]).append(len(user_groups_un[idx]))
        loss_avg = sum(local_losses) / len(local_losses)  
        tra_loss_list.append(loss_avg)
        
        # global aggregation
        sum_selectdata = sum(len(item) for item in idxs_tra.values()) 
        aggregation_weights = [len(item)/sum_selectdata for item in idxs_tra.values()]
        global_weights = average_weights(local_weights, aggregation_weights)
        global_model.load_state_dict(global_weights)

        if args.method == 'chase-subset':
            sample_start = time.time()
            for idx in idxs_users:
                score_aggre[idx],_ = local_model.unlabel_inference(idx, global_model,idxs_un[idx][unlabel_index[idx]],train_dataset,scores=[]) 
                idxs_un[idx],idxs_tra[idx],idxs_dormant[idx] = AL.sample(idx,epoch,local_anotcycle[idx],local_anotnumber[idx],\
                                score_local[idx],score_aggre[idx],num_lofl[idx],user_groups_un[idx],idxs_tra[idx],idxs_un[idx],idxs_dormant[idx],unlabel_index[idx])
                sample_end = time.time()  
                sample_time += sample_end-sample_start
                print("The sampling time of chase:",sample_time)
        list_acc, list_loss, list_tra_acc, list_tra_loss = [], [], [], []
       
        global_model.eval()
        for idx in idxs_users:
            local_model = LocalUpdate(args=args, logger=logger)
            acc, loss = local_model.inference(idx, global_model,train_dataset, idxs_test[idx]) 
            tra_acc, tra_loss = local_model.inference(idx,global_model,train_dataset, idxs_tra[idx]) 
            list_acc.append(acc)
            list_loss.append(loss)
            list_tra_acc.append(tra_acc) 
            list_tra_loss.append(tra_loss)
        local_tra_acc.append(sum(list_tra_acc)/len(idxs_users))
        local_tra_loss.append(sum(list_tra_loss)/len(idxs_users))
        tra_acc_list.append(sum(list_acc)/len(list_acc))

        # global accuracy
        test_acc, test_loss = test_inference(args, global_model, test_dataset)
        test_acc_list.append(test_acc)
        test_loss_list.append(test_loss)
        
        if (epoch+1) % print_every == 0: 
            print(f' \n Results after {epoch+1} global rounds of training:')
            print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))
            print("|---- local Test Accuracy: {:.2f}%".format(100*tra_acc_list[-1]))
            print("|---- Avg Train Accuracy: {:.2f}%".format(100*local_tra_acc[-1]))
            print('\n Run Time: {0:0.4f}'.format(time.time()-start_time)) 
            if (epoch+1) == args.epochs: print("Testing End")

    file_name_metric = './save/Rebuttal/Peform{}.pkl'.format(sample_strategy)  
    with open(file_name_metric, 'wb') as f:
        pickle.dump([tra_loss_list, tra_acc_list,test_acc_list], f) 
    
    fam_val_file = './save/Rebuttal/npr{}.pkl'.format(sample_strategy)
    with open(fam_val_file, 'wb') as f:
        pickle.dump([npr_un,var_list], f)
    
    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))
    plot('Test loss','epochs', 'Test loss', test_loss_list)
    plot('Test accuracy','epochs', 'Test accuracy', test_acc_list)
    plot('Train loss','epochs', 'Train loss', tra_loss_list)
    plot('Train acccuracy','epochs', 'Train acc', tra_acc_list)
    plot('Local Tra loss','epochs', 'local train loss', local_tra_loss)
    plot('Local Tra acc','epochs', 'local train acc', local_tra_acc)