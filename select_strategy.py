from glob import glob
import random
import copy
import numpy as np
import torch
from localtraining import *

class CHASe_EV(object):
    def __init__(self, args):
        self.args = args
        self.sample_sum, self.sample_fluctuate, self.sample_preround, self.sample_metric, self.sample_square  = {}, {}, {}, {}, {}
        self.semi_mean, self.semi_fluctuate, self.shadow_list= {}, {}, {}
        self.window_size = args.w
    
    def normalization(self,data,name):
        _range = np.max(data) - np.min(data)
        if _range == 0 or np.isnan(np.max(data)) or np.isnan(np.min(data)):
            print(name,'var\'s range',_range, 'have 0, report warning')
            return data 
        else:
            return (data - np.min(data)) / _range

    def sample(self,idx,epoch,T_sample,N_sample,score_local,score_aggre,num_fluc,user_groups_un,idxs_tra,idxs_un,idxs_dormant,unlabel_index):
            if self.args.method == 'chase-subset':
                if epoch%T_sample == 0:
                    if self.args.abla[:4] == 'Tloc':
                        if self.args.abla[:5] == 'Tloc-':
                            a = self.normalization(num_fluc,'num_fluc')
                            glo_var =  score_aggre - score_local
                            b = self.normalization(glo_var,'glo_var')

                        
                        if self.args.abla== 'Tloc':
                            local_rank = np.argsort(-num_fluc)[:N_sample]
                        if self.args.abla == 'Tloc-ab':
                            local_rank = np.argsort(-(a+b))[:N_sample]
                 
                        sample_id = idxs_un[unlabel_index[local_rank]]
                        if self.args.Ta !=0:
                            zero_rank = np.where(num_fluc==0)
                            froze_id = idxs_un[unlabel_index[zero_rank]]
                            sample_froze_inter = set(sample_id)&set(froze_id)
                            if len(sample_froze_inter) != 0:
                                froze_id = [i for i in froze_id if i not in sample_id]
                            
                            idxs_dormant = np.array(list(set(idxs_dormant)|set(froze_id)))
                            idxs_un = list(set(idxs_un)-set(sample_id)-set(froze_id)) 
                        else:
                            idxs_un = list(set(idxs_un)-set(sample_id))
                        idxs_tra += list(sample_id)
                        idxs_un.sort(key=user_groups_un.index)
                        idxs_un = np.array(idxs_un)

                    
                if self.args.Ta != 0 and len(idxs_un) <= 3*N_sample and self.args.abla[:4] == 'Tloc':
                    awake_id = np.random.choice(idxs_dormant, int(len(idxs_dormant)*self.args.Ta),replace=False)
                    idxs_dormant = np.array(list(set(idxs_dormant)-set(awake_id)))
                    idxs_un = list(set(idxs_un)|set(awake_id))
                    idxs_un.sort(key=user_groups_un.index)
                    idxs_un = np.array(idxs_un)
                                
                return idxs_un, idxs_tra, idxs_dormant
