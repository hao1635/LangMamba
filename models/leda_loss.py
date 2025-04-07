import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
import ipdb
import ruamel.yaml as yaml
from transformers import AutoTokenizer, AutoModel
from gensim.models import KeyedVectors
from omegaconf import OmegaConf
import importlib

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    #ipdb.set_trace()
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


class LEDA_loss(nn.Module):
    def __init__ (self):
        super(LEDA_loss,self).__init__()
        
        configs = [OmegaConf.load('models/taming/custom_vqgan.yaml')]
        unknown=[]
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli)

        self.vqgan= instantiate_from_config(config.model)
        self.vqgan.init_from_ckpt('/mnt/miah203/zhchen/vq_gan_result/2024-02-19T02-22-27_vqgan_spae_3l_2020/testtube/version_None/checkpoints/epoch=100-step=161600.ckpt') # 
        

        for p in self.vqgan.parameters():
            p.requires_grad = False

    def forward(self,pred,target):

        pred_quant,_,_,pred_h = self.vqgan.encode(pred)
        target_quant,_,_,target_h= self.vqgan.encode(target)  

        #ipdb.set_trace()   

        loss=F.mse_loss(pred_quant,target_quant)+F.mse_loss(pred_h,target_h)

        return loss
    
    def vqgan_test(self,pred):
        recon,diff = self.vqgan(pred)
        # return recon
        quant, emb_loss, info = self.vqgan.encode(pred)
        return recon,info


