import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import einsum
from einops import rearrange
from transformers import BertModel, BertConfig
import ipdb

from transformers import BertTokenizer, BertModel
from transformers import CLIPModel,CLIPProcessor
from info_nce import InfoNCE, info_nce

import json


class VectorQuantizer_SPAE(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """
    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(self, n_e, e_dim, beta, remap=None, unknown_index="random",
                 sane_index_shape=False, legacy=True):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        model = CLIPModel.from_pretrained("/mnt/miah203/zhchen/pubmed-clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("/mnt/miah203/zhchen/pubmed-clip-vit-base-patch32")
        self.embedding=model.text_model.embeddings.token_embedding
        self.e_dim = self.embedding.weight.shape[-1]
        self.n_e = self.embedding.weight.shape[0]

        with open('/home/zhchen/selctd_words_all_3l.json', 'r') as fcc_file:
            self.selctd_words = json.load(fcc_file)
        #self.select_indices=torch.from_numpy(np.load('/home/zhchen/LLM_DN/taming-transformers-master/indice2.npy'))
        self.nceloss=InfoNCE(negative_mode='unpaired')

        self.embedding.requires_grad = False 

    def forward(self, z, path=None,temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        #

        b,c,h,w=z.shape
        z = rearrange(z, 'b c h w -> b h w c').contiguous()

        mask1 = torch.zeros((32, 32), dtype=torch.uint8).to(z.device) #第一层mask 下采样16倍
        mask1[7::16, 7::16] = 1

        mask2 = torch.zeros((32, 32), dtype=torch.uint8).to(z.device) #第二层mask 比例4倍
        mask2[3::4, 3::4] = 1

        mask3 = torch.ones((32, 32), dtype=torch.uint8).to(z.device)  #第三层mask 无下采样

        masks=[mask1,mask2,mask3] 

         #b,h,w,c

        semantic_loss=0
        commitment_loss=0
        z_q_final=0  #z^<l

        k_list=[2*2,8*8,32*32]
        
        layer=3
        
        z_l=z

        for l in range(layer):

            z_flattened = z_l.view(-1, self.e_dim)
            # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

            d = torch.sum(z_flattened** 2, dim=1, keepdim=True) + \
                torch.sum(self.embedding.weight**2, dim=1) - 2 * \
                torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))

            min_encoding_indices = torch.argmin(d, dim=1)
            z_q = self.embedding(min_encoding_indices).view(z.shape)

            #公式(3）
            z_q_final=z_q_final+ z_q*masks[l].to(z.dtype).unsqueeze(0).unsqueeze(-1)/(mask1+mask2+mask3).to(z.dtype).unsqueeze(0).unsqueeze(-1)
            commitment_loss =commitment_loss+ torch.mean((z_q_final.detach()-z*masks[l].to(z.dtype).unsqueeze(0).unsqueeze(-1))**2)

            if path is not None:
                for i,id in enumerate(path): # path是一个batch的特征
                    image_id=id.split('.')[0]

                    h,w,c=z_l[i].shape #z_l[i] h,w,c

                    if 'train' in image_id:
                        #ipdb.set_trace()
                        select_indices=self.selctd_words[image_id][l] # 第l层选的词的index
                        random_select_indices=random.choices(select_indices, k=k_list[l])
                        random_select_indices=torch.from_numpy(np.array(random_select_indices))
                        selected_embedding_weight=self.embedding.weight[random_select_indices] #1,hl*wl,512
                        
                        if l<2: #1,2层选词
                            non_zero_indices=torch.nonzero(masks[l].squeeze(), as_tuple=True)
                            z_l_mask=z_l[i][non_zero_indices[0], non_zero_indices[1], :] #hl*wl,512
                            d_l=d.view(b,h,w,-1)*masks[l].to(z.dtype).unsqueeze(0).unsqueeze(-1) #第l层的semantic loss分母 b,h,w,T
                        
                        if l==2: #最后一层选词
                            z_l_mask=z_l[i]
                            d_l=d

                        z_c=selected_embedding_weight.view(z_l_mask.shape) ## hl*wl,512
                        
                        #公式(7)
                        semantic_loss+=(-torch.log(torch.exp(-torch.mean((z_c - z_l_mask) ** 2))/torch.exp(-d_l).sum()))/(z.shape[0]*layer)
            else:
                semantic_loss=0
            #b,h,w,c 公式(2)
            z_l= z_l+(z-z_q)*masks[l].to(z.dtype).unsqueeze(0).unsqueeze(-1)
            #z_l = z+z_l_ #
            

        # preserve gradients
        z_q_final = z + (z_q_final - z).detach()

        # reshape back to match original input shape
        z_q_final = rearrange(z_q_final, 'b h w c -> b c h w').contiguous()

        perplexity = None
        min_encodings = None
        return z_q_final, (commitment_loss,semantic_loss), (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        if self.remap is not None:
            indices = indices.reshape(shape[0],-1) # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1) # flatten again

        # get quantized latent vectors
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q
