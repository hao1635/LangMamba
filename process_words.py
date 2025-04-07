from tqdm import tqdm
import torch
import torch.nn.functional as F
import clip

import matplotlib.pyplot as plt
import ipdb
import os, glob, shutil
import os.path as osp
import numpy as np
from pydicom import dcmread
from PIL import Image

from transformers import CLIPModel,CLIPProcessor

import json


def sorted_list(path): 
    
    """ function for getting list of files or directories. """
    
    tmplist = glob.glob(path) # finding all files or directories and listing them.
    tmplist.sort() # sorting the found list
    
    return tmplist

def read_image(path):
    #ipdb.set_trace()
    CT=np.load(path).astype(np.float32)-1024
    CT=np.clip(CT,-160,240)
    CT=(CT+160)/400

    # CT=np.clip(CT,-1000,2000)
    # CT=(CT+1000)/3000

    # CT=np.clip(CT,-1350,150)
    # CT=(CT+1350)/1500

    ct_image=torch.from_numpy(CT).reshape(1,1,512,512)
    ct_image=ct_image.repeat(1,3,1,1)
    image = F.interpolate(ct_image, size=(224, 224), mode='bilinear', align_corners=False)
    return image

def compute_similarities(image,texts):

    similarities=[]
    #ipdb.set_trace()
    with torch.no_grad():
        image_features = model.get_image_features(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        for i,text in enumerate(tqdm(texts[::1000])):
            text=texts[1000*i:1000*i+1000]
            text = clip.tokenize(text).to(device)
            text_features = model.get_text_features(text)
            word_features = text_features / text_features.norm(dim=-1, keepdim=True)
            similarity=torch.matmul(image_features, word_features.T).cpu().tolist()
            similarities.extend(similarity[0])

    final_similarities=torch.tensor(similarities)
    #final_similarities=final_similarities.reshape(-1,1).squeeze()
    
    final_similarities=(final_similarities-final_similarities.min())/(final_similarities.max()-final_similarities.min())
    
    return final_similarities

if __name__ == '__main__':
    #image_list=sorted_list('/mnt/miah203/zhchen/Mayo2016_2d/train/full_1mm/*')
    image_list=sorted_list('/mnt/miah203/zhchen/Mayo2016_2d/train/quarter_1mm/*')

    device = "cuda:3" if torch.cuda.is_available() else "cpu"

    model = CLIPModel.from_pretrained("/mnt/miah203/zhchen/pubmed-clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("/mnt/miah203/zhchen/pubmed-clip-vit-base-patch32")

    model=model.to(device)

    tokenizer = processor.tokenizer

    # Accessing the tokenizer's vocabulary
    vocab = tokenizer.get_vocab()

    words=list(vocab.keys())

    print('len of voacab:',len(words))

    texts=['A CT of '+i for i in words]

    selected_token_ids={}

    thresholds=[0.95,0.9,0.8]

    for i,path in enumerate(tqdm(image_list[2400:])):
    #for i,path in enumerate(tqdm(image_list[:1200])):
    #for i,path in enumerate(tqdm(image_list[1200:2400])):
    #for i,path in enumerate(tqdm(image_list[2400:3600])):
    #for i,path in enumerate(tqdm(image_list[3600:])):
        
        image_id=path.split('/')[-1].split('.')[0]
        
        image=read_image(path).to(device)

        final_similarities=compute_similarities(image,texts)

        token_ids=[]
        for threshold in thresholds:
            selected_indices = torch.where(final_similarities >= threshold)[0]

            selected_words=[words[i] for i in selected_indices.cpu().tolist()]

            token_id = [vocab.get(token, -1) for token in selected_words]

            token_ids.append(token_id)
        
        #ipdb.set_trace()

        selected_token_ids[image_id]=token_ids

        print('image_id:{},len of select words:l1: {}, l2: {}, l3: {}:'.format(image_id,len(token_ids[0]),len(token_ids[1]),len(token_ids[2])))
        
    with open('selctd_words_ldcn_3l_2400_4800.json', 'w') as f:
        json.dump(selected_token_ids, f)