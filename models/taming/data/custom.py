import os
import numpy as np
import albumentations
from torch.utils.data import Dataset

from taming.data.base import ImagePaths, NumpyPaths, ConcatDatasetWithIndex

import taming.data.transforms as transforms
import os, glob, shutil

import ipdb

def get_transforms():
    GLOBAL_RANDOM_STATE = np.random.RandomState(47)
    seed = GLOBAL_RANDOM_STATE.randint(10000000)
    RandomState1=np.random.RandomState(seed)
    RandomState2=np.random.RandomState(seed)

    min_value=-1000
    max_value= 2000

    train_raw_transformer=transforms.Compose([
    # transforms.RandomFlip(RandomState1),
    # transforms.RandomRotate90(RandomState1),
    transforms.Normalize(min_value=min_value, max_value=max_value),
    transforms.ToTensor(expand_dims=False)
    ])
    train_label_transformer=transforms.Compose([
    # transforms.RandomFlip(RandomState2),
    # transforms.RandomRotate90(RandomState2),
    transforms.Normalize(min_value=min_value, max_value=max_value),
    transforms.ToTensor(expand_dims=False)
    ])

    val_raw_transformer=transforms.Compose([
    transforms.Normalize(min_value=min_value, max_value=max_value),
    transforms.ToTensor(expand_dims=False)
    ])

    val_label_transformer=transforms.Compose([
    transforms.Normalize(min_value=min_value, max_value=max_value),
    transforms.ToTensor(expand_dims=False)
    ])

    train_transforms=[train_raw_transformer,train_label_transformer]
    val_transforms=[val_raw_transformer,val_label_transformer]

    return train_transforms,val_transforms

def sorted_list(path): 
    tmplist = glob.glob(path) # finding all files or directories and listing them.
    tmplist.sort() # sorting the found list

    return tmplist


class CustomBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None
        self.transforms=None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = np.load(self.data[i]).astype(np.float32)
        #print(example.shape)
        example = self.transforms(example).permute(1,2,0)
        path=self.data[i].split('/')[-1]
        #example = self.data[i]
        return {'image':example,'file_path_':path}




class CustomTrain(CustomBase):
    def __init__(self, size, training_images_list_file):
        super().__init__()
        # with open(training_images_list_file, "r") as f:
        #     paths = f.read().splitlines()
        # self.data = ImagePaths(paths=paths, size=size, random_crop=False)
        self.transforms = get_transforms()[0][0]
        self.data=sorted_list('/data/zhchen/Mayo2016_2d/train/full_1mm/*')


class CustomTest(CustomBase):
    def __init__(self, size, test_images_list_file):
        super().__init__()
        # with open(test_images_list_file, "r") as f:
        #     paths = f.read().splitlines()
        # self.data = ImagePaths(paths=paths, size=size, random_crop=False)
        self.transforms = get_transforms()[1][0]
        self.data=sorted_list('/data/zhchen/Mayo2016_2d/test/full_1mm/*')



