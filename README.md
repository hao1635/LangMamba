# LangMamba
Official implementation of "LLangMamba: A Language-driven Mamba Framework for Low-dose CT Denoising with Vision-language Models" 


## Updates
March, 2025: initial commit.  


## Data Preparation
The 2016 AAPM-Mayo dataset can be downloaded from: [CT Clinical Innovation Center](https://ctcicblog.mayo.edu/2016-low-dose-ct-grand-challenge/) (B30 kernel)  
The 2020 AAPM-Mayo dataset can be downloaded from: [cancer imaging archive](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=52758026)   
#### Dataset structre:
```
Mayo2016_2d/
  |--train/
      |--quarter_1mm/
        train_quarter_00001.npy
        train_quarter_00002.npy
        train_quarter_00003.npy
        ...
      |--full_1mm/
        train_full_00001.npy
        train_full_00002.npy
        train_full_00003.npy
        ...
  |--test/
      |--quarter_1mm
      |--full_1mm
```

## Requirements
```
- Linux Platform
- torch==1.12.1+cu113 # depends on the CUDA version of your machine
- torchvision==0.13.1+cu113
- Python==3.8.0
- numpy==1.22.3
```

## Traning and & Inference

#### Training of LLM-guided NDCT autoencoder:      
Build per-layer candidate token:```python process_words.py```  
Then we used the official repository of VQ-GAN (https://github.com/CompVis/taming-transformers) to set up training. Please refer to (models/taming) to learn about our modifications to original VQ-GAN.

#### Employment of LEDA for training the denoising model:  
```
python train.py  --name LangMamba(experiment_name)   --model langmamba  --dataroot /data/zhchen/Mayo2016_2d(path to images) --lr 0.0002 --gpu_ids 6,7 --print_freq 25 --batch_size 8 --lr_policy cosine
```

#### Inference & testing:
```
python test.py  --name LangMamba(experiment_name)   --model langmamba --results_dir test_results --result_name test_results(path to save image)   --gpu_ids 6 --batch_size 1 --eval
```
Please refer to options files for more setting.


