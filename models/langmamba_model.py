"""Model class template

This module provides a template for users to implement custom models.
You can specify '--model template' to use this model.
The class name should be consistent with both the filename and its model option.
The filename should be <model>_dataset.py
The class name should be <Model>Dataset.py
It implements a simple image-to-image translation baseline based on regression loss.
Given input-output pairs (data_A, data_B), it learns a network netG that can minimize the following L1 loss:
    min_<netG> ||netG(data_A) - data_B||_1
You need to implement the following functions:
    <modify_commandline_options>:　Add model-specific options and rewrite default values for existing options.
    <__init__>: Initialize this model class.
    <set_input>: Unpack input data and perform data pre-processing.
    <forward>: Run forward pass. This will be called by both <optimize_parameters> and <test>.
    <optimize_parameters>: Update network weights; it will be called in every training iteration.
"""
import torch
from .base_model import BaseModel
from . import networks3
import os
import numpy as np
import torch.nn as nn
import util.util as util
import ipdb
from .VL_loss import vqgan_loss
import torchvision.models as models
from .emamba import SS2D
from .seed import SEED




class LANGMAMBAModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new model-specific options and rewrite default values for existing options.

        Parameters:
            parser -- the option parser
            is_train -- if it is training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        #parser.set_defaults(dataset_mode='aligned')  # You can rewrite default values for this model. For example, this model usually uses aligned dataset as its dataset.
        if is_train:
            parser.add_argument('--lambda_regression', type=float, default=1.0, help='weight for the regression loss')  # You can define new arguments for this model.

        return parser

    def __init__(self, opt):
        """Initialize this model class.

        Parameters:
            opt -- training/test options

        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, visualization images, model names, and optimizers
        """
        BaseModel.__init__(self, opt)  # call the initialization method of BaseModel
        # specify the training losses you want to print out. The program will call base_model.get_current_losses to plot the losses to the console and save them to the disk.
        self.loss_names = ['loss_G']
        # specify the images you want to save and display. The program will call base_model.get_current_visuals to save and display these images.
        self.visual_names = ['data_A', 'data_B', 'output']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks to save and load networks.
        # you can use opt.isTrain to specify different behaviors for training and test. For example, some networks will not be used during test, and you don't need to load them.
        self.model_names = ['G']
        # define networks; you can use opt.isTrain to specify different behaviors for training and test.


        self.vision_loss= vqgan_loss().to(self.device) 
        model=SEED(encoder=self.vision_loss.vqgan.encoder)
        self.netG = networks3.init_net(model,gpu_ids=opt.gpu_ids,initialize_weights=True).to(self.device)

        if self.isTrain:  # only defined during training time
            # define your loss functions. You can use losses provided by torch.nn such as torch.nn.L1Loss.
            # We also provide a GANLoss class "networks.GANLoss". self.criterionGAN = networks.GANLoss().to(self.device)
            self.criterionLoss = torch.nn.MSELoss()
            # define and initialize optimizers. You can define one optimizer for each network.
            # If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
            self.optimizer = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = [self.optimizer]

        # Our program will automatically call <model.setup> to define schedulers, load networks, and print networks
        self.phase=opt.phase
        self.loss_D,self.loss_G=0,0

        


    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        AtoB = self.opt.direction == 'AtoB'  # use <direction> to swap data_A and data_B
        self.data_A = input['A' if AtoB else 'B'].to(self.device)  # get image data A
        self.data_B = input['B' if AtoB else 'A'].to(self.device)  # get image data B
        self.image_paths = input['A_paths' if AtoB else 'B_paths']  # get image paths
        #ipdb.set_trace()

    def forward(self):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        #ipdb.set_trace()
        self.output = self.netG(self.data_A)  # generate output image given the input data_A


    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # caculate the intermediate results if necessary; here self.output has been computed during function <forward>
        #ipdb.set_trace()
        loss_vision=self.vision_loss(self.output, self.data_B)
        self.loss_D = loss_vision
        self.loss_G =self.criterionLoss(self.output, self.data_B)+ self.loss_D
        #self.loss_G =self.criterionLoss(self.output, self.data_B)
        
        
        self.loss_G.backward()       # calculate gradients of network G w.r.t. loss_G

    def optimize_parameters(self):
        """Update network weights; it will be called in every training iteration."""
        self.forward()               # first call forward to calculate intermediate results
        self.optimizer.zero_grad()   # clear network G's existing gradients
        self.backward()              # calculate gradients for network G
        self.optimizer.step()        # update gradients for network G
        #xipdb.set_trace()

    def compute_metrics(self):
        with torch.no_grad():
            y_pred=self.output
            y=self.data_B 
            #ipdb.set_trace()

            y=torch.clip(y*3000-1000,-160,240)
            y=(y+160)/400
            y_pred=torch.clip(y_pred*3000-1000,-160,240)
            y_pred=(y_pred+160)/400


            psnr=util.compute_psnr(y_pred,y)
            ssim=util.compute_ssim(y_pred,y)
            rmse=util.compute_rmse(y_pred,y)

            if self.phase == 'test':
                return psnr,ssim,rmse

            if 'train' in self.phase:
                return self.loss_D,self.loss_G,psnr,ssim,rmse


