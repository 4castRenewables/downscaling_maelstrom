
# SPDX-FileCopyrightText: 2021 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)

# SPDX-License-Identifier: MIT

__email__ = "b.gong@fz-juelich.de"
__author__ = "Bing Gong"
__date__ = "2022-07-22"

import time
import argparse
import sys
import os
import json
import torch
from torch.utils.data import DataLoader
from collections import OrderedDict
from torch.optim import lr_scheduler
from torch.optim import Adam
import torch.nn as nn
from dataset_prep import PrecipDatasetInter
sys.path.append('../')
from models.network_unet import UNet as net
import wandb
wandb.init(project="Precip_downscaling")


def create_loader(file_path: str = None, batch_size: int = 4, patch_size: int = 16,
                 vars_in: list = ["cape_in", "tclw_in", "sp_in", "tcwv_in", "lsp_in", "cp_in", "tisr_in",
                                  "yw_hourly_in"],
                 var_out: list = ["yw_hourly_tar"], sf: int = 10,
                 seed: int = 1234):

    """
    file_path : the path to the directory of .nc files
    vars_in   : the list contains the input variable namsaes
    var_out   : the list contains the output variable name
    batch_size: the number of samples per iteration
    patch_size: the patch size for low-resolution image,
                the corresponding high-resolution patch size should be muliply by scale factor (sf)
    sf        : the scaling factor from low-resolution to high-resolution
    seed      : specify a seed so that we can generate the same random index for shuffle function
    """

    dataset = PrecipDatasetInter(file_path, batch_size, patch_size, vars_in, var_out, sf, seed)

    return torch.utils.data.DataLoader(dataset, batch_size=None)



class BuildModel:
    def __init__(self, netG, G_lossfn_type: str = "l2", G_optimizer_type: str = "adam",
                 G_optimizer_lr: float = 0.0002, G_optimizer_betas: list = [0.9, 0.999],
                 G_optimizer_wd: int= 0, save_dir: str = "../results"):

        # ------------------------------------
        # define network
        # ------------------------------------
        self.netG = netG
        self.G_lossfn_type = G_lossfn_type
        self.G_optimizer_type = G_optimizer_type
        self.G_optimizer_lr = G_optimizer_lr
        self.G_optimizer_betas = G_optimizer_betas
        self.G_optimizer_wd = G_optimizer_wd
        self.schedulers = []
        self.save_dir = save_dir

    def init_train(self):
        wandb.watch(self.netG, log_freq=100)
        self.netG.train()
        self.define_loss()
        self.define_optimizer()
        self.define_scheduler()
    # ----------------------------------------
    # define loss
    # ----------------------------------------
    def define_loss(self):

        if self.G_lossfn_type == 'l1':
            self.G_lossfn = nn.L1Loss()
        elif self.G_lossfn_type == 'l2':
            self.G_lossfn = nn.MSELoss()
        else:
            raise NotImplementedError('Loss type [{:s}] is not found.'.format(self.G_lossfn_type))

    # ----------------------------------------
    # define optimizer
    # ----------------------------------------
    def define_optimizer(self):
        G_optim_params = []
        for k, v in self.netG.named_parameters():
            if v.requires_grad:
                G_optim_params.append(v)
            else:
                print('Params [{:s}] will not optimize.'.format(k))

        self.G_optimizer = Adam(G_optim_params, lr = self.G_optimizer_lr,
                                betas = self.G_optimizer_betas,
                                weight_decay = self.G_optimizer_wd)

    # ----------------------------------------
    # define scheduler, only "MultiStepLR"
    # ----------------------------------------
    def define_scheduler(self):
        self.schedulers.append(lr_scheduler.MultiStepLR(self.G_optimizer,
                                                        milestones = [1, 2, 3],
                                                        gamma = 0.1))

    # ----------------------------------------
    # save model / optimizer(optional)
    # ----------------------------------------
    def save(self, iter_label):
        self.save_network(self.save_dir, self.netG, 'G', iter_label)

    # ----------------------------------------
    # save the state_dict of the network
    # ----------------------------------------
    def save_network(self, save_dir, network, network_label, iter_label):
        save_filename = '{}_{}.pth'.format(iter_label, network_label)
        save_path = os.path.join(save_dir, save_filename)
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    # ----------------------------------------
    # feed L/H data
    # ----------------------------------------
    def feed_data(self, data, need_H=True):
        print("datat[L] shape",data["L"].shape)
        self.L = data['L']
        if need_H:
            self.H = data['H']

    # ----------------------------------------
    # feed L to netG
    # ----------------------------------------
    def netG_forward(self):
        self.E = self.netG(self.L)[:,0,:,:]

    # ----------------------------------------
    # update parameters and get loss
    # ----------------------------------------
    def optimize_parameters(self, current_step):
        self.G_optimizer.zero_grad()
        self.netG_forward()
        self.G_loss = self.G_lossfn(self.E, self.H)
        self.G_loss.backward()
        self.G_optimizer.step()

    def update_learning_rate(self, n):
        for scheduler in self.schedulers:
            scheduler.step(n)

    # ----------------------------------------
    # test / inference
    # ----------------------------------------
    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.netG_forward()
        self.netG.train()

    # ----------------------------------------
    # get L, E, H image
    # ----------------------------------------
    def current_visuals(self, need_H=True):
        out_dict = OrderedDict()
        out_dict['L'] = self.L.detach()[0].float()
        out_dict['E'] = self.E.detach()[0].float()
        if need_H:
            out_dict['H'] = self.H.detach()[0].float()
        return out_dict



def run(train_dir: str = "/p/scratch/deepacf/deeprain/bing/downscaling_maelstrom/train",
        test_dir: str = "/p/scratch/deepacf/deeprain/bing/downscaling_maelstrom/test",
        n_channels : int = 8, save_dir: str = "../results", checkpoint_save: int = 200,
        epochs: int = 2):

    """
    :param train_dir       : the directory that contains the training dataset NetCDF files
    :param test_dir        : the directory that contains the testing dataset NetCDF files
    :param checkpoint_save : how many steps to save checkpoint
    :param n_channels      : the number of input variables/channels
    :param save_dir        : the directory where the checkpoint results are save
    :param epochs          : the number of epochs

    """

    train_loader = create_loader(train_dir)
    test_loader = create_loader(test_dir)

    netG = net(n_channels = n_channels)
    netG_params = sum(p.numel() for p in netG.parameters() if p.requires_grad)
    print("Total trainalbe parameters:", netG_params)
    model = BuildModel(netG, save_dir = save_dir)
    model.init_train()
    current_step = 0

    for epoch in range(epochs):
        for i, train_data in enumerate(train_loader):
            st = time.time()

            current_step += 1

            # -------------------------------
            # 1) update learning rate
            # -------------------------------
            model.update_learning_rate(current_step)

            # -------------------------------
            # 2) feed patch pairs
            # -------------------------------
            model.feed_data(train_data)

            # -------------------------------
            # 3) optimize parameters
            # -------------------------------
            model.optimize_parameters(current_step)

            # -------------------------------
            # 6) Save model
            # -------------------------------
            if current_step == 1 or current_step % checkpoint_save == 0:
                model.save(current_step)
                print("Model Loss {} after step {}".format(model.G_loss, current_step))
                print("Model Saved")
                print("Time per step:", time.time() - st)
                wandb.log({"loss":loss, "lr":model.G_optimizer_lr}])
                

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type = str, required = True,
                        help = "The directory where training data (.nc files) are stored")
    parser.add_argument("--test_dir", type = str, required = True,
                        help = "The directory where testing data (.nc files) are stored")
    parser.add_argument("--save_dir", type = str, help = "The checkpoint directory")
    parser.add_argument("--epochs", type = int, default = 2, help = "The checkpoint directory")

    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    #save the args to the checkpoint directory
    with open(os.path.join(args.save_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(args), sort_keys = True, indent = 4))

    run(train_dir = args.train_dir,
        test_dir = args.test_dir,
        n_channels = 8,
        save_dir = args.save_dir,
        checkpoint_save = 200,
        epochs = args.epochs)





if __name__ == '__main__':
    main()


