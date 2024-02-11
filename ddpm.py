import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from utils import *

import logging


logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%H:%S")



class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=64, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = torch.linspace(beta_start, beta_end, noise_steps).to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        eps = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * eps, eps

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))


def train(args):
    device = args.device
    # Get Data Loader
    dataloader = get_data(args)
    #model = UNet().to(device)
    #optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    #mse = nn.MSELoss()    
    diffusion = Diffusion(img_size=args.image_size, device=device)

    pbar = tqdm(dataloader)
    for i, (images, _) in enumerate(pbar):
        images = images.to(device)
        t = diffusion.sample_timesteps(images.shape[0]).to(device)
        x_t, noise = diffusion.noise_images(images, t)
        #
        grid_img_noised = torchvision.utils.make_grid(x_t, nrow=6)
        grid_img = torchvision.utils.make_grid(images, nrow=6)
        torchvision.utils.save_image(grid_img_noised, 'eren_noised.png')
        torchvision.utils.save_image(grid_img, 'eren.png')
#def launch():


if __name__ == '__main__':
    #launch()
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_Uncondtional"
    args.epochs = 500
    args.batch_size = 12
    args.image_size = 64
    args.dataset_path = r"./landscape_img_folder"
    args.device = "cuda"
    args.lr = 3e-4
    train(args)