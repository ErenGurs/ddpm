import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from utils import *
from modules import UNet

from accelerate import Accelerator

import logging
from torch.utils.tensorboard import SummaryWriter
from denoising_diffusion_outlier import GaussianDiffusion

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%H:%S")

class Trainer(object):
    def __init__(
        self,
        args,
        amp = False,
        mixed_precision_type = 'fp16',
        split_batches = True
    ):
        super().__init__()
        self.args = args

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = mixed_precision_type if amp else 'no'
        )

        setup_logging(args.run_name)
        self.device = args.device
        # Get Data Loader
        self.dataloader = get_data(args)
        model = UNet().to(self.device)
        self.optimizer = optim.AdamW(model.parameters(), lr=args.lr)
        
        mse = nn.MSELoss() # not used, delete it!
        self.diffusion = GaussianDiffusion(model, img_size=args.image_size, device=self.device)
        self.logger = SummaryWriter(os.path.join("runs", args.run_name))


    def train(self):

        l = len(self.dataloader)

        for epoch in range(self.args.epochs):
            logging.info(f"Starting epoch {epoch}:")
        
            pbar = tqdm(self.dataloader)
            for i, (images, _) in enumerate(pbar):
                #if self.args.ckpt:  # If checkpoint is specified, do not continue training
                #    break
                images = images.to(self.device)

                loss = self.diffusion(images)
                #if (i > 10): # To be removed. Quick training for debugging etc.
                #    break
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                pbar.set_postfix(MSE=loss.item())
                self.logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)
            # Sample from Diffusion model by putting it into evaluation mode (see model.eval())
            if epoch % 10 == 0:
                sampled_images = self.diffusion.sample(n=images.shape[0])
                save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
                torch.save(self.diffusion.model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))


def test(args):
    if not os.path.exists(args.ckpt):
        print ("Error: No checkpoint to sample (see --ckpt).\n")
        return

    device = args.device
    # Load model: Different from training No need for data loader or optimizer
    model = UNet().to(device)

    checkpoint = torch.load(args.ckpt)
    model.load_state_dict(checkpoint)
    #
    # if you don't want to resume training, it is common to set to valuation mode.
    # However for diffusion models it is different, since training and sampling paths are different. 
    # model.eval()
    #
    # Remember in order to resume training from a ckpt, you need to "torch.save" more than model.state_dict(),
    # such as below. (Details from https://pytorch.org/tutorials/beginner/saving_loading_models.html)
    # torch.save({
    #    'epoch': epoch,
    #    'model_state_dict': model.state_dict(),
    #    'optimizer_state_dict': optimizer.state_dict(),
    #    'loss': loss,
    #    ...
    #    }, PATH)
    #
    # When you log all the info above, then you load them separately (model and optimizer state_dict, epoch and loss):
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # epoch = checkpoint['epoch']
    # loss = checkpoint['loss']

    # Instantiate Diffusion class
    diffusion = GaussianDiffusion(model, img_size=args.image_size, device=device)

    # Sample from Diffusion model by putting it into evaluation mode (see model.eval())
    sampled_images = diffusion.sample(n=args.batch_size)
    save_images(sampled_images, os.path.join("results", args.run_name, f"sample.jpg"))


if __name__ == '__main__':
    #launch()
    import argparse
    parser = argparse.ArgumentParser("DDPM training and sampling script. Training works without any args.\n \
    But sampling requires args --ckpt*.")
    parser.add_argument('--ckpt', type=str, default='',
                    help='Checkpoint file')
    parser.add_argument('--ckpt_sampling', default=False, action='store_true',
                    help='Random image sampling using the checkpoint provided by --ckpt')
    args = parser.parse_args()
    args.run_name = "DDPM_Unconditional"
    args.epochs = 500
    args.batch_size = 12
    args.image_size = 64
    #args.dataset_path = r"./landscape_img_folder"
    args.dataset_path = r"./img_align_celeba/"
    args.device = "cuda"
    args.lr = 3e-4
    if args.ckpt_sampling :
        test(args)
    else:
        trainer = Trainer(args)
        trainer.train()