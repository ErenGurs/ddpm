import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from utils import *

import sys
base_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(base_directory)

#from modules import UNet
from denoising_diffusion_pytorch import Unet, GaussianDiffusion  #, Trainer

from accelerate import Accelerator

import logging
from torch.utils.tensorboard import SummaryWriter
#from denoising_diffusion_outlier import GaussianDiffusion

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
        # DDP splits batch_size among training GPUs (better to use accelerator's num_process than torch.cuda.device_count
        args.batch_size = args.batch_size * self.accelerator.state.num_processes

        self.device = args.device
        # Get Data Loader
        #self.dataloader = get_data(args)
        self.dataloader = self.accelerator.prepare( get_data(args) )

        # Outlier's UNet(.) class
        #model = UNet(img_size=args.image_size).to(self.device)
        # Lucidrains Unet(.) class
        model = Unet(dim = 64, dim_mults = (1, 2, 4, 8), flash_attn = False) # flash_attn=True
        
        self.optimizer = optim.AdamW(model.parameters(), lr=args.lr)
        # Outlier's Diffusion class
        #self.diffusion = GaussianDiffusion(model, img_size=args.image_size, device=self.device)
        # Lucidrains Diffusion class
        self.diffusion = GaussianDiffusion(model,image_size = args.image_size, timesteps = 1000)
        self.diffusion, self.optimizer = self.accelerator.prepare(self.diffusion, self.optimizer)
        #print ('>>>> Current cuda device ', torch.cuda.current_device())


        self.logger = SummaryWriter(os.path.join("runs", args.run_name))

        if self.accelerator.is_main_process:
            print("Initializations only needed by the main process is done here  (for. ex model reading/writing, EMA)")

        mse = nn.MSELoss() # not used, delete it!

    def train(self):

        l = len(self.dataloader)

        # Disable logging
        #logging.basicConfig(disabled = not self.accelerator.is_main_process)

        for epoch in range(self.args.epochs):
            if self.accelerator.is_main_process:
                logging.info(f"Starting epoch {epoch} :")
        
            #print(f"Number of GPUS: {self.accelerator.state.num_processes}")
            #print(f"GPU{torch.cuda.current_device()}", self.dataloader)

            # Disable tqdm on "not main" processes (i.e. show tqdm progress only for process on GPU0)
            pbar = tqdm(self.dataloader, disable = not self.accelerator.is_main_process)
            for i, (images, _) in enumerate(pbar):

                images = images.to(self.device)

                #loss = self.diffusion(images)
                with self.accelerator.autocast():
                    loss = self.diffusion(images)
                #loss.backward()
                self.accelerator.backward(loss)

                pbar.set_postfix(MSE=loss.item(), GPU=torch.cuda.current_device())
                self.logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)
                #print(f"Loss={loss.item()},  GPU={torch.cuda.current_device()}, epoch*l+i={epoch*l} + {i}")

                self.accelerator.wait_for_everyone()

                self.optimizer.step()
                self.optimizer.zero_grad()

                self.accelerator.wait_for_everyone()

                #if (i > 100): # To be removed. Quick training for debugging etc.
                #    break

            # Sample from Diffusion model by putting it into evaluation mode (see model.eval())
            if epoch % 10 == 0 and self.accelerator.is_main_process:
                # Before learning 'accelerator.unwrap_model(.)' this was how I got rid of the DDP layer (called 'module') wrapped around the model.
                # diffusion = self.diffusion.module if isinstance(self.diffusion, nn.parallel.DistributedDataParallel) else self.diffusion
                diffusion = self.accelerator.unwrap_model(self.diffusion)
                # Outliers diffusionsample() function
                #sampled_images = diffusion.sample(n=images.shape[0])
                # Lucidrains diffusionsample() function
                sampled_images = diffusion.sample(batch_size=images.shape[0])

                # Denormalize to [0,255]: Clamp the output to normalized range of (-1,1)
                sampled_images = (sampled_images.clamp(-1, 1) + 1) / 2
                sampled_images = (sampled_images * 255).type(torch.uint8)

                save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
                #torch.save(self.accelerator.get_state_dict(self.diffusion.module.model), os.path.join("models", args.run_name, f"ckpt_diffusion.pt"))
                torch.save(self.accelerator.get_state_dict(diffusion.model), os.path.join("models", args.run_name, f"ckpt_diffusion.pt"))


def test(args):
    if not os.path.exists(args.ckpt):
        print ("Error: No checkpoint to sample (see --ckpt).\n")
        return

    device = args.device
    # Load model: Different from training No need for data loader or optimizer
    model = UNet(img_size=args.image_size).to(device)

    # 1) checkpoint is an OrderedDict with list of keys given by checkpoint.keys()
    #    For ex. checkpoint['inc.double_conv.0.weight'] gives weights (and biases) of the 
    #    DoubleConv (inc.double_conv) where 0-4 refers to Conv2d, GroupNorm, GELU, Conv2d, GroupNorm 
    # 2) OrderedDict can be iterated over its items():
    #      for i, (key, value) in enumerate(checkpoint.items()):
    #          print(key, value)
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
    # Denormalize to [0,255]: Clamp the output to normalized range of (-1,1)
    sampled_images = (sampled_images.clamp(-1, 1) + 1) / 2
    sampled_images = (sampled_images * 255).type(torch.uint8)
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
    #args.run_name = "DDPM_Unconditional_landscape"
    args.run_name = "DDPM_Unconditional"
    args.epochs = 500
    args.batch_size = 12    # 4/12 : Original batch size is reduced for 128x128 to fit into memory
    args.image_size = 128  # 64 : Original image size
    #args.dataset_path = r"./landscape_img_folder"
    #args.dataset_path = r"./img_align_celeba/"
    args.dataset_path = r"./ffhq512_full/"
    args.device = "cuda"
    args.lr = 3e-4

    setup_logging(args.run_name)

    if args.ckpt_sampling :
        test(args)
    else:
        trainer = Trainer(args)
        trainer.train()
