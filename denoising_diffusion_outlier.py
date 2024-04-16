
import torch
import torch.nn as nn
from tqdm import tqdm
from utils import *
#import torch.nn.functional as F

class GaussianDiffusion(nn.Module):
    def __init__(self, model, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=64, device="cuda"):
        
        super().__init__()

        self.model = model

        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device
        self.loss_func = nn.MSELoss()

        self.beta = torch.linspace(beta_start, beta_end, noise_steps).to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        #self.mse = nn.MSELoss()

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        eps = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * eps, eps

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def forward(self, images):
        t = self.sample_timesteps(images.shape[0]).to(self.device)
        x_t, noise = self.noise_images(images, t)
        #
        #grid_img_noised = torchvision.utils.make_grid(x_t, nrow=6)
        #grid_img = torchvision.utils.make_grid(images, nrow=6)
        #torchvision.utils.save_image(grid_img_noised, 'eren_noised.png')
        #torchvision.utils.save_image(grid_img, 'eren.png')

        predicted_noise = self.model(x_t, t)
        loss = self.loss_func(noise, predicted_noise)
        #loss = nn.MSELoss(noise, predicted_noise)
        #loss = mse(noise, predicted_noise)
        return loss

    # model: Denoising network (typically U-net or denoising autoencoder)
    # n:     Number of images to be generated
    def sample(self, model, n):
        #logging.info(f"Sampling {n} images ...")
        a=1
        os.makedirs("results/denoised/", exist_ok=True)
        model.eval()
        with torch.no_grad():
            # Create initial noised images x_T (for ex. T=1000) for the reverse diffusion process
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            # Reverse (diffusion) process time steps (for.ex 1000...1)
            for i in tqdm(reversed(range(1, self.noise_steps))):
                # time step (t=1000..1) for each of the n images 
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)


                alpha = self.alpha[t][:,None,None,None]
                alpha_hat = self.alpha_hat[t][:,None,None,None]
                beta = self.beta[t][:,None,None,None]
                # At time steps t>1: We subtract the predicted noise then add some noise back weighted with beta
                # At last time step t=1: No noise is added (see noise=0)
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)

                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * predicted_noise) + torch.sqrt(beta) * noise
                # See slowly denoised images
                if i % 50 == 1:
                    y = (x.clamp(-1, 1) + 1) / 2
                    y = (y * 255).type(torch.uint8)
                    save_images(y, os.path.join("results/denoised/", f"denoised_{a:03d}.jpg"))
                    a=a+1
        model.train()

        # Clamp the output to normalized range of (-1,1)
        #x = x.clamp(-1, 1)
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)

        return x