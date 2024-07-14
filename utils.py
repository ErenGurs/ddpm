import os
import torch
import torchvision
from PIL import Image
#from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from multiprocessing import cpu_count

def get_data(args):
    transforms = torchvision.transforms.Compose([
        #torchvision.transforms.Resize(80),  # args.image_size + 1/4 *args.image_size
        #torchvision.transforms.Resize(150),  # args.image_size + 1/4 *args.image_size
        torchvision.transforms.Resize(int(args.image_size + 1/4 *args.image_size)),  # args.image_size + 1/4 *args.image_size
        torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        #torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory = True, num_workers = cpu_count())
    return dataloader

def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs, nrow=images.shape[0]//2)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)

def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)