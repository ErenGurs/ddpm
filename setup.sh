#!/bin/bash

export PATH=$PATH:/coreflow/venv/bin/
alias blobby='aws --endpoint-url https://blob.mr3.simcloud.apple.com --cli-read-timeout 300'
alias conductor='AWS_EC2_METADATA_DISABLED=true aws --endpoint-url https://conductor.data.apple.com'

mkdir -p landscape_img_folder/train

echo "Downloading s3://egurses-diffusion/Datasets/archive.zip"
#blobby s3 cp s3://Qixin_Selected_for_FlowAnalysis/Datasets/archive.zip ./
conductor s3 cp s3://egurses-diffusion/Datasets/archive.zip .
echo "Uncompressing archive.zip to landscape_img_folder/train/"
unzip archive.zip -d landscape_img_folder/train/ > /dev/null 2>&1

echo
mkdir -p cifar10
echo "Downloading s3://egurses-diffusion/Datasets/CIFAR.zip"
conductor s3 cp s3://egurses-diffusion/Datasets/CIFAR.zip .
echo "Uncompressing CIFAR.zip to cifar/"
unzip CIFAR.zip -d cifar10/ > /dev/null 2>&1


echo
mkdir -p img_align_celeba/train/
echo "Downloading s3://egurses-diffusion/Datasets/celeba.zip"
conductor s3 cp s3://egurses-diffusion/Datasets/celeba.zip ./
echo "Uncompressing celeba.zip to celeba/"
unzip ./celeba.zip
echo "Uncompress celeba/img_align_celeba.zip to img_align_celeba/"
unzip -j celeba/img_align_celeba.zip -d img_align_celeba/train/ > /dev/null 2>&1

# First tar'ed 40000 images (out of 70000) from ffhq and put it in conductor as ffhq512_full1.tar
#   $ tar -cvf ffhq512_full1.tar ffhq512_full/train/ > /dev/null 2>&1
echo
mkdir -p ffhq512_full/train/
echo "Downloading s3://egurses-diffusion/Datasets/ffhq512_full1.tar"
conductor s3 cp s3://egurses-diffusion/Datasets/ffhq512_full1.tar ./
echo "Untar ffhq512_full1.tar to ffhq512_full/"
tar -xvf ./ffhq512_full1.tar

echo "Downloading checkpoints for celeba"
conductor s3 cp s3://egurses-diffusion/ddpm/models_celeba/ckpt_epoch80.pt ./models/
conductor s3 cp s3://egurses-diffusion/ddpm/models_celeba/ckpt_epoch300.pt ./models/
#conductor s3 cp s3://egurses-diffusion/ddpm/models_celeba/ckpt_epoch100_ddpm_accelerate.pt ./models/
#conductor s3 cp s3://egurses-diffusion/ddpm/models_celeba/ckpt_epoch140_ddpm_accelerate.pt ./models/
conductor s3 cp s3://egurses-diffusion/ddpm/models_celeba/ckpt_epoch80_ddpm_accelerate.pt ./models/
conductor s3 cp s3://egurses-diffusion/ddpm/models_celeba/ckpt_epoch490_ddpm_accelerate.pt ./models/
conductor s3 cp s3://egurses-diffusion/ddpm/models_landscape/ckpt_epoch490_ddpm_accelerate.pt ./models/ckpt_epoch490_ddpm_accelerate_landscape.pt

