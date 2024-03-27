#!/bin/bash

export PATH=$PATH:/coreflow/venv/bin/
alias blobby='aws --endpoint-url https://blob.mr3.simcloud.apple.com --cli-read-timeout 300'
alias conductor='AWS_EC2_METADATA_DISABLED=true aws --endpoint-url https://conductor.data.apple.com'

echo "Downloading s3://Qixin_Selected_for_FlowAnalysis/Datasets/archive.zip"
#blobby s3 cp s3://Qixin_Selected_for_FlowAnalysis/Datasets/archive.zip ./
conductor s3 cp s3://egurses-diffusion/Datasets/archive.zip .
mkdir -p landscape_img_folder/train

echo
echo "Uncompressing archive.zip to landscape_img_folder/train/"
unzip archive.zip -d landscape_img_folder/train/ > /dev/null 2>&1
