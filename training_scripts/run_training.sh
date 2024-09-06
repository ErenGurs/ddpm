# With the new docker, I had problems to install the environment.yaml below. For some reason,
# it installed more recent torch, torchvision,... than the Cuda drivers of the docker support.
# So it ended up with no Cuda support for torch. If I run the following command before creating
# the environment, it worked.
#
#conda clean --all
#

conda env create -f environment.yaml
conda activate ddpm

# Run the DDPM training
accelerate launch ddpm_accelerate.py
