conda env create -f environment.yaml
conda activate ddpm

# Run the DDPM training
accelerate launch ddpm_accelerate.py
