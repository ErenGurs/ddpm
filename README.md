# Diffusion Models
Create environment:
```
conda env create -f environment.yaml
conda activate ddpm
```

Download the [Landscape Dataset](https://www.kaggle.com/datasets/utkarshsaxenadn/landscape-recognition-image-dataset-12k-images) from Kaggle as archive.zip or the [Celeba dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) (aligned & cropped) as [img_align_celeba.zip](https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view?usp=drive_link&resourcekey=0-dYn9z10tMJOBAkviAcfdyQ).

```
$ mkdir -p landscape_img_folder/train
$ unzip archive.zip -d landscape_img_folder/train/
```

For my convenience I set up the buckets and download datasets, checkpoints etc. by running:
```
source ./setup.sh
```

## Sampling
Also trained it for the `celeba` dataset. Download three example checkpoints (epoch 30, 80, 490) from the bucket (or `/ddpm/models_celeba/`). Then sample from these three checkpoints (saved as models/ckpt_epoch[30, 80, 490]_ddpm.pt):
```
python ddpm_accelerate.py --ckpt /mnt/task_runtime/ddpm/models/ckpt_epoch490.pt --ckpt_sampling
```
<figure>
<figcaption>Epoch 80 ckpt</figcaption>
<img src="images/ddpm_slow.gif" width=75%>
</figure>

<figure>
<figcaption>Epoch 300 ckpt</figcaption>
<img src="images/ddpm_slow_ckpt_epoch300.gif" width=75%>
</figure>



Generated the gif by ``ffmpeg -framerate 5  -i results/denoised/denoised_%3d.jpg ddpm_slow.gif``


## Training
Use multi-GPU training script using <a href="https://huggingface.co/docs/accelerate/">🤗 Accelerate </a>.
```
accelerate launch ddpm_accelerate.py
```
**Note:** Running the same script as `python ddpm_acclerate.py` will fall back to single GPU mode (no effect of _accelerate_). Therefore, the same script can be directly used for single GPU tasks like sampling/inference or debugging. So I decided to retire the single GPU script `ddpm.py`.

Below images show noising of images used in training (see [Details on Notation](#details) for more on the notation)
- **Noised samples   :** from $q(\mathbf{x}_t |\mathbf{x}_0)$
- **Original samples :** from $q(\mathbf{x}_0)$


|   Noised samples  |      Original  |
|:-----------------------------------------:|:------------------------------------------:|
| ![](images/noised.png)      |  ![](images/original.png)  |
<!--|  <img src="images/noised.png"><img>      |  <img src="images/original.png"><img>   |  -->

Fig. For batch B=12, illustrates the noising process for $t=[962, 237,  38,  39, 988, 559, 299, 226, 985, 791, 859, 485]$

### <a name="details"></a> Details on Notation 
Noising (or diffusion) is defined as a Markovian process over states $\mathbf{x}_0,...\mathbf{x}_T$ with transitions from Normal distributions given by $q(\mathbf{x} _t|\mathbf{x} _{t-1}) = \mathcal{N}(\mathbf{x} _t; \sqrt{1-\beta _t} \mathbf{x} _{t-1}, \beta _t \mathbf{I} )$ for small values of $\beta_t$.
<p align="center">
<img src="https://lilianweng.github.io/posts/2021-07-11-diffusion-models/DDPM.png" style="width: 90%;" class="center" />
<figcaption>Fig. Markov chain of forward (reverse) diffusion process (Ref: <a href="https://arxiv.org/abs/2006.11239" target="_blank">Ho et al. 2020</a> with additions by <a href="https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#forward-diffusion-process"> Lil'log</a>)</figcaption>
</p>

 Then the noising process $q(\mathbf{x}_t|\mathbf{x}_0)$ at any time $t$ from this Markovian process is given as below (details in [Lil'log: What are Diffusion Models](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#forward-diffusion-process))

$$q(\mathbf{x}_t|\mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha} _t} \mathbf{x}_0, (1-\bar{\alpha}_t) \mathbf{I} )$$

In training, U-Net is trained to estimate the noise (i.e. the mean of $q(\mathbf{x} _{t-1} | \mathbf{x} _{t}, \mathbf{x} _0)$ the unknown de-noising process) from the noisy pictures sampled from $q(\mathbf{x}_t|\mathbf{x}_0)$ which are given by normal distribution above using a linear schedule $\beta_t \in [0.0001, 0.02]$ where $\alpha_t= 1-\beta_t$  and $\bar{\alpha}_t = \Pi _{s=1}^t \alpha_s$ (see Fig. below for $\bar{\alpha} _t$ and $\beta _t$ ).


<p align="center">
<img src="images/beta_alpha_hat.png" width="450">

<figcaption>

Fig. For given $\beta_t$ schedule and large  $t$, the $q(\mathbf{x}_t|\mathbf{x}_0)$ becomes zero mean, unit variance Normal distribution $\mathcal{N}(\mathbf{x} _T; \mathbf{0}, \mathbf{I})$ for $T=1000$

</figcaption>
</p>
<!--
Due to bug: https://stackoverflow.com/questions/78158848/how-to-render-both-of-latex-formula-and-image-in-markdown-table-in-github-readme
-->





<!--
<a id="Reconstruction-table"></a>
<table>
<caption style="caption-side:bottom"> Table: For batch=12, noising process for t=[962, 237,  38,  39, 988, 559, 299, 226, 985, 791, 859, 485] random iterations $$t \in [0,1000]$$ </caption>
  <tr>
    <td align="center"> Original </td>
    <td align="center"> Noised </td>
  </tr>
  <tr>
    <td> <img src="images/original.png" width="500"/> </td>
    <td> <img src="images/noised.png" width="500"/> </td>
  </tr>
</table>
-->

</br>

## References:
[1] Started the code based on outlier's [Diffusion-Models-pytorch](https://github.com/dome272/Diffusion-Models-pytorch) repo.

[2] Also used his youtube tutorial [[Diffusion Models | Pytorch Implementation](https://www.youtube.com/watch?v=TBCRlnwJtZU)].

[3] Referring and using Phil Wang's (lucidrains)   [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch) repo.

[4] Lillian Weng's blog (lil'log) https://lilianweng.github.io/posts/2021-07-11-diffusion-models/

[5] <a href="https://arxiv.org/abs/2006.11239" target="_blank">Ho et al. 2020</a> "Denoising Diffusion Probabilistic Models" paper with [original implementation](https://github.com/hojonathanho/diffusion) and its [pytorch reimplementation](https://github.com/pesser/pytorch_diffusion) by Patrick Esser.

[6] Referred to CompVis' [Latent Diffusion Models](https://github.com/CompVis/latent-diffusion) repo and paper [Rombach et al. 2022](https://arxiv.org/abs/2112.10752) "High-Resolution Image Synthesis with Latent Diffusion Models"

[7] HuggingFace's [Diffusers](https://github.com/huggingface/diffusers): State-of-the-art diffusion models for image and audio generation in PyTorch and FLAX. For quickly testing DDPM [here](https://huggingface.co/docs/diffusers/api/pipelines/ddpm)

[8] [MMagic](https://github.com/open-mmlab/mmagic) from OpenMMLab with nice resource on [Stable Diffusion](https://github.com/open-mmlab/mmagic/blob/main/configs/stable_diffusion/README.md) which mainly builds on HuggingFace's [diffusers](https://github.com/huggingface/diffusers).

