# ddpm
Create environment:
```
conda env create -f environment.yaml
conda activate ddpm
```

Download the [Landscape Dataset](https://www.kaggle.com/datasets/utkarshsaxenadn/landscape-recognition-image-dataset-12k-images) from Kaggle as archive.zip.

```
$ mkdir -p landscape_img_folder/train
$ unzip archive.zip -d landscape_img_folder/train/
```

For my convenience I set up the buckets and download datasets, checkpoints etc. by running:
```
source ./setup.sh
```

Then for training:
```
$ python ddpm.py
```

<a id="Reconstruction-table"></a>
<table>
<caption style="caption-side:bottom"> Table: Noising process for iterations randomly sampled from [0,1000] </caption>
  <tr>
    <td align="center"> Original </td>
    <td align="center"> Noised </td>
  </tr>
  <tr>
    <td> <img src="images/original.png" width="500"/> </td>
    <td> <img src="images/noised.png" width="500"/> </td>
  </tr>
</table>

</br>
</br>
</br>





Also trained it for the `celeba` dataset. Download two example checkpoints (epoch 80,300) from the bucket (or `/ddpm/models_celeba/`). Then sample from these two checkpoints (saved as models/ckpt_epoch[80,300].pt):
```
python ddpm.py --ckpt /mnt/task_runtime/ddpm/models/ckpt_epoch80.pt --ckpt_sampling
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

## References:
[1] Started the code based on outlier's "Diffusion-Models-pytorch" repo on [github](https://github.com/dome272/Diffusion-Models-pytorch).

[2] Also used his youtube tutorial [[Diffusion Models | Pytorch Implementation](https://www.youtube.com/watch?v=TBCRlnwJtZU)].
