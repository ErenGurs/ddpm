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

For my convenience I put in blobby and everything is setup by running:
```
source ./setup.sh
```

Then run:
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


## References:
[1] Started the code based on outlier's "Diffusion-Models-pytorch" repo on [github](https://github.com/dome272/Diffusion-Models-pytorch).

[2] Also used his youtube tutorial [[Diffusion Models | Pytorch Implementation](https://www.youtube.com/watch?v=TBCRlnwJtZU)].
