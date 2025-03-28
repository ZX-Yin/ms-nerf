# MS-NeRF(CVPR 2023 & TPAMI 2025)

This repository contains the official implementation of the following paper:

> **MS-NeRF: Multi-Space Neural Radiance Fields**<br>
> Ze-Xin Yin, Peng-Yi Jiao, Jiaxiong Qiu, Ming-Ming Cheng, Bo Ren<sup>*</sup><br>
> IEEE Transactions on Pattern Analysis and Machine Intelligence (**TPAMI**), 2025<br>

> **Multi-Space Neural Radiance Fields**<br>
> Ze-Xin Yin, Jiaxiong Qiu, Ming-Ming Cheng, Bo Ren<sup>*</sup><br>
> IEEE/CVF Conference on Computer Vision and Pattern Recognition (**CVPR**), 2023<br>

[[Arxiv](https://arxiv.org/abs/2305.04268)]
[[Project Page](https://zx-yin.github.io/msnerf/)] 
[[Dataset](https://huggingface.co/datasets/JasonYinnnn/ms-nerf-dataset)]
[Dataset(百度云[TBD])]
[[Checkpoints](https://github.com/ZX-Yin/ms-nerf-ckpts.git)]

##  Work in Progress

- [√] Release the training and evaluation code for Mip-NeRF 360-based experiments.
- [ ] Release the training and evaluation code for TensoRF-based experiments.
- [ ] Release the training and evaluation code for iNGP-based experiments.
- [ ] Release the training and evaluation code for Mip-NeRF-based experiments.
- [ ] Release the training and evaluation code for NeRF-based experiments.

## Mip-NeRF 360-based implementation

We build our code on top of [MultiNeRF](https://github.com/google-research/multinerf), please check the code in `mip360/`. This part of implementation is heavily borrowed from [MultiNeRF](https://github.com/google-research/multinerf), please consider also cite this repository:

```
@misc{multinerf2022,
      title={{MultiNeRF}: {A} {Code} {Release} for {Mip-NeRF} 360, {Ref-NeRF}, and {RawNeRF}},
      author={Ben Mildenhall and Dor Verbin and Pratul P. Srinivasan and Peter Hedman and Ricardo Martin-Brualla and Jonathan T. Barron},
      year={2022},
      url={https://github.com/google-research/multinerf},
}
```

## Citation

If you find our repo useful for your research, please consider citing our paper:

```bibtex
@ARTICLE{msnerf_2025_TPAMI,
  author={Yin, Ze-Xin and Jiao, Peng-Yi and Qiu, Jiaxiong and Cheng, Ming-Ming and Ren, Bo},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={MS-NeRF: Multi-Space Neural Radiance Fields}, 
  year={2025},
  volume={},
  number={},
  pages={1-18},
  doi={10.1109/TPAMI.2025.3540074}}
```

```bibtex
@InProceedings{msnerf_2023_CVPR,
    author    = {Yin, Ze-Xin and Qiu, Jiaxiong and Cheng, Ming-Ming and Ren, Bo},
    title     = {Multi-Space Neural Radiance Fields},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {12407-12416}
}
```


## License
Licensed under a [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/) for Non-commercial use only.
Any commercial use should get formal permission first.

## Acknowledgement

We thank for some wonderful nerf repos, including [mipnerf](https://github.com/google/mipnerf), [multinerf](https://github.com/google-research/multinerf), [nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch) , [nerfren](https://github.com/bennyguo/nerfren), and [nerf_pl](https://github.com/kwea123/nerf_pl). We heavily borrow codes from these projects.