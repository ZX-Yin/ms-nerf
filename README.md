# MS-NeRF(CVPR 2023)

This repository contains the official implementation of the following paper:
> **Multi-Space Neural Radiance Fields**<br>
> Ze-Xin Yin, Jiaxiong Qiu, Ming-Ming Cheng, Bo Ren<sup>*</sup><br>
> IEEE/CVF Conference on Computer Vision and Pattern Recognition (**CVPR**), 2023<br>

[[Arxiv](https://arxiv.org/abs/2305.04268)]
[[Project Page](https://zx-yin.github.io/msnerf/)] 
[[Dataset](https://github.com/ZX-Yin/ms-nerf-dataset.git)]
[[Checkpoints](https://github.com/ZX-Yin/ms-nerf-ckpts.git)]

##  Work in Progress

- [√] Release the training and evaluation code for Mip-NeRF 360-based experiments.
- [ ] Integrate the Mip-NeRF and NeRF -based code into the Jax version of implementation.
- [ ] Re-implement a PyTorch version of the codebase.

## Jax implementation

We build our code on top of [MultiNeRF](https://github.com/google-research/multinerf), please check the code in `jax/`.

## PyTorch implementation

We are working on this...

## Citation

If you find our repo useful for your research, please consider citing our paper:

```bibtex
@InProceedings{Yin_2023_CVPR,
    author    = {Yin, Ze-Xin and Qiu, Jiaxiong and Cheng, Ming-Ming and Ren, Bo},
    title     = {Multi-Space Neural Radiance Fields},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {12407-12416}
}
```

And the codebase is heavily borrowed from [MultiNeRF](https://github.com/google-research/multinerf), please consider also cite this repository:

```
@misc{multinerf2022,
      title={{MultiNeRF}: {A} {Code} {Release} for {Mip-NeRF} 360, {Ref-NeRF}, and {RawNeRF}},
      author={Ben Mildenhall and Dor Verbin and Pratul P. Srinivasan and Peter Hedman and Ricardo Martin-Brualla and Jonathan T. Barron},
      year={2022},
      url={https://github.com/google-research/multinerf},
}
```

## License
Licensed under a [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/) for Non-commercial use only.
Any commercial use should get formal permission first.

## Acknowledgement

We thank for some wonderful nerf repos, including [mipnerf](https://github.com/google/mipnerf), [multinerf](https://github.com/google-research/multinerf), [nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch) , [nerfren](https://github.com/bennyguo/nerfren), and [nerf_pl](https://github.com/kwea123/nerf_pl). We heavily borrow codes from these projects.