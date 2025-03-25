# This is the code release for MS-NeRF(CVPR 2023) based on [MultiNeRF](https://github.com/google-research/multinerf)


This repository contains the code release for CVPR 2023 paper [MS-NeRF](https://arxiv.org/abs/2305.04268), and we conduct all the Mip-NeRF 360 based experiments in this repository, therefore, you should reproduce the results reported in our paper.

Besides, the original repository contains three CVPR 2022 papers: 
[Mip-NeRF 360](https://jonbarron.info/mipnerf360/),
[Ref-NeRF](https://dorverbin.github.io/refnerf/), and
[RawNeRF](https://bmild.github.io/rawnerf/).
As we make minimal modifications, other methods should be runnable.
But we recommend using the original repository.

This implementation is written in [JAX](https://github.com/google/jax), and
is a fork of [MultiNeRF](https://github.com/google-research/multinerf).
This is research code, and should be treated accordingly.

## Setup

```
# Clone the repo.
git clone https://github.com/ZX-Yin/ms-nerf.git
cd ms-nerf/jax/

# Make a conda environment.
conda create --name ms-nerf python=3.9
conda activate ms-nerf

# Prepare pip.
conda install pip
pip install --upgrade pip

# Install requirements.
pip install -r requirements.txt

# Manually install rmbrualla's `pycolmap` (don't use pip's! It's different).
git clone https://github.com/rmbrualla/pycolmap.git ./internal/pycolmap

# Confirm that all the unit tests pass.
./scripts/run_all_unit_tests.sh
```
You'll probably also need to update your JAX installation to support GPUs or TPUs.

## Running MS-Mip-NeRF 360

Example scripts for training, evaluating and rendering with MS-Mip-NeRF 360 can be found in `scripts/msnerf/`. And we evaluate the PSNR, SSIM, and LPIPS using our own script.

## Running MS-Mip-NeRF and NeRF

We are trying to integrate this two experiments into this repository.

## Evaluating

We use PyTorch-based Python scripts to evaluate all our results, as there are many convient packages to use. Experiments on synthetic part of our dataset are evaluated using `../eval_metrics_syn.py`, and those on real captured part are using `../eval_metrics_llff.py`. You just need to set the variables `path` and `gt_path_root` in the scripts.

### OOM errors

You may need to reduce the batch size (`Config.batch_size`) to avoid out of memory
errors. If you do this, but want to preserve quality, be sure to increase the number
of training iterations and decrease the learning rate by whatever scale factor you
decrease batch size by.

### Notification

We make minimal modifications to this repository, therefore, all the other properties should remain the same. Please follow the [instructions](https://github.com/google-research/multinerf/blob/main/README.md) to further conduct researches.