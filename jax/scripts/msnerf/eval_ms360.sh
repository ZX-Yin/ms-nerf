#!/bin/bash
# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

export CUDA_VISIBLE_DEVICES='0,1'

# for synthetic part
SCENE=Scene04
EXPERIMENT=MS-Mip-NeRF-360
DATA_DIR=/jiaopengyi/ms-nerf/jax/dataset
CHECKPOINT_DIR=/jiaopengyi/ms-nerf/jax/output/"$EXPERIMENT"/"$SCENE"

python -m eval \
  --gin_configs=configs/ms-nerf/ms360.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}/${SCENE}'" \
  --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}'" \
  --logtostderr

# for real captured part
# SCENE=Scan01
# EXPERIMENT=logs_MS-Mip-NeRF-360
# DATA_DIR=/mnt/sda/experiments/cvpr23_real_cap_dataset
# CHECKPOINT_DIR=/mnt/sda/experiments/cvpr23/Mip-NeRF-360/"$EXPERIMENT"/"$SCENE"

# python -m eval \
#   --gin_configs=configs/ms-nerf/ms360.gin \
#   --gin_bindings="Config.dataset_loader = 'llff'" \
#   --gin_bindings="Config.factor = 8" \
#   --gin_bindings="Config.data_dir = '${DATA_DIR}/${SCENE}'" \
#   --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}'" \
#   --logtostderr