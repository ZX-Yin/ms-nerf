#!/bin/bash
# Copyright 2023 Ze-Xin Yin
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

export CUDA_VISIBLE_DEVICES=0

SCENE=Scene01
EXPERIMENT=logs_Mip-NeRF-360
DATA_DIR=/mnt/sda/T3/cvpr23/dataset/synthetic_scenes
CHECKPOINT_DIR=/mnt/sda/experiments/cvpr23/Mip-NeRF-360/"$EXPERIMENT"/"$SCENE"

# If running one of the indoor scenes, add
# --gin_bindings="Config.factor = 2"

rm "$CHECKPOINT_DIR"/*
python -m train \
  --gin_configs=configs/ms-nerf/360.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}/${SCENE}'" \
  --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}'" \
  --logtostderr
