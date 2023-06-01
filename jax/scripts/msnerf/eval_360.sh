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

SCENE=Scene01
EXPERIMENT=logs_Mip-NeRF-360
DATA_DIR=/mnt/sda/T3/cvpr23/dataset/synthetic_scenes
CHECKPOINT_DIR=/mnt/sda/experiments/cvpr23/Mip-NeRF-360/"$EXPERIMENT"/"$SCENE"

python -m eval \
  --gin_configs=configs/ms-nerf/360.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}/${SCENE}'" \
  --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}'" \
  --logtostderr


SCENE=Scene02
EXPERIMENT=logs_Mip-NeRF-360
DATA_DIR=/mnt/sda/T3/cvpr23/dataset/synthetic_scenes
CHECKPOINT_DIR=/mnt/sda/experiments/cvpr23/Mip-NeRF-360/"$EXPERIMENT"/"$SCENE"

python -m eval \
  --gin_configs=configs/ms-nerf/360.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}/${SCENE}'" \
  --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}'" \
  --logtostderr


SCENE=Scene03
EXPERIMENT=logs_Mip-NeRF-360
DATA_DIR=/mnt/sda/T3/cvpr23/dataset/synthetic_scenes
CHECKPOINT_DIR=/mnt/sda/experiments/cvpr23/Mip-NeRF-360/"$EXPERIMENT"/"$SCENE"

python -m eval \
  --gin_configs=configs/ms-nerf/360.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}/${SCENE}'" \
  --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}'" \
  --logtostderr


SCENE=Scene04
EXPERIMENT=logs_Mip-NeRF-360
DATA_DIR=/mnt/sda/T3/cvpr23/dataset/synthetic_scenes
CHECKPOINT_DIR=/mnt/sda/experiments/cvpr23/Mip-NeRF-360/"$EXPERIMENT"/"$SCENE"

python -m eval \
  --gin_configs=configs/ms-nerf/360.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}/${SCENE}'" \
  --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}'" \
  --logtostderr


SCENE=Scene05
EXPERIMENT=logs_Mip-NeRF-360
DATA_DIR=/mnt/sda/T3/cvpr23/dataset/synthetic_scenes
CHECKPOINT_DIR=/mnt/sda/experiments/cvpr23/Mip-NeRF-360/"$EXPERIMENT"/"$SCENE"

python -m eval \
  --gin_configs=configs/ms-nerf/360.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}/${SCENE}'" \
  --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}'" \
  --logtostderr


SCENE=Scene06
EXPERIMENT=logs_Mip-NeRF-360
DATA_DIR=/mnt/sda/T3/cvpr23/dataset/synthetic_scenes
CHECKPOINT_DIR=/mnt/sda/experiments/cvpr23/Mip-NeRF-360/"$EXPERIMENT"/"$SCENE"

python -m eval \
  --gin_configs=configs/ms-nerf/360.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}/${SCENE}'" \
  --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}'" \
  --logtostderr


SCENE=Scene07
EXPERIMENT=logs_Mip-NeRF-360
DATA_DIR=/mnt/sda/T3/cvpr23/dataset/synthetic_scenes
CHECKPOINT_DIR=/mnt/sda/experiments/cvpr23/Mip-NeRF-360/"$EXPERIMENT"/"$SCENE"

python -m eval \
  --gin_configs=configs/ms-nerf/360.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}/${SCENE}'" \
  --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}'" \
  --logtostderr


SCENE=Scene08
EXPERIMENT=logs_Mip-NeRF-360
DATA_DIR=/mnt/sda/T3/cvpr23/dataset/synthetic_scenes
CHECKPOINT_DIR=/mnt/sda/experiments/cvpr23/Mip-NeRF-360/"$EXPERIMENT"/"$SCENE"

python -m eval \
  --gin_configs=configs/ms-nerf/360.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}/${SCENE}'" \
  --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}'" \
  --logtostderr


SCENE=Scene09
EXPERIMENT=logs_Mip-NeRF-360
DATA_DIR=/mnt/sda/T3/cvpr23/dataset/synthetic_scenes
CHECKPOINT_DIR=/mnt/sda/experiments/cvpr23/Mip-NeRF-360/"$EXPERIMENT"/"$SCENE"

python -m eval \
  --gin_configs=configs/ms-nerf/360.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}/${SCENE}'" \
  --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}'" \
  --logtostderr


SCENE=Scene10
EXPERIMENT=logs_Mip-NeRF-360
DATA_DIR=/mnt/sda/T3/cvpr23/dataset/synthetic_scenes
CHECKPOINT_DIR=/mnt/sda/experiments/cvpr23/Mip-NeRF-360/"$EXPERIMENT"/"$SCENE"

python -m eval \
  --gin_configs=configs/ms-nerf/360.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}/${SCENE}'" \
  --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}'" \
  --logtostderr


SCENE=Scene11
EXPERIMENT=logs_Mip-NeRF-360
DATA_DIR=/mnt/sda/T3/cvpr23/dataset/synthetic_scenes
CHECKPOINT_DIR=/mnt/sda/experiments/cvpr23/Mip-NeRF-360/"$EXPERIMENT"/"$SCENE"

python -m eval \
  --gin_configs=configs/ms-nerf/360.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}/${SCENE}'" \
  --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}'" \
  --logtostderr


SCENE=Scene12
EXPERIMENT=logs_Mip-NeRF-360
DATA_DIR=/mnt/sda/T3/cvpr23/dataset/synthetic_scenes
CHECKPOINT_DIR=/mnt/sda/experiments/cvpr23/Mip-NeRF-360/"$EXPERIMENT"/"$SCENE"

python -m eval \
  --gin_configs=configs/ms-nerf/360.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}/${SCENE}'" \
  --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}'" \
  --logtostderr


SCENE=Scene13
EXPERIMENT=logs_Mip-NeRF-360
DATA_DIR=/mnt/sda/T3/cvpr23/dataset/synthetic_scenes
CHECKPOINT_DIR=/mnt/sda/experiments/cvpr23/Mip-NeRF-360/"$EXPERIMENT"/"$SCENE"

python -m eval \
  --gin_configs=configs/ms-nerf/360.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}/${SCENE}'" \
  --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}'" \
  --logtostderr


SCENE=Scene14
EXPERIMENT=logs_Mip-NeRF-360
DATA_DIR=/mnt/sda/T3/cvpr23/dataset/synthetic_scenes
CHECKPOINT_DIR=/mnt/sda/experiments/cvpr23/Mip-NeRF-360/"$EXPERIMENT"/"$SCENE"

python -m eval \
  --gin_configs=configs/ms-nerf/360.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}/${SCENE}'" \
  --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}'" \
  --logtostderr


SCENE=Scene15
EXPERIMENT=logs_Mip-NeRF-360
DATA_DIR=/mnt/sda/T3/cvpr23/dataset/synthetic_scenes
CHECKPOINT_DIR=/mnt/sda/experiments/cvpr23/Mip-NeRF-360/"$EXPERIMENT"/"$SCENE"

python -m eval \
  --gin_configs=configs/ms-nerf/360.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}/${SCENE}'" \
  --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}'" \
  --logtostderr


SCENE=Scene16
EXPERIMENT=logs_Mip-NeRF-360
DATA_DIR=/mnt/sda/T3/cvpr23/dataset/synthetic_scenes
CHECKPOINT_DIR=/mnt/sda/experiments/cvpr23/Mip-NeRF-360/"$EXPERIMENT"/"$SCENE"

python -m eval \
  --gin_configs=configs/ms-nerf/360.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}/${SCENE}'" \
  --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}'" \
  --logtostderr


SCENE=Scene17
EXPERIMENT=logs_Mip-NeRF-360
DATA_DIR=/mnt/sda/T3/cvpr23/dataset/synthetic_scenes
CHECKPOINT_DIR=/mnt/sda/experiments/cvpr23/Mip-NeRF-360/"$EXPERIMENT"/"$SCENE"

python -m eval \
  --gin_configs=configs/ms-nerf/360.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}/${SCENE}'" \
  --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}'" \
  --logtostderr


SCENE=Scene18
EXPERIMENT=logs_Mip-NeRF-360
DATA_DIR=/mnt/sda/T3/cvpr23/dataset/synthetic_scenes
CHECKPOINT_DIR=/mnt/sda/experiments/cvpr23/Mip-NeRF-360/"$EXPERIMENT"/"$SCENE"

python -m eval \
  --gin_configs=configs/ms-nerf/360.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}/${SCENE}'" \
  --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}'" \
  --logtostderr


SCENE=Scene19
EXPERIMENT=logs_Mip-NeRF-360
DATA_DIR=/mnt/sda/T3/cvpr23/dataset/synthetic_scenes
CHECKPOINT_DIR=/mnt/sda/experiments/cvpr23/Mip-NeRF-360/"$EXPERIMENT"/"$SCENE"

python -m eval \
  --gin_configs=configs/ms-nerf/360.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}/${SCENE}'" \
  --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}'" \
  --logtostderr


SCENE=Scene20
EXPERIMENT=logs_Mip-NeRF-360
DATA_DIR=/mnt/sda/T3/cvpr23/dataset/synthetic_scenes
CHECKPOINT_DIR=/mnt/sda/experiments/cvpr23/Mip-NeRF-360/"$EXPERIMENT"/"$SCENE"

python -m eval \
  --gin_configs=configs/ms-nerf/360.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}/${SCENE}'" \
  --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}'" \
  --logtostderr


SCENE=Scene21
EXPERIMENT=logs_Mip-NeRF-360
DATA_DIR=/mnt/sda/T3/cvpr23/dataset/synthetic_scenes
CHECKPOINT_DIR=/mnt/sda/experiments/cvpr23/Mip-NeRF-360/"$EXPERIMENT"/"$SCENE"

python -m eval \
  --gin_configs=configs/ms-nerf/360.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}/${SCENE}'" \
  --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}'" \
  --logtostderr


SCENE=Scene22
EXPERIMENT=logs_Mip-NeRF-360
DATA_DIR=/mnt/sda/T3/cvpr23/dataset/synthetic_scenes
CHECKPOINT_DIR=/mnt/sda/experiments/cvpr23/Mip-NeRF-360/"$EXPERIMENT"/"$SCENE"

python -m eval \
  --gin_configs=configs/ms-nerf/360.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}/${SCENE}'" \
  --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}'" \
  --logtostderr


SCENE=Scene23
EXPERIMENT=logs_Mip-NeRF-360
DATA_DIR=/mnt/sda/T3/cvpr23/dataset/synthetic_scenes
CHECKPOINT_DIR=/mnt/sda/experiments/cvpr23/Mip-NeRF-360/"$EXPERIMENT"/"$SCENE"

python -m eval \
  --gin_configs=configs/ms-nerf/360.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}/${SCENE}'" \
  --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}'" \
  --logtostderr


SCENE=Scene24
EXPERIMENT=logs_Mip-NeRF-360
DATA_DIR=/mnt/sda/T3/cvpr23/dataset/synthetic_scenes
CHECKPOINT_DIR=/mnt/sda/experiments/cvpr23/Mip-NeRF-360/"$EXPERIMENT"/"$SCENE"

python -m eval \
  --gin_configs=configs/ms-nerf/360.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}/${SCENE}'" \
  --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}'" \
  --logtostderr


SCENE=Scene25
EXPERIMENT=logs_Mip-NeRF-360
DATA_DIR=/mnt/sda/T3/cvpr23/dataset/synthetic_scenes
CHECKPOINT_DIR=/mnt/sda/experiments/cvpr23/Mip-NeRF-360/"$EXPERIMENT"/"$SCENE"

python -m eval \
  --gin_configs=configs/ms-nerf/360.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}/${SCENE}'" \
  --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}'" \
  --logtostderr


