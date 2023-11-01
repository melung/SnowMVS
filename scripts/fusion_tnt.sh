#!/usr/bin/env bash
#source scripts/data_path.sh

THISNAME="8844_fine_blend"

LOG_DIR="./checkpoints/tnt/"$THISNAME 
TNT_OUT_DIR="./outputs/tnt/"$THISNAME
#TNT_ROOT = "/hdd1/lhs/dev/dataset/tankandtemples/intermediate"

# Intermediate
# python3 fusions/tnt/dypcd.py ${@} \
#     --root_dir="/hdd1/lhs/dev/dataset/tankandtemples/" --list_file="datasets/lists/tnt/intermediate.txt" --split="intermediate" \
#     --out_dir=$TNT_OUT_DIR --ply_path=$TNT_OUT_DIR"/dypcd_fusion_plys" \
#     --img_mode="resize" --cam_mode="origin" --single_processor 

# Advanced
python3 fusions/tnt/dypcd.py ${@} \
    --root_dir="/hdd1/lhs/dev/dataset/tankandtemples/" --list_file="datasets/lists/tnt/test.txt" --split="intermediate" \
    --out_dir=$TNT_OUT_DIR --ply_path=$TNT_OUT_DIR"/dypcd_fusion_plys" \
    --img_mode="resize" --cam_mode="origin" --single_processor