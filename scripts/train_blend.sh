#!/usr/bin/env bash
DTU_TRAINING="/hdd1/lhs/dev/dataset/dataset_low_res/"
DTU_TRAINLIST="/hdd1/lhs/dev/dataset/dataset_low_res/training_list.txt"
DTU_TESTLIST="/hdd1/lhs/dev/dataset/dataset_low_res/validation_list.txt"

DTU_trainsize=$1
exp=$2
PY_ARGS=${@:3}

DTU_LOG_DIR="./checkpoints/dtu/"$exp 
if [ ! -d $DTU_LOG_DIR ]; then
    mkdir -p $DTU_LOG_DIR
fi

DTU_CKPT_FILE=$DTU_LOG_DIR"/finalmodel.ckpt"
DTU_OUT_DIR="./outputs/dtu/"$exp



python -m torch.distributed.launch --nproc_per_node=6 train_mvs4.py --logdir $DTU_LOG_DIR --dataset=blendedmvs_snow --batch_size=1 --trainpath=$DTU_TRAINING --summary_freq 100 \
                --group_cor --rt --inverse_depth --attn_temp 2 --trainlist $DTU_TRAINLIST --testlist $DTU_TESTLIST  $PY_ARGS | tee -a $DTU_LOG_DIR/log.txt


