#!/usr/bin/env bash
DTU_TESTPATH="/hdd1/lhs/dev/code/MVSTER/data"
DTU_TESTLIST="lists/mdi/test.txt"

DTU_size=$1
exp=$2
PY_ARGS=${@:3}

DTU_LOG_DIR="./checkpoints/dtu/"$exp 
if [ ! -d $DTU_LOG_DIR ]; then
    mkdir -p $DTU_LOG_DIR
fi
DTU_CKPT_FILE=$DTU_LOG_DIR"/finalmodel.ckpt"
DTU_OUT_DIR="./outputs/mdi/"$exp




python test_mvs4.py --dataset=general_eval4_mdi --batch_size=1 --testpath=$DTU_TESTPATH  --testlist=$DTU_TESTLIST --loadckpt $DTU_CKPT_FILE --interval_scale 1.06 --outdir $DTU_OUT_DIR\
             --inverse_depth --thres_view 3 --conf 0.5 --group_cor --max_h 1920 --max_w 1920 --attn_temp 2 $PY_ARGS | tee -a $DTU_LOG_DIR/log_test.txt

#inverse!!!!!!