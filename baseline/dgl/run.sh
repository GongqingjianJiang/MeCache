#!/bin/bash

# ./run.sh rgcn igb-full-small paper 19 1024 paper,author,institute,conference,fos,journal none 256 64 10

MODEL=${1:-"rgcn"}
DATASET=${2:-"ogbn-mag"}
PREDICT_CATEGORY=${3:-"paper"}
N_CLASSES=${4:-"349"}
BATCH_SIZE=${5:-"512"}
NTYPES_W_FEATS=${6:-"paper"}
CACHE_METHOD=${7:-"none"}
NUM_HIDDEN=${8:-"256"}
EMBEDDING_SIZE=${9:-"64"}
NUM_EPOCHS=${10:-"10"}
NUM_WORKERS=${11:-"4"}
IP_CONFIG=${12:-"ip_config_gn71.txt"}
BACKEND=${13:-"nccl"}

cmd="python3 train_dist.py --graph_name ${DATASET} \
--model ${MODEL} --ip_config ${IP_CONFIG} \
--num_epochs ${NUM_EPOCHS} --batch_size ${BATCH_SIZE} --n_classes ${N_CLASSES} \
--predict_category ${PREDICT_CATEGORY} --eval_every 1 --fan_out 5,10,15 \
--num_hidden ${NUM_HIDDEN} --embed_dim ${EMBEDDING_SIZE} \
--part_dir partitions/heta/${DATASET}_1parts_depth_3 \
--num_gpus ${NUM_WORKERS} --dgl-sparse --cache-method ${CACHE_METHOD} --num_layers 3 \
--ntypes-w-feats ${NTYPES_W_FEATS} --no-test --backend ${BACKEND}"

module add cuda/11.7

python3 ../launch.py \
    --workspace /gf3/home/jgqj/test_code/hydro/baseline/dgl \
    --num_trainers ${NUM_WORKERS} \
    --num_samplers 0 \
    --num_servers 1 \
    --part_config partitions/heta/${DATASET}_1parts_depth_3/${DATASET}.json \
    --ip_config ${IP_CONFIG} \
    "${cmd}" > log/speed/dgl_${MODEL}_${DATASET}_${CACHE_METHOD}_${EMBEDDING_SIZE}_${NUM_WORKERS}_${BACKEND}.log
