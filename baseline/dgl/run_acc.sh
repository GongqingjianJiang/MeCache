#!/bin/bash

# ./run_gn70.sh rgcn igb-full-small paper 19 1024 paper,author,institute,conference,fos,journal miss_penalty 256 64 "5,10,15" 1000 0.5 3 8,8

MODEL=${1:-"rgcn"}
DATASET=${2:-"igb-part-small-pca"}
PREDICT_CATEGORY=${3:-"paper"}
N_CLASSES=${4:-"19"}
BATCH_SIZE=${5:-"1024"}
NTYPES_W_FEATS=${6:-"paper"}
CACHE_METHOD=${7:-"none"}
NUM_HIDDEN=${8:-"256"}
EMBEDDING_SIZE=${9:-"64"}
NUM_EPOCHS=${10:-"10"}
FANOUT=${11:-"25,20"}
IP_CONFIG=${12:-"ip_config_gn71.txt"}
NUM_LAYERS=${13:-"2"}
LR=${14:-"0.01"}
SPARSE_LR=${15:-"0.06"}

cmd="python3 train_dist.py --graph_name ${DATASET} \
--model ${MODEL} --ip_config ${IP_CONFIG} \
--num_epochs ${NUM_EPOCHS} --batch_size ${BATCH_SIZE} --n_classes ${N_CLASSES} \
--predict_category ${PREDICT_CATEGORY} --eval_every 1 --fan_out ${FANOUT} \
--num_hidden ${NUM_HIDDEN} --embed_dim ${EMBEDDING_SIZE} \
--part_dir partitions/heta/${DATASET}_1parts_depth_3 \
--num_gpus 4 --dgl-sparse --cache-method ${CACHE_METHOD} --num_layers ${NUM_LAYERS} \
--ntypes-w-feats ${NTYPES_W_FEATS} --lr ${LR} --sparse-lr ${SPARSE_LR}"

module add cuda/11.7

python3 ../launch.py \
    --workspace /gf3/home/jgqj/test_code/hydro/baseline/dgl \
    --num_trainers 4 \
    --num_samplers 0 \
    --num_servers 1 \
    --part_config partitions/heta/${DATASET}_1parts_depth_3/${DATASET}.json \
    --ip_config ${IP_CONFIG} \
    "${cmd}" > log/acc/dgl_${MODEL}_${DATASET}_${CACHE_METHOD}_${EMBEDDING_SIZE}.log
