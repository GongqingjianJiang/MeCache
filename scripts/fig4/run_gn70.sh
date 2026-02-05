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
FANOUT=${10:-"25,20"}
EPOCH=${11:-"10"}
DROPOUT=${12:-"0.2"}
NUM_LAYERS=${13:-"2"}
REDUCTION_LEVEL=${14:-"128,8"}
LR=${15:-"0.0001"}
SPARSE_LR=${16:-'0.00001'}

# cmd="python3 train_dist.py --graph_name ${DATASET} --root /datasets/gnn/dgldata \
# cmd="python3 train_dist.py --graph_name ${DATASET} --root /datasets/gnn/mag240m \
cmd="python3 train_dist.py --graph_name ${DATASET} --root /datasets/gnn/IGB \
--model ${MODEL} --ip_config ip_config_gn70.txt --num_epochs ${EPOCH} \
--batch_size ${BATCH_SIZE} --n_classes ${N_CLASSES} --predict_category ${PREDICT_CATEGORY} \
--eval_every 1 --fan_out ${FANOUT} --num_hidden ${NUM_HIDDEN} --embed_dim ${EMBEDDING_SIZE} \
--preprocess_dir preprocess --num_gpus 4 \
--dgl-sparse --cache-method ${CACHE_METHOD} --num_layers ${NUM_LAYERS} \
--dropout ${DROPOUT} --batch_size_eval ${BATCH_SIZE} --reduction-level ${REDUCTION_LEVEL} \
--lr ${LR} --sparse-lr ${SPARSE_LR} --no-test"

# not freebase
if [ "$DATASET" != "freebase" ]; then
    cmd="${cmd} --ntypes-w-feats ${NTYPES_W_FEATS}"
fi
echo $cmd

module add cuda/11.7

# 1 machine
python3 /gf3/home/jgqj/test_code/hydro/third_party/dgl/tools/launch_heta.py \
    --workspace /gf3/home/jgqj/test_code/hydro \
    --num_trainers 4 \
    --num_samplers 0 \
    --num_servers 1 \
    --part_config preprocess/${DATASET}/${DATASET}.json \
    --ip_config ip_config_gn70.txt \
    "${cmd}" > log/Heta_${MODEL}_${DATASET}_${CACHE_METHOD}_${EMBEDDING_SIZE}_${REDUCTION_LEVEL}.log
