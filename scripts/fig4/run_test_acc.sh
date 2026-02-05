#!/bin/bash

for i in 512 256 128 64 32 16 8 4 2 1; do
    for j in 512 256 128 64 32 16 8 4 2 1; do
        if [ $j -le $i ]; then
            # ./run_gn70.sh rgcn igb-full-small paper 19 1024 paper,author,institute,conference,fos,journal none 256 64 "5,10,15" 3 0.5 3 ${i},${j}
            ./run_reduction.sh rgcn igb-full-small paper 19 1024 paper,author,institute,conference,fos,journal none 256 64 25,20 30 0.5 ${i},${j} /datasets/gnn/IGB ip_config_gn72.txt 0.01 0.01 4 --use_node_projs
        fi
    done
done

# for i in 64 32 16; do
#     ./run_gn70.sh rgat ogbn-mag paper 349 1024 paper none 64 64 '25,25' 10 0.5 2 ${i},8 0.01 0.06
#     ./run_gn70.sh rgcn ogbn-mag paper 349 1024 paper none 64 64 '25,25' 10 0.5 2 ${i},8 0.01 0.06
# done

# ./run_gn70.sh rgat mag240m paper 349 1024 paper none 256 64 '5,10,15' 1000 0.5 3 128,8 0.0001 0.000001