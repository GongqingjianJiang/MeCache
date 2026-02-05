#!/bin/bash

for i in 512 256 128 64 32 16 8 4 2 1; do
    for j in 512 256 128 64 32 16 8 4 2 1; do
# for i in 512 256 128 64 32 16 8 4 2 1; do
#     for j in 4 2 1; do
        if [ $j -le $i ]; then
            # srun --reservation=256 python -u ../../preprocess.py --dataset igb-full-small --out-dir ../../preprocess --root /datasets/gnn/IGB --reduction-level ${i},${j}
            srun --reservation=256 python -u ../../preprocess.py --dataset igb-full-small --out-dir ../../preprocess --root /datasets/gnn/IGB --reduction-level ${i},${j}
        fi
    done
done

# for i in 64 32 16; do
#     srun --reservation=jgqj python -u ../../preprocess.py --dataset ogbn-mag --out-dir ../../preprocess --root /datasets/gnn/dgldata --reduction-level ${i},8
# done
