#!/bin/bash

python -u partition_graph.py --root /datasets/gnn/dgldata --dataset ogbn-mag --num_parts 1 --output partitions/heta

python -u partition_graph.py --root /datasets/gnn/dataset/IGB --dataset igb-full-small --num_parts 1 --output partitions/heta

python -u partition_graph.py --root /datasets/gnn/dataset/IGB --dataset igb-full-medium --num_parts 1 --output partitions/heta

python -u partition_graph.py --root /datasets/gnn/mag240m --dataset mag240m --num_parts 1 --output partitions/heta
