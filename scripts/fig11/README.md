# fig11

Time to accuracy comparison of MeCache and DGL.

## run

First, place the logs from `MeCache` and `DGL` (using nccl) running R-GAT on igbh-medium and R-GCN on MAG into the `log` directory.  
Specifically, logs from `baseline/dgl/run_all_acc.py` and `run_all_acc.py`.  
Second, produce figure 11.

```bash
python draw.py
```

## output

Executing `draw.py` generates two results: the figure `time_to_acc.pdf` (Figure 10) and a console log as shown below.

```
DRGNN 660.41378
DGL 3706.6456000000003
igbn-medium acc speed up: 5.612610930074779
DRGNN 24617.553333333297
DGL 129175.46333333277
mag acc speed up: 5.247290889725554
```