# fig12

Performance gain analysis for R-GCN. MR indicates the Meta Reduction strategy and FC indicates the Fusion Cache module.

## run

First, place the logs from `DGL` (using nccl) `DGL+MR` and `MeCache` running R-GCN on igbh-medium and MAG into the `log` directory.  
Specifically, logs from `baseline/dgl/run_all.py`, `run_all_dgl_mr.py` and `run_all_speed.py`.  
Second, produce figure 12.

```bash
python draw.py
```

## output

Executing `draw.py` generates two results: the figure `ablation.pdf` (Figure 10) and a console log as shown below.

```
{'ME': {'DGL': 1, '+MR': 4.36651206315545, '+MR+FC': 5.6142034004713395}, 'MAG': {'DGL': 1, '+MR': 2.4261265125794336, '+MR+FC': 5.099132088133317}}
```