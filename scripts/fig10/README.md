# fig10

Data transfer time comparison of different cache strategies on MAG with R-GCN and 2 GPUs.  
The height of the bars represents the data transfer time (s), and the numbers above the bars indicate the speedup compared to DGL+MR.  
DGL+MR indicates no cache, MeCache-R prefers caching read-only features, MeCache-O prefers caching optimizer states, MeCache-E prefers caching embeddings, and MeCache-EO prefers caching both embeddings and optimizer states.

## run

First, place the logs from `DGL+MR` and output los of `MeCache` using different caching strategy into the `log` directory.  
Specifically, logs from `run_all_dgl_mr.py`, `run_cost_model_for_fig10.py` and `run_all_speed.py`.  
Second, produce figure 10.

```bash
python draw.py
```

## output

Executing `draw.py` generates two results: the figure `cost_model_efficiency.pdf` (Figure 10) and a console log as shown below.

```
DGL+MR featcopy_values: 35.97995, update_values: 93.17283333333334, speedup: 1.0
DRGNN-R featcopy_values: 18.227149999999998, update_values: 103.87433333333333, speedup: 1.057749503179664
DRGNN-E featcopy_values: 32.588033333333335, update_values: 78.79366666666665, speedup: 1.1595511949748778
DRGNN-O featcopy_values: 39.7355, update_values: 47.28783333333333, speedup: 1.484116711992952
DRGNN-EO featcopy_values: 35.39853333333334, update_values: 25.43616666666667, speedup: 2.1230117569961435
DRGNN featcopy_values: 15.240466666666668, update_values: 28.7745, speedup: 2.9342924262883305
```