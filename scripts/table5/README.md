# table5

CPU-GPU PCIe communication volume (GB) and data transfer time (s) comparison on R-GAT with 2 and 4 GPU.  
The communication and time are broken down into feature retrieval ($feat.$) and embedding update ($emb.$) phases.  
The last column shows the communication reduction compared to baselines.

## run

First, place the logs from `DGL` and `Heta` (using nccl) and `DGL+FC` running R-GAT on ogbn-mag, mag240m into the `log` directory.  
Specifically, logs from `baseline/dgl/run_all.py`, `baseline/heta/run_all.py` and `run_all_dgl_fc.py`.  
Second, produce table 5.

```bash
python draw.py
```

## output

Executing `draw.py` generates the console log as shown below.  

```
RGAT-MAG 1 DRGNN 15.338893770003923 2.902809078491888 Heta 12.88548526228228 6.898311716649085 DGL 100 100
RGAT-MAG 2 DRGNN 14.82759399841097 2.02718984233467 Heta 14.542428993665233 7.305968610586468 DGL 100 100
RGAT-MAG 4 DRGNN 15.661752818358748 1.6441052877156577 Heta 15.367261148170652 7.882343299144902 DGL 100 100

RGAT-OM 1 DRGNN 0.0 0.0 Heta 0.0 9.984027685830071 DGL 100 100
RGAT-OM 2 DRGNN 0.0 0.0 Heta 4.991990755184853 9.984159625338036 DGL 100 100
RGAT-OM 4 DRGNN 0.0 0.0 Heta 7.489721512733984 9.982080019600472 DGL 100 100

DGL & 40.1+59.9 & 40.1+59.9 & 76.5+23.5 & 76.5+23.5 & \\
Heta & 5.0+10.0 & 7.5+10.0 & 14.5+7.3 & 15.4+7.9 & \\
DRGNN & 0.0+0.0 & 0.0+0.0 & 14.8+2.0 & 15.7+1.6 & \\

DGL & 0.0\%,0.0\% & 0.0\%,0.0\% & 0.0\%,0.0\% & 0.0\%,0.0\% & \\
Heta & 80.1\%,100.0\% & 70.1\%,100.0\% & 73.1\%,82.6\% & 68.0\%,79.7\% & \\
DRGNN & 100.0\%,100.0\% & 100.0\%,100.0\% & 82.6\%,92.3\% & 81.6\%,94.0\% & \\


data tranfer volume, forward
DGL & 140.8 & 140.3 & 871.5 & 869.9 & \\
Heta & 17.5 & 26.2 & 165.6 & 174.7 & \\
DRGNN & 0.0 & 0.0 & 168.8 & 178.0 & \\

cache hit rate, forward
DGL & 0.0\% & 0.0\% & 0.0\% & 0.0\% & \\
Heta & 80.1\% & 70.1\% & 73.1\% & 68.0\% & \\
DRGNN & 100.0\% & 100.0\% & 82.6\% & 81.6\% & \\

data tranfer volume, backward
DGL & 210.4 & 209.6 & 267.2 & 266.8 & \\
Heta & 35.1 & 34.9 & 83.2 & 89.6 & \\
DRGNN & 0.0 & 0.0 & 23.1 & 18.7 & 

cache hit rate, backward
DGL & 0.0\% & 0.0\% & 0.0\% & 0.0\% & \\
Heta & 100.0\% & 100.0\% & 82.6\% & 79.7\% & \\
DRGNN & 100.0\% & 100.0\% & 92.3\% & 94.0\% & \\

data tranfer volume, total
DGL & 351.2 & 349.9 & 1138.7 & 1136.8 & \\
Heta & 52.6 & 61.1 & 248.8 & 264.3 & \\
DRGNN & 0.0 & 0.0 & 191.9 & 196.7 & \\

table:
DGL & 2/4 & 140 & 0.0\% & 210 & 0.0\% & 100\% \\
\multirow{2}{*}{ Heta } & 2 & 18 & 80.1\% & 35 & 100.0\% & 100\% \\
 & 4 & 26 & 70.1\% & 35 & 100.0\% & 100\% \\
\multirow{2}{*}{ DGL+FC } & 2 & 0 & 100.0\% & 0 & 100.0\% & 0\% \\
 & 4 & 0 & 100.0\% & 0 & 100.0\% & 0\% \\

DGL & 2/4 & 872 & 0.0\% & 267 & 0.0\% & 83\% \\
\multirow{2}{*}{ Heta } & 2 & 166 & 73.1\% & 83 & 82.6\% & 23\% \\
 & 4 & 175 & 68.0\% & 90 & 79.7\% & 26\% \\
\multirow{2}{*}{ DGL+FC } & 2 & 169 & 82.6\% & 23 & 92.3\% & 0\% \\
 & 4 & 178 & 81.6\% & 19 & 94.0\% & 0\% \\


RGCN
\multirow{3}{*}{2} & DGL & 22 & 61 & 141 & 167  & 100\% \\
 & Heta & 7 & 26 & 18 & 28  & 100\% \\
 & DGL+FC & 4 & 14 & 0 & 0  & - \\
\midrule
\multirow{3}{*}{4} & DGL & 14 & 21 & 141 & 123  & 100\% \\
 & Heta & 4 & 10 & 27 & 20  & 100\% \\
 & DGL+FC & 2 & 8 & 0 & 0  & - \\
\midrule

\multirow{3}{*}{2} & DGL & 393 & 90 & 874 & 246  & 84\% \\
 & Heta & 89 & 46 & 180 & 86  & 33\% \\
 & DGL+FC & 85 & 27 & 156 & 22  & - \\
\midrule
\multirow{3}{*}{4} & DGL & 230 & 29 & 872 & 210  & 82\% \\
 & Heta & 59 & 25 & 189 & 80  & 29\% \\
 & DGL+FC & 46 & 20 & 155 & 37  & - \\
\midrule

RGAT
\multirow{3}{*}{2} & DGL & 21 & 60 & 141 & 167  & 100\% \\
 & Heta & 7 & 26 & 18 & 28  & 100\% \\
 & DGL+FC & 4 & 15 & 0 & 0  & - \\
\midrule
\multirow{3}{*}{4} & DGL & 15 & 22 & 141 & 123  & 100\% \\
 & Heta & 5 & 11 & 27 & 20  & 100\% \\
 & DGL+FC & 2 & 9 & 0 & 0  & - \\
\midrule

\multirow{3}{*}{2} & DGL & 443 & 100 & 874 & 246  & 81\% \\
 & Heta & 93 & 46 & 180 & 86  & 21\% \\
 & DGL+FC & 88 & 30 & 169 & 41  & - \\
\midrule
\multirow{3}{*}{4} & DGL & 229 & 30 & 872 & 210  & 81\% \\
 & Heta & 51 & 25 & 189 & 80  & 25\% \\
 & DGL+FC & 47 & 20 & 163 & 39  & - \\
\midrule
```