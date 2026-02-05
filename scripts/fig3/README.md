# fig3
HGNN training time breakdown in DGL and Heta on four datasets using R-GCN.  
The dataset properties and training settings are listed in Table I and Table II.

## run

First, place the logs from `DGL` and `Heta` (using gloo) using 4GPU into the `log` directory.  
Specifically, logs from `baseline/dgl/run_all.py` and `baseline/heta/run_all.py`.  
Second, produce figure 3.

```bash
python draw.py
```

## Output
Executing `draw.py` generates two results: the figure `normalized_epoch.pdf` (Figure 3) and a console log as shown below.

```
OM DGL
sample 0.1925264756390654, featcopy 0.18930068858624374, update 0.47660974290967434, train 0.1415630928650166
OM Heta
sample 0.24733516511099438, featcopy 0.08800341812835161, update 0.47996033007696365, train 0.1847010866836904
MAG DGL
sample 0.06389939364901043, featcopy 0.697552513173899, update 0.15204901327656606, train 0.0864990799005244
MAG Heta
sample 0.12574905422026825, featcopy 0.46913521369094957, update 0.29019697875456896, train 0.1149187533342132
SM DGL
sample 0.08702491369376562, featcopy 0.7747861268366304, update 0.003958545944881673, train 0.13423041352472231
SM Heta
sample 0.3266923688001831, featcopy 0.2819937704491291, update 0.016771651953018052, train 0.3745422087976697
ME DGL
sample 0.08562144066249074, featcopy 0.7941352618789748, update 0.0031042379734927874, train 0.11713905948504165
ME Heta
sample 0.10810681152158984, featcopy 0.7358657803589511, update 0.004527485980519599, train 0.15149992213893954

average DGL: total: 0.7728740326450907 feature retrieval:  0.6139436476189369 embedding update 0.15893038502615373
average Heta: total: 0.5916136573481129 feature retrieval:  0.3937495456568454 embedding update 0.19786411169126755

max DGL: total: 0.849601526450465 feature retrieval:  0.7941352618789748 embedding update 0.47660974290967434
max Heta: total: 0.7593321924455185 feature retrieval:  0.7358657803589511 embedding update 0.47996033007696365
```