# fig9
Data communication volume breakdown and data transmission time comparison in the embedding update phase for R-GAT with 2 and 4 GPUs.  
The height of the bars represents the total data communication volume (GB) (labeled with text), while the triangles represent the data transfer time.

## run

First, place the logs from `Heta` (using nccl) and `DGL+FC` into the `log` directory.  
Specifically, logs from `baseline/heta/run_all.py` and `run_all_dgl_fc.py`.  
Second, produce figure 9.

```bash
python draw.py
```

## output

Executing `draw.py` generates two results: the figure `consistency.pdf` (Figure 9) and a console log as shown below.

```
RGCN
OM 2 Heta {'CPU-GPU': 27.896357822418214, 'A2A': 0, 'time': 25.982999999999997}
OM 2 DRGNN {'CPU-GPU': 0.0, 'A2A': 27.894949674606323, 'time': 14.241166666666667}
OM 4 Heta {'CPU-GPU': 20.444632598331996, 'A2A': 0, 'time': 10.418846153846154}
OM 4 DRGNN {'CPU-GPU': 0.0, 'A2A': 61.30904656648636, 'time': 8.12757142857143}
MAG 2 Heta {'CPU-GPU': 86.06222315010427, 'A2A': 0, 'time': 45.763999999999996}
MAG 2 DRGNN {'CPU-GPU': 21.814985997991545, 'A2A': 38.25578167367459, 'time': 26.803800000000003}
MAG 4 Heta {'CPU-GPU': 79.82380676327007, 'A2A': 0, 'time': 25.3123}
MAG 4 DRGNN {'CPU-GPU': 36.59070226661123, 'A2A': 92.61177304042928, 'time': 20.110941176470586}
2 gpu: CPU-GPU PCIe bandwidth 1.073638833945973
2 gpu: A2A PCIe bandwidth 1.9587545267550404
4 gpu: CPU-GPU PCIe bandwidth 1.9622741613076595
4 gpu: A2A PCIe bandwidth 7.543341464950071
2 gpu: CPU-GPU PCIe bandwidth 1.8805660158662765
2 gpu: A2A PCIe bandwidth 2.516235673219719
4 gpu: CPU-GPU PCIe bandwidth 3.153558023698758
4 gpu: A2A PCIe bandwidth 10.885320249631787

OM 2 Heta {'CPU-GPU': {'emb': 27.896357822418214, 'opt': 0.0, 'a2a': 0}, 'A2A': 0, 'time': 25.982999999999997}
OM 2 DRGNN {'CPU-GPU': {'emb': 0.0, 'opt': 0.0, 'a2a': 27.894949674606323}, 'A2A': 27.894949674606323, 'time': 14.241166666666667}
OM 4 Heta {'CPU-GPU': {'emb': 20.444632598331996, 'opt': 0.0, 'a2a': 0}, 'A2A': 0, 'time': 10.418846153846154}
OM 4 DRGNN {'CPU-GPU': {'emb': 0.0, 'opt': 0.0, 'a2a': 61.30904656648636}, 'A2A': 61.30904656648636, 'time': 8.12757142857143}
MAG 2 Heta {'CPU-GPU': {'emb': 50.01264547383189, 'opt': 36.049577676272385, 'a2a': 0}, 'A2A': 0, 'time': 45.763999999999996}
MAG 2 DRGNN {'CPU-GPU': {'emb': 5.488391067018496, 'opt': 16.32659493097305, 'a2a': 38.25578167367459}, 'A2A': 38.25578167367459, 'time': 26.803800000000003}
MAG 4 Heta {'CPU-GPU': {'emb': 43.96330037809696, 'opt': 35.86050638517312, 'a2a': 0}, 'A2A': 0, 'time': 25.3123}
MAG 4 DRGNN {'CPU-GPU': {'emb': 8.22555414948183, 'opt': 28.365148117129408, 'a2a': 92.61177304042928}, 'A2A': 92.61177304042928, 'time': 20.110941176470586}

RGAT
OM 2 Heta {'CPU-GPU': 27.897579860687255, 'A2A': 0, 'time': 26.19388888888889}
OM 2 DRGNN {'CPU-GPU': 0.0, 'A2A': 27.89422047138214, 'time': 14.791857142857143}
OM 4 Heta {'CPU-GPU': 20.437904728783504, 'A2A': 0, 'time': 11.303615384615386}
OM 4 DRGNN {'CPU-GPU': 0.0, 'A2A': 61.30354719895582, 'time': 9.110411764705882}
MAG 2 Heta {'CPU-GPU': 86.05576167950363, 'A2A': 0, 'time': 46.1212}
MAG 2 DRGNN {'CPU-GPU': 41.166153938199045, 'A2A': 35.600632678520675, 'time': 30.167399999999997}
MAG 4 Heta {'CPU-GPU': 79.82239147782953, 'A2A': 0, 'time': 24.92723076923077}
MAG 4 DRGNN {'CPU-GPU': 39.15502049692242, 'A2A': 92.13871124217795, 'time': 19.597266666666663}
2 gpu: CPU-GPU PCIe bandwidth 1.065041543812956
2 gpu: A2A PCIe bandwidth 1.8857821706892304
4 gpu: CPU-GPU PCIe bandwidth 1.8080856463499462
4 gpu: A2A PCIe bandwidth 6.728954605152792
2 gpu: CPU-GPU PCIe bandwidth 1.8658612889409563
2 gpu: A2A PCIe bandwidth 4.392653815187691
4 gpu: CPU-GPU PCIe bandwidth 3.2022165725829144
4 gpu: A2A PCIe bandwidth 12.502212450747804

OM 2 Heta {'CPU-GPU': {'emb': 27.897579860687255, 'opt': 0.0, 'a2a': 0}, 'A2A': 0, 'time': 26.19388888888889}
OM 2 DRGNN {'CPU-GPU': {'emb': 0.0, 'opt': 0.0, 'a2a': 27.89422047138214}, 'A2A': 27.89422047138214, 'time': 14.791857142857143}
OM 4 Heta {'CPU-GPU': {'emb': 20.437904728783504, 'opt': 0.0, 'a2a': 0}, 'A2A': 0, 'time': 11.303615384615386}
OM 4 DRGNN {'CPU-GPU': {'emb': 0.0, 'opt': 0.0, 'a2a': 61.30354719895582}, 'A2A': 61.30354719895582, 'time': 9.110411764705882}
MAG 2 Heta {'CPU-GPU': {'emb': 50.00904997146659, 'opt': 36.04671170803705, 'a2a': 0}, 'A2A': 0, 'time': 46.1212}
MAG 2 DRGNN {'CPU-GPU': {'emb': 10.797752167677881, 'opt': 30.368401770521164, 'a2a': 35.600632678520675}, 'A2A': 35.600632678520675, 'time': 30.167399999999997}
MAG 4 Heta {'CPU-GPU': {'emb': 43.964574425936995, 'opt': 35.85781705189253, 'a2a': 0}, 'A2A': 0, 'time': 24.92723076923077}
MAG 4 DRGNN {'CPU-GPU': {'emb': 8.56050525221635, 'opt': 30.594515244706066, 'a2a': 92.13871124217795}, 'A2A': 92.13871124217795, 'time': 19.597266666666663}



Data Size: 198MB per GPU, World Size: 2
GPU->CPU:     Time: 0.1181s, Total Volume: 396.00MB, Bandwidth: 3352.68 MB/s (3.27 GB/s)
All-to-All:   Time: 0.0071s, Total Volume: 198.00MB, Bandwidth: 27738.73 MB/s (27.09 GB/s)

Data Size: 199MB per GPU, World Size: 4
GPU->CPU:     Time: 0.1282s, Total Volume: 796.00MB, Bandwidth: 6209.35 MB/s (6.06 GB/s)
All-to-All:   Time: 0.0197s, Total Volume: 597.00MB, Bandwidth: 30345.86 MB/s (29.63 GB/s)



RGAT
OM:
  2 GPU:
    Heta: Total=27.9GB, emb=27.9GB, opt=0.0GB, A2A=0.0GB, Time=26.2s
    MeCache: Total=27.9GB, emb=0.0GB, opt=0.0GB, A2A=27.9GB, Time=14.8s
  4 GPU:
    Heta: Total=20.4GB, emb=20.4GB, opt=0.0GB, A2A=0.0GB, Time=11.3s
    MeCache: Total=61.3GB, emb=0.0GB, opt=0.0GB, A2A=61.3GB, Time=9.1s

MAG:
  2 GPU:
    Heta: Total=86.1GB, emb=50.0GB, opt=36.0GB, A2A=0.0GB, Time=46.1s
    MeCache: Total=76.8GB, emb=10.8GB, opt=30.4GB, A2A=35.6GB, Time=30.2s
  4 GPU:
    Heta: Total=79.8GB, emb=44.0GB, opt=35.9GB, A2A=0.0GB, Time=24.9s
    MeCache: Total=131.3GB, emb=8.6GB, opt=30.6GB, A2A=92.1GB, Time=19.6s

RGCN
OM:
  2 GPU:
    Heta: Total=27.9GB, emb=27.9GB, opt=0.0GB, A2A=0.0GB, Time=26.0s
    MeCache: Total=27.9GB, emb=0.0GB, opt=0.0GB, A2A=27.9GB, Time=14.2s
  4 GPU:
    Heta: Total=20.4GB, emb=20.4GB, opt=0.0GB, A2A=0.0GB, Time=10.4s
    MeCache: Total=61.3GB, emb=0.0GB, opt=0.0GB, A2A=61.3GB, Time=8.1s

MAG:
  2 GPU:
    Heta: Total=86.1GB, emb=50.0GB, opt=36.0GB, A2A=0.0GB, Time=45.8s
    MeCache: Total=60.1GB, emb=5.5GB, opt=16.3GB, A2A=38.3GB, Time=26.8s
  4 GPU:
    Heta: Total=79.8GB, emb=44.0GB, opt=35.9GB, A2A=0.0GB, Time=25.3s
    MeCache: Total=129.2GB, emb=8.2GB, opt=28.4GB, A2A=92.6GB, Time=20.1s
```