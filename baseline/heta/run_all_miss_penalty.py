import os
import itertools

models = ['rgcn', 'rgat']
datasets = ['ogbn-mag','igb-full-small', 'igb-full-medium','mag240m']
root={
    'ogbn-mag': '/datasets/gnn/dgldata',
    'igb-full-small': '/datasets/gnn/dataset/IGB',
    'igb-full-medium': '/datasets/gnn/dataset/IGB',
    'mag240m': '/datasets/gnn/mag240m'
}
budget={
    'igb-full-medium': {'rgat':6,'rgcn':14},
    'igb-full-small': {'rgat':2,'rgcn':9},
    'mag240m': {'rgat':13,'rgcn':18},
    'ogbn-mag': {'rgat':16,'rgcn':20},
}
embedding_sizes = 64

# python -u profile_miss_penalty.py --dataset mag240m --model gcn --root /datasets/gnn/mag240m --budget 4 --embed-dim 128
for dataset, model in itertools.product(datasets, models):
    print(f"Profiling {model} on {dataset} with budget {budget[dataset][model]} GB")
    cmd = f"srun python -u profile_miss_penalty.py --dataset {dataset} "
    cmd += f"--model {model} --root {root[dataset]} "
    cmd += f"--budget {budget[dataset][model]} --embed-dim {embedding_sizes}"
    print(cmd)
    # os.system(cmd)
