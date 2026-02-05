import os
import itertools

num_workers=['4','2','1']
models = ['rgat','rgcn']
datasets = ['ogbn-mag', 'igb-full-small', 'igb-full-medium', 'mag240m']
cache_methods = ['miss_penalty']
backends = ['nccl', 'gloo']
embedding_sizes = [64]
hidden_sizes = 256
ip_config={
    'ogbn-mag': 'ip_config_gn71.txt',
    'igb-full-small': 'ip_config_gn71.txt',
    'igb-full-medium': 'ip_config_gn71.txt',
    'mag240m': 'ip_config_gn80.txt',
}
root={
    'ogbn-mag': '/datasets/gnn/dgldata',
    'igb-full-small': '/datasets/gnn/dataset/IGB/',
    'igb-full-medium': '/datasets/gnn/dataset/IGB/',
    'mag240m': '/datasets/gnn/mag240m'
}
predict_category = {
    'ogbn-mag': 'paper',
    'igb-full-small': 'paper',
    'igb-full-medium': 'paper',
    'mag240m': 'paper'
}
number_of_classes = {
    'ogbn-mag': 349,
    'igb-full-small': 19,
    'igb-full-medium': 19,
    'mag240m': 153
}
batch_size = {
    'ogbn-mag': 1024,
    'igb-full-small': 1024,
    'igb-full-medium': 1024,
    'mag240m': 1024
}
ntypes_w_feats = {
    'ogbn-mag': ['paper'],
    'igb-full-small': ['paper','author','institute','conference','fos','journal'],
    'igb-full-medium': ['paper','author','institute','conference','fos','journal'],
    'mag240m': ['paper']
}
fan_out='5,10,15'
epochs=3
dropout=0.5
num_layers=3

# ./run.sh rgcn igb-full-small paper 2983 1024 paper,author,institute,conference,fos,journal none 256 64 10
for num_worker,dataset, model, cache_method, embedding_size, backend in itertools.product(num_workers, datasets, models, cache_methods, embedding_sizes, backends):
    if backend=='gloo' and num_worker!='4':
        continue
    print(f"Running {model} on {dataset}")
    cmd = f"./run.sh {model} {dataset} {predict_category[dataset]}"
    cmd += f" {number_of_classes[dataset]} {batch_size[dataset]}"
    if len(ntypes_w_feats[dataset]) > 0:
        cmd += f" {','.join(ntypes_w_feats[dataset])}"
    cmd += f" {cache_method} {hidden_sizes} {embedding_size} {epochs} {num_worker} {ip_config[dataset]} {backend}"
    print(cmd)
    # os.system(cmd)
