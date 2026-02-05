import os
import itertools

models = ['rgcn', 'rgat']
datasets = ['ogbn-mag', 'igb-full-small', 'igb-full-medium', 'mag240m']
embedding_sizes = [64]
cache_methods=['none']
hidden_sizes = {
    'ogbn-mag': 64,
    'igb-full-small': 256,
    'igb-full-medium': 256,
    'mag240m': 256
}
root={
    'ogbn-mag': '/datasets/gnn/dgldata',
    'igb-full-small': '/datasets/gnn/IGB',
    'igb-full-medium': '/datasets/gnn/IGB',
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
batch_size = 1024
ntypes_w_feats = {
    'ogbn-mag': ['paper'],
    'igb-full-small': ['paper','author','institute','conference','fos','journal'],
    'igb-full-medium': ['paper','author','institute','conference','fos','journal'],
    'mag240m': ['paper']
}
fan_out = {
    'ogbn-mag': '25,25',
    'igb-full-small': '25,20',
    'igb-full-medium': '25,20',
    'mag240m': '5,10,15',
}
ip_config={
    'ogbn-mag': 'ip_config_gn72.txt',
    'igb-full-small': 'ip_config_gn72.txt',
    'igb-full-medium': 'ip_config_gn72.txt',
    'mag240m': 'ip_config_gn80.txt',
}
epochs={
    'ogbn-mag': 10,
    'igb-full-small': 30,
    'igb-full-medium': 30,
    'mag240m': 1000,
}
lrs={
    'ogbn-mag': 1e-2,
    'igb-full-small': 1e-2,
    'igb-full-medium': 1e-2,
    'mag240m': 1e-4,
}
sp_lrs={
    'ogbn-mag': 6e-2,
    'igb-full-small': 1e-2,
    'igb-full-medium': 1e-2,
    'mag240m': 1e-5,
}
num_layers={
    'ogbn-mag': 2,
    'igb-full-small': 2,
    'igb-full-medium': 2,
    'mag240m': 3,
}
dropout=0.5

# ./run_gn70.sh rgcn igb-full-small paper 2983 1024 paper,author,institute,conference,fos,journal miss_penalty 256 64 "5,10,15" 1000 0.5 3 128,8
for dataset, model, cache_method, embedding_size in itertools.product(datasets, models, cache_methods, embedding_sizes):
    print(f"Running {model} on {dataset}")
    cmd = f"./run_acc.sh {model} {dataset} {predict_category[dataset]} {number_of_classes[dataset]} {batch_size}"
    if len(ntypes_w_feats[dataset]) > 0:
        cmd += f" {','.join(ntypes_w_feats[dataset])}"
    cmd += f" {cache_method} {hidden_sizes[dataset]} {embedding_size} {epochs[dataset]}"
    cmd += f" {fan_out[dataset]} {ip_config[dataset]} {num_layers[dataset]} {lrs[dataset]} {sp_lrs[dataset]}"
    print(cmd)
    # os.system(cmd)
