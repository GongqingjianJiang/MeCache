import argparse
import os
import numpy as np
import time
from collections import defaultdict

import torch as th
from tqdm import tqdm
import pickle

import dgl
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.load_dataset import load_dataset
# python -u profile_miss_penalty.py --dataset igb-full-small --model gcn --root /datasets/gnn/IGB --budget 17
# python -u profile_miss_penalty.py --dataset mag240m --model gcn --root /datasets/gnn/mag240m --budget 4 --embed-dim 128

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="ogbn-mag")
parser.add_argument("--root", type=str, default="datasets")
parser.add_argument("--out-dir", type=str, default="cache")
parser.add_argument("--budget", type=int, default=4)
parser.add_argument("--embed-dim", type=int, default=64)
parser.add_argument("--fan-out", type=str, default="5,10,15")
parser.add_argument("--model", type=str, default='gcn')
args = parser.parse_args()
print(args)

time_per_ntype = defaultdict(float)
num_nodes_per_ntype = defaultdict(int)
read_time = 0
write_time = 0
embed_ntypes = set()
embed_dim = args.embed_dim
budget = args.budget * 1024 * 1024 * 1024 # budget GB
fanouts = [int(fanout) for fanout in args.fan_out.split(",")]

def load_subtensor(g, seeds, input_nodes, device, load_feat=True):
    """
    Copys features and labels of a set of nodes onto GPU.

    Profile each node type time and number of nodes.
    """
    global read_time, write_time
    for ntype in g.ntypes:
        if ntype in input_nodes:
            th.cuda.synchronize()
            t0 = time.perf_counter()
            g.nodes[ntype].data["feat"][input_nodes[ntype]].to(device)
            th.cuda.synchronize()
            read_time += time.perf_counter() - t0
            time_per_ntype[ntype] += time.perf_counter() - t0
            num_nodes_per_ntype[ntype] += len(input_nodes[ntype])

            if ntype in embed_ntypes:
                # simulate read optimizer states
                th.cuda.synchronize()
                t0 = time.perf_counter()
                g.nodes[ntype].data["os1"][input_nodes[ntype]].to(device)
                g.nodes[ntype].data["os2"][input_nodes[ntype]].to(device)
                th.cuda.synchronize()
                read_time += time.perf_counter() - t0
                time_per_ntype[ntype] += time.perf_counter() - t0

                # simulate write
                new_feat = th.randn(
                    len(input_nodes[ntype]), embed_dim, device=device)
                new_os1 = th.randn(
                    len(input_nodes[ntype]), embed_dim, device=device)
                new_os2 = th.randn(
                    len(input_nodes[ntype]), embed_dim, device=device)
                th.cuda.synchronize()
                t0 = time.perf_counter()
                g.nodes[ntype].data["feat"][input_nodes[ntype]] = new_feat.cpu()
                g.nodes[ntype].data["os1"][input_nodes[ntype]] = new_os1.cpu()
                g.nodes[ntype].data["os2"][input_nodes[ntype]] = new_os2.cpu()
                th.cuda.synchronize()
                write_time += time.perf_counter() - t0
                time_per_ntype[ntype] += time.perf_counter() - t0

# fixed a bug where the cache size is slightly smaller than budget.
# this function is not applied in the experiments in DRGNN.
# mannualy replace `save_cached_node_heta` in line 276 with `save_cached_node_drgnn` to fix this bug.
def save_cached_node_drgnn(budget, ntype2count, cache_ratio_dict, part, method, label):
    feat_shape={
        'ogbn-mag': 128,
        'igb-full-small': 1024,
        'igb-full-medium': 1024,
        'mag240m': 768
    }
    embed_ntypes1={
        'ogbn-mag': ['institution','field_of_study','author'],
        'igb-full-small': [],
        'igb-full-medium': [],
        'mag240m': ['institution','author']
    }
    # how many nodes can be cached
    dtype_size = 2 if args.dataset == 'mag240m' else 4 
    # the scaling factor for the feature size, to fix a bug in the origin version of heta.
    # Specifically, certain circumstances occur when num_cached_nodes > num_nodes, which leads to smaller actual cache size
    # so we set scaling_factor, 
    # num_cached_nodes = int(scaling_factor* ratio* budget / feat_size)
    # scaling_factor*=(1-scaling_factor*ratio*(len(cached_nodes)/num_cached_nodes))/(1-scaling_factor*ratio)
    scaling_factor = 1
    total_size=0
    for ntype, ratio in cache_ratio_dict.items():
        # feat_size = g.nodes[ntype].data["feat"].shape[1] * dtype_size
        feat_size = feat_shape[args.dataset] * dtype_size
        # feat_size = feat_size * 3 if ntype in embed_ntypes else feat_size # embedding + optimizer states
        feat_size = feat_size * 3 if ntype in embed_ntypes1 else feat_size # embedding + optimizer states
        # number of nodes can be cached for this node type
        # scaling_factor > 1 means the actual used cache size is smaller than it should be, need to scale up
        num_cached_nodes = int(scaling_factor* ratio* budget / feat_size)
        count = ntype2count[ntype]
        # cache the hottest nodes
        count = count.numpy()
        # get the cached nodes
        cached_nodes = np.argsort(count)[::-1][:num_cached_nodes]
        print("scaling_factor",scaling_factor)
        # if num_cached_nodes > len(cached_nodes): scaling_factor > 1
        scaling_factor*=(1-scaling_factor*ratio*(len(cached_nodes)/num_cached_nodes))/(1-scaling_factor*ratio)
        print(ntype, len(cached_nodes), num_cached_nodes)
        print(f"cache {len(cached_nodes)} {ntype} nodes, size: {len(cached_nodes) * feat_size / (1024*1024):.2f} MB")
        total_size+=len(cached_nodes) * feat_size / (1024*1024)
        print(f"theoretical hit rate: {np.sum(count[cached_nodes]) / np.sum(count)}")
        dir = f"cache/{label}/{args.dataset}_{method}_{args.model}/{part}"
        os.makedirs(dir, exist_ok=True)
        np.save(os.path.join(dir, f'{ntype}.npy'), cached_nodes)
    print("total_size",total_size)

# this function has a bug, which will lead to slightly smaller cache size than budget.
# replace `save_cached_node_heta` with `save_cached_node_drgnn` to avoid it.
def save_cached_node_heta(budget, ntype2count, cache_ratio_dict, part, method, label):
    feat_shape={
        'ogbn-mag': 128,
        'igb-full-small': 1024,
        'igb-full-medium': 1024,
        'mag240m': 768
    }
    embed_ntypes1={
        'ogbn-mag': ['institution','field_of_study','author'],
        'igb-full-small': [],
        'igb-full-medium': [],
        'mag240m': ['institution','author']
    }
    # how many nodes can be cached
    dtype_size = 2 if args.dataset == 'mag240m' else 4 
    for ntype, ratio in cache_ratio_dict.items():
        # feat_size = g.nodes[ntype].data["feat"].shape[1] * dtype_size
        feat_size = feat_shape[args.dataset] * dtype_size
        # feat_size = feat_size * 3 if ntype in embed_ntypes else feat_size # embedding + optimizer states
        feat_size = feat_size * 3 if ntype in embed_ntypes1 else feat_size # embedding + optimizer states
        # number of nodes can be cached for this node type
        num_cached_nodes = int(ratio* budget / feat_size)
        count = ntype2count[ntype]
        # cache the hottest nodes
        count = count.numpy()
        # get the cached nodes
        cached_nodes = np.argsort(count)[::-1][:num_cached_nodes]
        print(f"cache {len(cached_nodes)} {ntype} nodes, size: {len(cached_nodes) * feat_size / (1024*1024):.2f} MB")
        print(f"theoretical hit rate: {np.sum(count[cached_nodes]) / np.sum(count)}")
        dir = f"cache/{label}/{args.dataset}_{method}_{args.model}/{part}"
        os.makedirs(dir, exist_ok=True)
        np.save(os.path.join(dir, f'{ntype}.npy'), cached_nodes)

if __name__ == "__main__":
    file_name=os.path.join(f"cache/Heta", f'{args.dataset}.pkl')
    if not os.path.isfile(file_name):
        start = time.time()
        g, n_classes, target_node_type, _, _ = load_dataset(args.dataset, args.root, load_feat=True)
        load_time = time.time() - start
        print(f"Load {args.dataset} time: {load_time:.2f}s")

        # if does not have features, create a embedding layer
        for ntype in g.ntypes:
            if "feat" not in g.nodes[ntype].data:
                g.nodes[ntype].data["feat"] = th.randn(
                    g.num_nodes(ntype), embed_dim)
                g.nodes[ntype].data["os1"] = th.randn(
                    g.num_nodes(ntype), embed_dim)
                g.nodes[ntype].data["os2"] = th.randn(
                    g.num_nodes(ntype), embed_dim)
                embed_ntypes.add(ntype)

        device = th.device("cuda" if th.cuda.is_available() else "cpu")
        ntype2count_tensor = {ntype: th.zeros(g.number_of_nodes(ntype), dtype=th.int64, device=device) for ntype in g.ntypes}

        train_nid = {target_node_type: g.nodes[target_node_type].data["train_mask"].nonzero(
            as_tuple=True)[0]}
        sampler = dgl.dataloading.NeighborSampler(fanouts)
        dataloader = dgl.dataloading.DataLoader(
            g,
            train_nid,
            sampler,
            batch_size=1024*4,
            shuffle=True,
            drop_last=True,
            use_uva=False
        )

        num_batches = len(dataloader)
        print("Number of batches: ", num_batches)
        count_time = 0
        start = time.time()
        for input_nodes, seeds, blocks in tqdm(dataloader):
            count_start = time.time()
            for ntype, nid in input_nodes.items():
                ntype2count_tensor[ntype][nid] += 1
            count_time += time.time() - count_start

            load_subtensor(g, seeds, input_nodes,
                        th.device("cuda"), load_feat=True)
    
        tot_time = time.time() - start
        sample_time = tot_time - read_time - write_time - count_time
        # tot_time += load_time
        print(
            f"Total time: {tot_time:.2f}s, read time: {read_time:.2f}s, write time: {write_time:.2f}s, sample time: {sample_time:.2f}s")

        shape_dict = {}
        for ntype in g.ntypes:
            shape_dict[ntype] = g.nodes[ntype].data["feat"].shape

        miss_penalty_ratio = defaultdict(float)
        for ntype in g.ntypes:
            if ntype in embed_ntypes:
                miss_penalty_ratio[ntype] = time_per_ntype[ntype] / \
                    num_nodes_per_ntype[ntype] / shape_dict[ntype][1] / \
                    4 / 3  # embedding + optimizer states
            else:
                miss_penalty_ratio[ntype] = time_per_ntype[ntype] / \
                    num_nodes_per_ntype[ntype] / shape_dict[ntype][1] / 4

        if device == th.device("cuda"):
            ntype2count_tensor_cpu = {ntype: count_tensor.cpu() for ntype, count_tensor in ntype2count_tensor.items()}
        else:
            ntype2count_tensor_cpu = ntype2count_tensor

        print("Time per node type: ", time_per_ntype)
        print("Number of nodes per node type: ", num_nodes_per_ntype)
        # shape
        print("Shape of each node type: ", shape_dict)
        print("Miss penalty ratio: ", miss_penalty_ratio)
        
        with open(file_name,'wb') as f:
            pickle.dump([ntype2count_tensor_cpu,miss_penalty_ratio],f, protocol=4)
            print(f'{file_name} dumped!')
            print("pkl dump: ", ntype2count_tensor_cpu,miss_penalty_ratio)
    
    ## calculate cache size ratio for each node type
    # ogbn-mag for 2 parts
    # ntype_list=[ntype for ntype in g.ntypes]
    # part_dict = {
    #     'part0': ntype_list,
    # }

    with open(file_name,'rb') as f:
        ntype2count_tensor_cpu, miss_penalty_ratio=pickle.load(f)
        print("pkl read: ", ntype2count_tensor_cpu,miss_penalty_ratio)
    ntypes_w_feats = {
        'ogbn-mag': ['paper'],
        'igb-full-small': ['paper','author','institute','conference','fos','journal'],
        'igb-full-medium': ['paper','author','institute','conference','fos','journal'],
        'mag240m': ['paper', 'institution','author']
    }
    part_dict={
        'part0': ntypes_w_feats[args.dataset]
    }
    for part, ntypes in part_dict.items():
        cache_ratio_dict = {}
        denominator = sum([ntype2count_tensor_cpu[ntype].sum() * miss_penalty_ratio[ntype] for ntype in ntypes])
        for ntype in ntypes:
            cache_ratio_dict[ntype] = ntype2count_tensor_cpu[ntype].sum() * miss_penalty_ratio[ntype] / denominator
    
        print("Cache size ratio: ", cache_ratio_dict)

        save_cached_node_heta(budget, ntype2count_tensor_cpu, cache_ratio_dict, part, "miss_penalty", "Heta")

