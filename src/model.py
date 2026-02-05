"""hgnn model (R-GCN, R-GAT, and HGT)"""
import math
import time
from typing import Dict, List
import os

import dgl
import dgl.function as fn
import numpy as np
import torch
import torch.distributed as dist
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
from dgl import apply_each
from dgl.distributed.graph_partition_book import NodePartitionPolicy
from dgl.distributed.kvstore import get_kvstore
from dgl.nn.functional import edge_softmax
from dgl.nn.pytorch import GATConv, GraphConv, HeteroGraphConv

from .gpu_cache import GPUCache
from .reduction import load_reductioned_feat,Feat
from .distributed.sparse_emb import DistEmbedding

# bfs meta图，从seed节点开始，根据meta图(canonical_etypes)bfs，筛出每层需要的模块；
# 用于减少冗余模型计算量，但会导致层间计算的不均衡
# 主要是ddp需要，模型中不能有没有参加loss计算的模块，因此删掉多余模块
def _meta_bfs(num_layers, canonical_etypes, seeds):
    # print("_meta_bfs", num_layers, canonical_etypes, seeds)
    rev_layers=[]
    if not isinstance(seeds, list):
        seeds=[seeds]
    nodes=seeds
    for i in range(num_layers):
        layer_i=[]
        for node in nodes:
            for canonical_etype in canonical_etypes:
                if canonical_etype[2]==node:
                    layer_i.append(canonical_etype)
        rev_layers.append(layer_i)
        nodes=[]
        for canonical_etype in layer_i:
            if canonical_etype[0] not in nodes:
                nodes.append(canonical_etype[0])
    
    return rev_layers

def _broadcast_layers(layers: List[nn.Module], src=0):
    """Function to broadcast the parameters from the given source rank.
    """
    if not dist.is_initialized():
        # no need to broadcast if not using distributed training
        return 

    for layer in layers:
        if isinstance(layer, nn.Parameter):
            dist.broadcast(layer.data, src=src)
        else:
            for p in layer.parameters():
                if p.requires_grad:
                    dist.broadcast(p.data, src=src)

def _broadcast_cache(caches: List[torch.tensor], src=0):
    """Function to broadcast the parameters from the given source rank.
    """
    if not dist.is_initialized():
        # no need to broadcast if not using distributed training
        return 
    for cache in caches:
        print(f"rank {dist.get_rank()}, cache id=",id(cache))
        if isinstance(cache, torch.Tensor):
            dist.broadcast(cache, src=src)
        else:
            print(f"cache {cache} is no tensor!")

def init_emb(shape, dtype):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)
    print("init emb shape: ", shape)
    arr = torch.zeros(shape, dtype=dtype)
    nn.init.uniform_(arr, -1.0, 1.0)
    return arr

class DistEmbedLayer(nn.Module):
    r"""Embedding layer for featureless heterograph.
    Parameters
    ----------
    dev_id : int
        Device to run the layer.
    g : DistGraph
        training graph
    embed_size : int
        Output embed size
    dgl_sparse_emb: bool
        Whether to use DGL sparse embedding
        Default: False
    embed_name : str, optional
        Embed name
    """

    def __init__(
        self,
        dev_id,
        g,
        embed_size,
        ntypes_w_feat,
        dataset,
        dgl_sparse_emb=False,
        feat_name="feat",
        embed_name="node_emb",
        partition_book=None,
        predict_category=None, 
        cache_method='none',
        reduction_level=None,
        use_node_projs=True,
        args=None
    ):
        super(DistEmbedLayer, self).__init__()
        self.dev_id = dev_id
        self.embed_size = embed_size
        self.embed_name = embed_name
        self.feat_name = feat_name
        self.g = g
        self.ntypes_w_feat = ntypes_w_feat
        self.predict_category = predict_category
        self.dgl_sparse_emb = dgl_sparse_emb
        self.dataset = dataset
        self.use_node_projs=use_node_projs
        
        self.node_projs = nn.ModuleDict()
        ntypes = partition_book.ntypes if partition_book is not None else g.ntypes
        ntypes_wo_feat = set(ntypes) - set(ntypes_w_feat)
        ntypes_w_feat = sorted(set(ntypes_w_feat) & set(g.ntypes))
        print(f"rank {dist.get_rank()} ntypes_w_feat: {ntypes_w_feat}")
        if dist.get_rank()==0:
            print(f"self.use_node_projs={self.use_node_projs}")

        self._proj_time = 0
        self._fetch_feat_time = 0

        self._cache_read_hit_rate = {ntype: [] for ntype in ntypes}
        # self._cache_write_hit_rate = {ntype: [] for ntype in ntypes}
        self._local_world_size = int(os.environ['LOCAL_WORLD_SIZE'])
        self._machine_id = dist.get_rank() // self._local_world_size
        self._gpu_id = dist.get_rank() % self._local_world_size
        self._gpu_caches = {}
        self.feat = {}
        self._gen_feat_count = [0,0]  # count for feature generated and feature extracted

        self.reduction_level=None
        if reduction_level!=None:
            self.reduction_level= [int(level) for level in reduction_level.split(",")]
            dir_name=args.graph_name+"_"+reduction_level.replace(",","-")
            raw_feat=load_reductioned_feat(os.path.join(args.preprocess_dir, args.graph_name,dir_name))
            self.feat={ntype:raw_feat[ntype] for ntype in ntypes_w_feat}

        label = 'dgl' if partition_book is None else 'Heta'
        for ntype in ntypes_w_feat:
            if partition_book is not None:
                part_policy = NodePartitionPolicy(partition_book, ntype=ntype)
            else:
                part_policy = g.get_node_partition_policy(ntype) 
            
            if self.reduction_level==None:
                if dataset == 'mag240m' and label != 'dgl':
                    self.feat[ntype] = np.load('/datasets/gnn/mag240m/mag240m_kddcup2021/processed/paper/node_feat.npy', mmap_mode='r')
                    if use_node_projs:
                        self.node_projs[ntype] = nn.Linear(
                            768, embed_size
                        )
                elif ntype in g.ntypes and feat_name in g.nodes[ntype].data:
                    if use_node_projs:
                        self.node_projs[ntype] = nn.Linear(
                            g.nodes[ntype].data[feat_name].shape[1], embed_size
                        )
                elif dataset == 'igb-het':
                    if use_node_projs:
                        self.node_projs[ntype] = nn.Linear(
                            1024, embed_size
                        )
                if use_node_projs:
                    nn.init.xavier_uniform_(self.node_projs[ntype].weight)
                print("node {} has data {}".format(ntype, feat_name))
            else:
                for dim in self.reduction_level:
                    # check if there is any feat, avoid redundant param
                    feat_shape=self.feat[ntype].shape(dim)
                    if feat_shape[0]!=0:
                        if use_node_projs:
                            self.node_projs[f'{ntype}_{dim}']=nn.Linear(
                                feat_shape[1], embed_size
                            )
                        if use_node_projs:
                            nn.init.xavier_uniform_(self.node_projs[f'{ntype}_{dim}'].weight)
                        print("node {} has data {}, shape {}".format(ntype, feat_name, feat_shape))

            if cache_method != 'none' and self.reduction_level==None:
                part_policy = None
                if partition_book is not None:
                    part_policy = NodePartitionPolicy(partition_book, ntype=ntype)
                else:
                    part_policy = g.get_node_partition_policy(ntype)
                part_size = part_policy.get_part_size()
                print(f"Rank {dist.get_rank()} part_size for {ntype}: {part_size}")
                if dataset == 'mag240m' and label != 'dgl':
                    feat_size = self.feat[ntype].shape[1]
                    feat_dtype = torch.float16
                else:
                    feat_size = g.nodes[ntype].data[feat_name].shape[1]
                    feat_dtype = g.nodes[ntype].data[feat_name].dtype

                try:
                    # cache_nodes = np.load(f"cache/{label}/{dataset}_{cache_method}/part{self._machine_id}/{ntype}.npy")
                    cache_nodes = np.load(f"cache/drgnn/{dataset}_{cache_method}_{args.model}_{args.num_gpus}/{ntype}.npy")
                except FileNotFoundError:
                    print(f"Rank {dist.get_rank()} cache file for {ntype} not found")
                    continue
                if dataset == 'donor':
                    cache = GPUCache(len(cache_nodes), g.number_of_nodes(ntype), feat_size, feat_dtype, dev_id)
                else:
                    cache = GPUCache(len(cache_nodes), part_policy.get_size(), feat_size, feat_dtype, dev_id)
                if dataset == 'mag240m' and label != 'dgl':
                    cache.init_cache(cache_nodes, g, ntype, init_data=self.feat[ntype])
                else:
                    cache.init_cache(cache_nodes, g, ntype)
                self._gpu_caches[ntype] = cache
            elif cache_method != 'none' and self.reduction_level!=None:
                part_policy = None
                if partition_book is not None:
                    part_policy = NodePartitionPolicy(partition_book, ntype=ntype)
                else:
                    part_policy = g.get_node_partition_policy(ntype)
                part_size = part_policy.get_part_size()
                print(f"Rank {dist.get_rank()} part_size for {ntype}: {part_size}")
                feat_dtype = torch.float16 if dataset == 'mag240m' and label != 'dgl' else torch.float32
                try:
                    cache_nodes = np.load(f"cache/drgnn/{dataset}_{cache_method}_{args.model}_{args.num_gpus}/{ntype}.npy")
                except FileNotFoundError:
                    print(f"Rank {dist.get_rank()} cache file for {ntype} not found")
                    continue
                cache_dict={}
                # split cache_nodes into cache_nodes[dim]
                masks=self.feat[ntype].get_cache_mask(cache_nodes)
                idx = {dim: cache_nodes[masks[dim]] for dim in self.reduction_level}
                for dim in self.reduction_level:
                    feat_shape=self.feat[ntype].shape(dim)
                    if feat_shape[0]!=0:
                        cache=GPUCache(len(idx[dim]), feat_shape[0], feat_shape[1], feat_dtype, dev_id)
                        cache.init_cache(idx[dim], g, ntype, init_data=self.feat[ntype])
                        cache_dict[dim]=cache
                self._gpu_caches[ntype] = cache_dict

        if dgl_sparse_emb:
            self.node_embeds = {}
            for ntype in sorted(ntypes_wo_feat):
                # We only create embeddings for nodes without node features.
                part_policy = None
                if partition_book is not None:
                    part_policy = NodePartitionPolicy(partition_book, ntype=ntype)
                else:
                    part_policy = g.get_node_partition_policy(ntype) 
                part_size = part_policy.get_part_size()
                print(f"Rank {dist.get_rank()} part_size for {ntype}: {part_size}")

                gpu_cache = None
                if cache_method != 'none' and part_size > 0 and label != 'dgl':
                    cache_nodes = np.load(f"cache/drgnn/{dataset}_{cache_method}_{args.model}_{args.num_gpus}/{ntype}.npy")
                    # cache_nodes = np.load(f"cache/{label}/{dataset}_{cache_method}/part{self._machine_id}/{ntype}.npy")
                    # 非复制cache，写直达
                    # local_cache_nodes = cache_nodes[cache_nodes % self._local_world_size == self._gpu_id]
                    # local_cache_nodes = cache_nodes
                    # 复制cache，不写直达
                    local_cache_nodes = cache_nodes
                    # local_cache_nodes = cache_nodes[:int(len(cache_nodes)/self._local_world_size)]
                    gpu_cache = GPUCache(len(local_cache_nodes), part_policy.get_size(), embed_size, torch.float32, dev_id, write_through=False)
                    gpu_cache.init_cache(local_cache_nodes, g, ntype, init_func=init_emb)
                    print(f"Rank {dist.get_rank()} init cache for {ntype}")

                emb = DistEmbedding(
                    part_policy.get_size(),
                    self.embed_size,
                    embed_name + "_" + ntype,
                    init_emb,
                    part_policy,
                    gpu_cache=gpu_cache
                )
                self.node_embeds[ntype] = emb
                print(f"Rank {dist.get_rank()} create DistEmbedding for {ntype}")
                if gpu_cache is not None:
                    self._gpu_caches[emb.name] = gpu_cache
                    # gpu_cache._tensor=emb._tensor
                    # if torch.distributed.get_rank()==0:
                    #     print("before",torch.equal(self._gpu_caches[emb.name].cache_buffer,gpu_cache.cache_buffer),
                    #           gpu_cache.cache_buffer, emb._tensor._get(local_cache_nodes))
                    cache_idx = torch.arange(len(local_cache_nodes), dtype=torch.long, device=dev_id)
                    gpu_cache.cache_buffer[cache_idx]=emb._tensor._get(local_cache_nodes).to(dev_id)
                    # if torch.distributed.get_rank()==0:
                    #     print(torch.distributed.get_rank(),ntype,"after",torch.equal(self._gpu_caches[emb.name].cache_buffer,gpu_cache.cache_buffer),
                    #             gpu_cache.cache_buffer, emb._tensor._get(local_cache_nodes))
            # if replicate cache, broadcast
            # self.broadcast()  # now dont broadcast, directly read from DistEmbedding.
            # self.init_cache_again()
        else:
            self.node_embeds = nn.ModuleDict()
            for ntype in sorted(ntypes_wo_feat):
                part_policy = None
                if partition_book is not None:
                    part_policy = NodePartitionPolicy(partition_book, ntype=ntype)
                else:
                    part_policy = g.get_node_partition_policy(ntype) 
                part_size = part_policy.get_part_size()

                print(f"Rank {dist.get_rank()} part_size for {ntype}: {part_size}")
                self.node_embeds[ntype] = nn.Embedding(
                    part_policy.get_size(), embed_size, sparse=True
                )
                nn.init.uniform_(self.node_embeds[ntype].weight, -1.0, 1.0)

    # def init_cache_again(self):
    #     for ntype in sorted(self.node_embeds.keys()):
    #         if self.node_embeds[ntype].cache != None:
    #             print(f"init_cache_again node_embeds for {ntype}")
    #             if not dist.is_initialized():
    #                 # no need to broadcast if not using distributed training
    #                 return 
    #             cache = self.node_embeds[ntype].cache.cache_buffer
    #             print(f"rank {dist.get_rank()}, cache id=",id(cache))
    #             if isinstance(cache, torch.Tensor):
    #                 cache.buffer=self.node_embeds[ntype]._tensor._get
    #             else:
    #                 print(f"cache {cache} is no tensor!")

    # all_reduce cache counter;
    # return {ntype, cache_counter}
    # return None if cache_counter==None
    # def all_reduce_counter(self):
    #     cache_counter={}
    #     for ntype, cache in self._gpu_caches.items():
    #         cache_counter[ntype]=cache.all_reduce_counter()
    #     return cache_counter

    def get_feat_shape_dict(self):
        feat_shape_dict={}
        # for ntypes_w_feat
        if self.reduction_level!=None:
            for ntype in self.feat:
                for dim in self.reduction_level:
                    feat_shape=self.feat[ntype].shape(dim)
                    if feat_shape[0]!=0:
                        feat_shape_dict[ntype]=feat_shape[1]
                        continue    # in this case, the features of each type of node may have only one dimension.
        else:
            if self.dataset == 'mag240m':
                feat_shape_dict={ntype:768 for ntype in self.ntypes_w_feat}
            else:
                feat_shape_dict={ntype:self.g.nodes[ntype].data[self.feat_name].shape[1] for ntype in self.ntypes_w_feat}
        # for ntypes_wo_feat
        for ntype in self.node_embeds:
            feat_shape_dict[ntype]=self.embed_size
        print("feat_shape_dict",feat_shape_dict)
        return feat_shape_dict

    def update_cache(self):
        update_num=0
        for _, cache in self._gpu_caches.items():
            load_count=cache.update(self._local_world_size, self._gpu_id)
            if load_count!=0:
                update_num+=load_count
        return update_num

    def broadcast(self):
        for ntype in sorted(self.node_embeds.keys()):
            if self.node_embeds[ntype].cache != None:
                print(f"broadcast node_embeds for {ntype}")
                _broadcast_cache([self.node_embeds[ntype].cache.cache_buffer], src=0)
                # dist.all_reduce(self.node_embeds[ntype].cache.cache_buffer, op=dist.ReduceOp.SUM)
                # self.node_embeds[ntype].cache.cache_buffer /= float(dist.get_world_size())

    @property
    def cache_read_hit_rate(self):
        return {
            ntype: f"{np.mean(self._cache_read_hit_rate[ntype])*100:.2f}" for ntype in self._cache_read_hit_rate if len(self._cache_read_hit_rate[ntype]) > 0 
        }
    
    # @property
    # def cache_write_hit_rate(self):
    #     return {
    #         ntype: f"{np.mean(self._cache_write_hit_rate[ntype])*100:.2f}" for ntype in self._cache_write_hit_rate if len(self._cache_write_hit_rate[ntype]) > 0 
    #     }
    
    @property
    def gpu_caches(self):
        return self._gpu_caches

    def _fetch_feat(self, ntype, node_ids):
        """Fetch features
        """
        if ntype in self.feat:
            if self.reduction_level==None:
                return torch.from_numpy(self.feat[ntype][node_ids])
            else:
                feats, masks=self.feat[ntype].get(node_ids)
                # feats = {dim:feat.type(torch.float32).to(self.dev_id,non_blocking=True) for dim,feat in feats.items()}
                feats = {dim:feat.to(self.dev_id,non_blocking=True) for dim,feat in feats.items()}
                masks = {dim:mask.to(self.dev_id,non_blocking=True) for dim,mask in masks.items()}
                return feats,masks
        else:
            return self.g.nodes[ntype].data[self.feat_name][node_ids]

    @property
    def num_gen(self):
        if len(self._gen_feat_count)==2:
            return self._gen_feat_count
        else:
            return [0,0]

    def forward(self, node_ids):
        """Forward computation
        Parameters
        ----------
        node_ids : dict of Tensor
            node ids to generate embedding for.
        Returns
        -------
        tensor
            embeddings as the input of the next layer
        """
        embeds = {}
        self._fetch_feat_time = 0
        self._proj_time = 0

       
        # if dist.get_rank()==0:
        #     print("self.node_embeds['author'](dgl.utils.Index([0]))=", self.node_embeds['author'](dgl.utils.Index([0])).to(torch.device('cpu')))
        #     print("self.g.nodes['author'].data['feat'][dgl.utils.Index([0])]", self.g.nodes['author'].data['feat'][dgl.utils.Index([0])].to(torch.device('cpu')))

        # if torch.distributed.get_rank()==1:
        #     print("in!:")
        #     for ntype in node_ids:
        #         print(ntype,node_ids[ntype].device)
        # if torch.distributed.get_rank()==1:
        #     print("out!:")
        #     print(f"rank {torch.distributed.get_rank()} {'paper'} {id(node_ids['paper'])} {node_ids['paper'].device} {node_ids['paper']}")
        for ntype in node_ids:
            start = time.time()
            if ntype in self.ntypes_w_feat:
                if ntype in self._gpu_caches:
                    if self.reduction_level ==None:
                        idx = node_ids[ntype].to(self.dev_id)
                        gpu_cache = self._gpu_caches[ntype]
                        cached_feat, cache_mask = gpu_cache.get(idx)
                        self._cache_read_hit_rate[ntype].append(gpu_cache.cache_read_hit_rate)
                        # self._cache_write_hit_rate[ntype].append(gpu_cache.cache_write_hit_rate)
                        cached_feat = cached_feat.type(torch.float32)
                        uncached_mask = ~cache_mask
                        uncached_idx = idx[uncached_mask].to('cpu')

                        uncached_values = (self._fetch_feat(ntype, uncached_idx)
                                        .type(torch.float32)
                                        .to(self.dev_id))
                        feat = torch.empty((idx.shape[0], gpu_cache.dim), dtype=torch.float32, device=self.dev_id)
                        feat[cache_mask] = cached_feat
                        feat[uncached_mask] = uncached_values
                    else:
                        # get mask[dim] for node_ids[ntype], for return also
                        masks=self.feat[ntype].get_cache_mask(node_ids[ntype].to('cpu'))
                        # split node_ids[ntype] into node_ids[ntype][dim]
                        idx = {dim: node_ids[ntype][masks[dim]].to(self.dev_id) for dim in self.reduction_level}
                        # for each dim, use node_ids[ntype][dim] to get cached_feat, cache_mask
                        cached_feat={}
                        cache_mask={}
                        uncached_idx=[]
                        for dim in self.reduction_level:
                            if idx[dim].shape[0]==0:
                                # no nodes needed
                                continue
                            gpu_cache = self._gpu_caches[ntype][dim]
                            # if torch.distributed.get_rank()==1:
                            #     for dim in idx:
                            #         print(dim,idx[dim].device)
                            cached_feat[dim], cache_mask[dim] = gpu_cache.get(idx[dim])
                            self._cache_read_hit_rate[ntype].append(gpu_cache.cache_read_hit_rate)
                            # if gpu_cache.cache_write_hit_rate!=-1:
                            #     self._cache_write_hit_rate[ntype].append(gpu_cache.cache_write_hit_rate)
                            #     gpu_cache.cache_write_hit_rate=-1
                            cached_feat[dim] = cached_feat[dim].type(torch.float32)
                            uncached_mask = ~cache_mask[dim]
                            uncached_idx.append(idx[dim][uncached_mask])
                        # stack into uncached_idx, get uncached_feats,uncached_masks
                        # transfer data through pcie once, avoiding extra cost
                        uncached_idx=torch.stack(uncached_idx)
                        uncached_feats,_=self._fetch_feat(ntype, uncached_idx.cpu())
                        # combine uncached_feats with cached_feats
                        feats = {}
                        for dim in self.reduction_level:
                            feats[dim]=torch.empty((idx[dim].shape[0], dim), dtype=torch.float32, device=self.dev_id)
                            if idx[dim].shape[0]==0:
                                # no nodes needed
                                continue
                            feats[dim][cache_mask[dim]] = cached_feat[dim]
                            feats[dim][~cache_mask[dim]] = uncached_feats[dim]
                else:
                    if self.reduction_level ==None:
                        feat = (self._fetch_feat(ntype, node_ids[ntype].cpu())
                                .type(torch.float32)
                                .to(self.dev_id))
                    else:
                        feats,masks=self._fetch_feat(ntype, node_ids[ntype].cpu())
                        
                self._fetch_feat_time += time.time() - start

                start = time.time()
                if self.reduction_level ==None:
                    if self.use_node_projs:
                        embeds[ntype] = self.node_projs[ntype](feat)
                    else:
                        embeds[ntype]=feat
                else:
                    num_nodes=node_ids[ntype].shape[0]
                    if self.use_node_projs:  # for this case, each node type may have different dimensions.
                        embeds[ntype]=torch.empty((num_nodes, self.embed_size), dtype=torch.float32, device=self.dev_id)
                        for dim in self.reduction_level:                        
                            if feats[dim].shape[0]==0: 
                                continue
                            embeds[ntype][masks[dim]]=self.node_projs[f'{ntype}_{dim}'](feats[dim])
                    else:   # for this case, each node type can only have one unique dimensions.
                        for dim in self.reduction_level:
                            if feats[dim].shape[0]==0: 
                                continue
                            embeds[ntype]=feats[dim]
                self._proj_time += time.time() - start
            elif self.dgl_sparse_emb:
                embeds[ntype] = self.node_embeds[ntype](node_ids[ntype], self.dev_id)
                if self.node_embeds[ntype].gpu_cache is not None:
                    gpu_cache=self.node_embeds[ntype].gpu_cache
                    self._cache_read_hit_rate[ntype].append(gpu_cache.cache_read_hit_rate)
                    # cache_write_hit_rate = self.node_embeds[ntype].gpu_cache.cache_write_hit_rate
                    # if cache_write_hit_rate!=-1:
                    #     self._cache_write_hit_rate[ntype].append(cache_write_hit_rate)
                    #     self.node_embeds[ntype].gpu_cache.cache_write_hit_rate=-1
                self._fetch_feat_time += time.time() - start
            else:
                embeds[ntype] = self.node_embeds[ntype](node_ids[ntype]).to(self.dev_id)
        
        # cul gen_feat_count
        self._gen_feat_count=[0,0]
        for ntype in node_ids:
            if ntype in self.ntypes_w_feat:
                self._gen_feat_count[0]+=embeds[ntype].shape[0]
            else:
                self._gen_feat_count[1]+=embeds[ntype].shape[0]
        # print(self._gen_feat_count)
        return embeds

class RGCN(nn.Module):
    """use dglnn.HeteroGraphConv"""
    def __init__(self, g: dgl.DGLGraph, predict_category: str, in_size: int, hid_size: int,
                 out_size: int, n_layers: int = 3, dropout: float = 0.5, feat_shape_dict=None):
        super().__init__()
        self.predict_category = predict_category
        self.hid_size = hid_size
        self.out_size = out_size
        self.layers = nn.ModuleList()
        etypes=g.etypes
        canonical_etypes=g.canonical_etypes
        print(f"created RGCN with etypes: {etypes}, some will be set requires_grad=False")
        for layer in range(n_layers):
            if layer == n_layers - 1:
                layer_out_size = out_size
            else:
                layer_out_size = hid_size
            if layer == 0:
                layer_in_size = in_size
                if feat_shape_dict==None:
                    self.layers.append(HeteroGraphConv({
                        etype: GraphConv(in_size, layer_out_size, norm='right') for etype in etypes
                    }, aggregate='sum'))
                else:
                    self.layers.append(HeteroGraphConv({
                        canonical_etype[1]: GraphConv(feat_shape_dict[canonical_etype[0]], layer_out_size, norm='right') for canonical_etype in canonical_etypes
                    }, aggregate='sum'))
            else:
                layer_in_size = hid_size
                self.layers.append(HeteroGraphConv({
                    etype: GraphConv(hid_size, layer_out_size, norm='right') for etype in etypes
                }, aggregate='sum'))
            
        # set unused sub-layers requires_grad=False
        rev_layers=_meta_bfs(n_layers, g.canonical_etypes, predict_category)
        # print("rev_layers",rev_layers)
        for i, layer in enumerate(self.layers):
            used_layers=[canonical_etype[1] for canonical_etype in rev_layers[-(i+1)]]
            for key in layer.mods.keys():
                if key not in used_layers:
                    if torch.distributed.get_rank()==0:
                        print(f"{i}'s layer {key} requires_grad=False")
                    for param in layer.mods[key].parameters():
                        param.requires_grad=False
        self.self_loop_layer = nn.Linear(layer_in_size, layer_out_size)    # this wont work when num_layer==1
        self.dropout = nn.Dropout(dropout)
    
    def broadcast(self):
        _broadcast_layers([self.self_loop_layer], src=0)
    
    def forward(self, blocks: List[dgl.DGLHeteroGraph], inputs: Dict[str, torch.Tensor]):
        h = inputs
        num_seeds = blocks[-1].num_dst_nodes(self.predict_category)
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            input_dst = {
                k: v[:block.number_of_dst_nodes(k)] for k, v in h.items()
            }
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = apply_each(h, F.relu)
                h = apply_each(h, self.dropout)
        input_dst = input_dst[self.predict_category][:num_seeds]
        h[self.predict_category]=torch.add(h[self.predict_category], self.self_loop_layer(input_dst))
        return h[self.predict_category]


class RGAT(nn.Module):
    """use dglnn.HeteroGraphConv"""

    def __init__(self, g: dgl.DGLGraph, predict_category: str, in_size: int, hid_size: int,
                 out_size: int, n_layers: int = 3, n_heads: int = 4, dropout: float = 0.5, feat_shape_dict=None):
        super().__init__()
        self.predict_category = predict_category
        self.layers = nn.ModuleList()
        etypes=g.etypes
        canonical_etypes=g.canonical_etypes
        print(f"created RGAT with etypes: {etypes}, some will be set requires_grad=False")
        for layer in range(n_layers):
            if layer == 0:
                if feat_shape_dict==None:
                    self.layers.append(HeteroGraphConv({
                        etype: GATConv(in_size, hid_size // n_heads, n_heads) for etype in etypes
                    }, aggregate='sum'))
                else:
                    self.layers.append(HeteroGraphConv({
                        canonical_etype[1]: GATConv(feat_shape_dict[canonical_etype[0]], hid_size // n_heads, n_heads) for canonical_etype in canonical_etypes
                    }, aggregate='sum'))
            else:
                self.layers.append(HeteroGraphConv({
                    etype: GATConv(hid_size, hid_size // n_heads, n_heads) for etype in etypes
                }, aggregate='sum'))

        # set unused sub-layers requires_grad=False
        rev_layers=_meta_bfs(n_layers, g.canonical_etypes, predict_category)
        # print("rev_layers",rev_layers)
        for i, layer in enumerate(self.layers):
            used_layers=[canonical_etype[1] for canonical_etype in rev_layers[-(i+1)]]
            for key in layer.mods.keys():
                if key not in used_layers:
                    if torch.distributed.get_rank()==0:
                        print(f"{i}'s layer {key} requires_grad=False")
                    for param in layer.mods[key].parameters():
                        param.requires_grad=False
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hid_size, out_size)
    
    def broadcast(self):
        _broadcast_layers([self.fc], src=0)
    
    def forward(self, blocks: List[dgl.DGLHeteroGraph], inputs: Dict[str, torch.Tensor]):
        h = inputs
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            h = apply_each(
                h, lambda x: x.view(x.shape[0], x.shape[1] * x.shape[2])
            )

            if l != len(self.layers) - 1:
                h = apply_each(h, F.relu)
                h = apply_each(h, self.dropout)

        return self.fc(h[self.predict_category])

class HGTLayer(nn.Module):
    def __init__(self, in_dim, out_dim, node_dict, edge_dict, n_heads, dropout = 0.2, use_norm = False, 
                 predict_category = None, dist: bool =False, process_group=None):
        super(HGTLayer, self).__init__()

        self.in_dim        = in_dim
        self.out_dim       = out_dim
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.num_types = len(node_dict)
        self.num_relations = len(edge_dict)
        self.n_heads       = n_heads
        self.d_k           = out_dim // n_heads
        self.sqrt_dk       = math.sqrt(self.d_k)
        
        self.k_linears   = nn.ModuleList()
        self.q_linears   = nn.ModuleList()
        self.v_linears   = nn.ModuleList()
        self.a_linears   = nn.ModuleList()
        self.norms       = nn.ModuleList()
        self.use_norm    = use_norm
        
        for t in range(self.num_types):
            self.k_linears.append(nn.Linear(in_dim,   out_dim))
            self.q_linears.append(nn.Linear(in_dim,   out_dim))
            self.v_linears.append(nn.Linear(in_dim,   out_dim))
            self.a_linears.append(nn.Linear(out_dim,  out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))
            
        self.relation_pri   = nn.Parameter(torch.ones(self.num_relations, self.n_heads))
        self.relation_att   = nn.Parameter(torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))
        self.relation_msg   = nn.Parameter(torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))
        self.skip           = nn.Parameter(torch.ones(self.num_types))
        self.drop           = nn.Dropout(dropout)
        
        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

        self.predict_category = predict_category
        self.dist = dist
        self.process_group = process_group


    def forward(self, G: dgl.DGLGraph, h: Dict[str, torch.Tensor]):
        with G.local_scope():
            node_dict, edge_dict = self.node_dict, self.edge_dict
            for srctype, etype, dsttype in G.canonical_etypes:
                # skip empty 
                sub_graph = G[srctype, etype, dsttype]

                k_linear = self.k_linears[node_dict[srctype]]
                v_linear = self.v_linears[node_dict[srctype]] 
                q_linear = self.q_linears[node_dict[dsttype]]

                k = k_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
                v = v_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
                q = q_linear(h[dsttype][:G.num_dst_nodes(dsttype)]).view(-1, self.n_heads, self.d_k)

                e_id = self.edge_dict[etype]
                relation_att = self.relation_att[e_id]
                relation_pri = self.relation_pri[e_id]
                relation_msg = self.relation_msg[e_id]

                k = torch.einsum("bij,ijk->bik", k, relation_att)
                v = torch.einsum("bij,ijk->bik", v, relation_msg)

                sub_graph.srcdata["k"] = k
                sub_graph.dstdata["q"] = q
                sub_graph.srcdata["v_%d" % e_id] = v

                sub_graph.apply_edges(fn.v_dot_u("q", "k", "t"))
                attn_score = (
                    sub_graph.edata.pop("t").sum(-1)
                    * relation_pri
                    / self.sqrt_dk
                )
                attn_score = edge_softmax(sub_graph, attn_score, norm_by="dst")

                sub_graph.edata["t"] = attn_score.unsqueeze(-1)

            G.multi_update_all(
                {
                    etype : (
                            fn.u_mul_e("v_%d" % self.edge_dict[etype], "t", "m"),
                            fn.sum("m", "t"),
                    )
                    for etype in G.etypes
                }, 
                cross_reducer = 'mean'
            )


            new_h = {}
            for ntype in G.ntypes:
                if isinstance(G.dstdata['t'], dict) and ntype not in G.dstdata['t']:
                    new_h[ntype] = h[ntype][:G.num_dst_nodes(ntype)]
                    continue
                n_id = node_dict[ntype]
                alpha = torch.sigmoid(self.skip[n_id])
                if isinstance(G.dstdata['t'], dict):
                    t = G.dstdata['t'][ntype].view(-1, self.out_dim)
                else:
                    t = G.dstdata['t'].view(-1, self.out_dim)

                trans_out = self.drop(self.a_linears[n_id](t))
                trans_out = trans_out * alpha + h[ntype][:G.num_dst_nodes(ntype)] * (1-alpha)

                # if self.dist and ntype == self.predict_category:
                #     dist.all_reduce(trans_out, op=dist.ReduceOp.SUM, group=self.process_group)
                #     trans_out /= dist.get_world_size(group=self.process_group)

                if self.use_norm:
                    new_h[ntype] = self.norms[n_id](trans_out)
                else:
                    new_h[ntype] = trans_out
            return new_h

    def __repr__(self):
        return '{}(in_dim={}, out_dim={}, num_types={}, num_types={})'.format(
            self.__class__.__name__, self.in_dim, self.out_dim,
            self.num_types, self.num_relations)



class HGT(nn.Module):
    def __init__(
        self,
        node_dict,
        edge_dict,
        predict_category,
        in_feats,
        num_hidden,
        n_classes,
        n_layers=3,
        n_heads=4,
        use_norm=True,
        dist: bool = False,
        process_group=None,
    ):
        super(HGT, self).__init__()
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.predict_category = predict_category
        self.gcs = nn.ModuleList()
        self.n_inp = in_feats
        self.n_hid = num_hidden
        self.n_out = n_classes
        self.n_layers = n_layers
        self.adapt_ws = nn.ModuleList()
        self.use_norm = use_norm

        # for _ in range(len(node_dict)):
        #     self.adapt_ws.append(nn.Linear(in_feats,num_hidden))
        for i in range(n_layers):
            if i == n_layers - 1:
                self.gcs.append(
                    HGTLayer(
                        num_hidden,
                        num_hidden,
                        node_dict,
                        edge_dict,
                        n_heads,
                        use_norm=use_norm,
                        predict_category=predict_category,
                        dist=dist,
                        process_group=process_group
                    )
                )
            else:
                self.gcs.append(
                    HGTLayer(
                        num_hidden,
                        num_hidden,
                        node_dict,
                        edge_dict,
                        n_heads,
                        use_norm=use_norm,
                    )
                )

        self.fc = nn.Linear(num_hidden, n_classes)
        self.dist = dist
        self.process_group = process_group

    def broadcast(self):
        predict_category_id = self.node_dict[self.predict_category]
        layers_to_broadcast = [self.fc, self.gcs[-1].skip, self.gcs[-1].a_linears[predict_category_id]]
        if self.use_norm:
            layers_to_broadcast.append(self.gcs[-1].norms[predict_category_id])
        _broadcast_layers(layers_to_broadcast, src=0)

    def forward(self, blocks, inputs):
        h = inputs
        # for ntype,n_id in self.node_dict.items():
        #     h[ntype] = F.gelu(self.adapt_ws[n_id](h[ntype]))
        for l, (layer, block) in enumerate(zip(self.gcs, blocks)):
            h = layer(block, h)
         
        return self.fc(h[self.predict_category])


def get_model(model_name: str, g: dgl.DGLGraph, predict_category: str, 
              in_feats: int, num_hidden: int,  n_classes: int,
              num_layers: int, dropout: float = 0.5, feat_shape_dict=None):
    # 现在是0，之后改完bug设回0.5!
    """get model 
    """
    if model_name == "rgcn":
        model = RGCN(
            g, 
            predict_category,
            in_feats,
            num_hidden,
            n_classes,
            num_layers,
            dropout=dropout,
            feat_shape_dict=feat_shape_dict
        )
    elif model_name == "rgat":
        model = RGAT(
            g,
            predict_category,
            in_feats, 
            num_hidden, 
            n_classes,
            num_layers,
            dropout=dropout,
            feat_shape_dict=feat_shape_dict
        )
    elif model_name == "hgt":
        node_dict = {ntype: i for i, ntype in enumerate(g.ntypes)}
        edge_dict = {etype: i for i, etype in enumerate(g.etypes)}
        model = HGT(
            node_dict,
            edge_dict,
            predict_category,
            in_feats,
            num_hidden,
            n_classes,
            num_layers,
            use_norm=True,
            dist=False,
            process_group=None
        )
    else:
        raise ValueError(f"model_name {model_name} not supported")
    
    return model
