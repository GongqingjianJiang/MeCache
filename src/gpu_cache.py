"""gpu cache"""
import os
from typing import Optional, Tuple
import numpy as np

import torch
import torch.distributed as dist
import torch.distributed
from .reduction import Feat

def itemsize(dtype):
    """get item size of dtype"""
    if dtype == torch.float16:
        return 2
    elif dtype == torch.float32:
        return 4
    elif dtype == torch.float64:
        return 8
    elif dtype == torch.int8:
        return 1
    elif dtype == torch.int16:
        return 2
    elif dtype == torch.int32:
        return 4
    elif dtype == torch.int64:
        return 8
    elif dtype == torch.bool:
        return 1
    elif dtype == torch.uint8:
        return 1
    else:
        raise ValueError("unknown dtype {}".format(dtype))

class GPUCache:
    """gpu cache"""

    def __init__(self, capacity: float, num_nodes: int, dim: int, dtype, device, write_through: bool = True):
        self.num_nodes = num_nodes
        self.capacity = capacity
        self.device = device
        self.dim = dim
        self.dtype = dtype
        self.write_through = write_through

        print(f"cache total: {num_nodes}, cache capacity: {self.capacity}, cache memory size: {self.capacity * dim * itemsize(dtype) / 1024 / 1024:.2f} MB")

        self.cache_buffer = torch.zeros(self.capacity, dim, dtype=dtype, device=device)
        # flag for indicating those cached nodes
        self.cache_flag = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        # maps node id -> index
        self.cache_map = torch.zeros(num_nodes, dtype=torch.long, device=device) - 1

        self.cache_read_hit_rate = -1
        self.cache_write_hit_rate = -1

        # # tensor in emb, used for update
        # self._tensor=None
        # # counter for dynamic cache
        # if dynamic:
        #     self.cache_counter = torch.zeros(self.num_nodes, dtype=torch.int32, device=self.device)
        # else:
        #     self.cache_counter = None
    
    def init_cache(self, node_idx, g, ntype, init_func=None, init_data=None):
        """init highest degree nodes as cached nodes"""
        self.cache_flag[node_idx] = True
        cache_idx = torch.arange(len(node_idx), dtype=torch.long, device=self.device)
        self.cache_map[node_idx] = cache_idx
        if init_data is not None:
            # for level reduction
            if isinstance(init_data, Feat):
                print(f"init cache with data of shape {self.num_nodes}, len(node_idx) = {len(node_idx)})")
                feats,_=init_data.get(node_idx)
                self.cache_buffer[cache_idx] = feats[self.dim].to(self.device, non_blocking=True)
            else:
                print(f"init cache with data of shape {init_data.shape}, len(node_idx) = {len(node_idx)})")
                if isinstance(init_data, np.ndarray):
                    feat = torch.from_numpy(init_data[node_idx])
                elif isinstance(init_data, torch.Tensor):
                    feat = init_data[node_idx]
                else:
                    raise TypeError(f'init_data type must be {np.ndarray} or {torch.Tensor}; instead, got{type(init_data)}')
                print("feat shape", feat.shape)
                self.cache_buffer[cache_idx] = feat.to(self.device, non_blocking=True)
        elif init_func is None:
            feat = g.nodes[ntype].data['feat'][node_idx]
            self.cache_buffer[cache_idx] = feat.to(self.device, non_blocking=True)
        elif init_func is not None:
            self.cache_buffer[cache_idx] = (init_func((self.capacity, self.cache_buffer.shape[1]), dtype=self.cache_buffer.dtype)
                                            .to(self.device, non_blocking=True))
        else:
            raise ValueError("data and init_func cannot be both None")
    
    def clone(self, write_through: bool = False):
        """clone a new cache"""
        new_cache = GPUCache(self.capacity, self.num_nodes, self.cache_buffer.shape[1], self.cache_buffer.dtype, self.device, write_through)
        new_cache.cache_buffer = self.cache_buffer.clone().zero_()
        new_cache.cache_flag = self.cache_flag.clone()
        new_cache.cache_map = self.cache_map.clone()
        # new_cache.cache_counter=None
        return new_cache

    def get(self, idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """get cached value by idx

        Args:
            idx (torch.Tensor): indices of cached value

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: cached value, cached mask
        """
        with torch.no_grad():
            # if self.cache_counter!=None:
            #     self.cache_counter[idx]+=1
            cache_mask = self.cache_flag[idx]
            self.cache_read_hit_rate=cache_mask.sum().item()/idx.shape[0] if idx.shape[0]!=0 else 0
            cache_node_idx = idx[cache_mask]
            cache_idx = self.cache_map[cache_node_idx]
            return self.cache_buffer[cache_idx], cache_mask

    def set(self, idx: torch.Tensor, val: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """set cached value by idx

        Args:
            idx (torch.Tensor): indices of cached value
            val (torch.Tensor): cached value
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: cached indices, and cache mask
        """
        with torch.no_grad():
            # self.cache_counter[idx]+=1
            cache_mask = self.cache_flag[idx]
            self.cache_write_hit_rate = cache_mask.sum().item()/idx.shape[0] if idx.shape[0]!=0 else 0
            cache_node_idx = idx[cache_mask]
            cache_idx = self.cache_map[cache_node_idx]
            self.cache_buffer[cache_idx] = val[cache_mask]
            # if self.cache_write_hit_rate<1:
            #     print(self.device,self.gate(idx))
            return cache_node_idx, cache_mask

    def _print(self):
        print(f"self.cache_buffer = {self.cache_buffer}\nself.cache_flag = {self.cache_flag}\nself.cache_map = {self.cache_map}")
    
    # @property
    # def gate(self,idx):
    #     cache_mask = self.cache_flag[idx]
    #     return idx[cache_mask],idx[~cache_mask]

    # @property
    # def cache_hit_rate(self):
    #     return self.cache_read_hit_rate
    
    # def all_reduce_counter(self):
    #     if dist.get_backend()!='nccl':
    #         raise NotImplementedError("dynamic cache do not support backend other than nccl! If u r using gloo, use static cache.")
    #     if self.cache_counter != None:
    #         dist.all_reduce(self.cache_counter, op=dist.ReduceOp.SUM)
    #     return self.cache_counter
    
    # # update 
    # # retuern load and evict count
    # def update(self, local_world_size, gpu_id, counter=None):
    #     with torch.no_grad():
    #         if counter==None and self.cache_counter!=None:
    #             counter=self.cache_counter
    #         elif counter==None and self.cache_counter==None:
    #             return -1
    #         # nodes with highest access rate, top self.capacity
    #         final_idx=torch.argsort(counter,descending=True)[:self.capacity]
    #         final_mask= torch.zeros(self.num_nodes, dtype=torch.bool, device=self.device)
    #         final_mask[final_idx]=True
    #         # nodes that need to be loaded or evicted
    #         load_evict_mask=torch.logical_xor(final_mask,self.cache_flag)
    #         # nodes that need to be loaded
    #         load_mask=load_evict_mask & final_mask
    #         load_idx=torch.nonzero(load_mask).squeeze().to('cpu',non_blocking=True)
    #         # nodes that need to be evicted
    #         evict_mask=load_evict_mask & self.cache_flag
    #         evict_idx=torch.nonzero(evict_mask).squeeze()
    #         # pull
    #         new_cacheline=self._tensor._get(load_idx).to(self.device,non_blocking=True)
    #         # new_cacheline=self._tensor[load_idx].to(self.device)
    #         # push 1/local_world_size, avoiding write conflict
    #         evict_idx_part = evict_idx[evict_idx % local_world_size == gpu_id].to('cpu',non_blocking=True)
    #         cache_evict_idx_part=self.cache_map[evict_idx_part]
    #         self._tensor._set(evict_idx_part,self.cache_buffer[cache_evict_idx_part].to('cpu',non_blocking=True))
    #         # update self.cache_buffer
    #         write_idx=self.cache_map[evict_idx]
    #         self.cache_buffer[write_idx]=new_cacheline
    #         # update self.cache_flag
    #         self.cache_flag=final_mask
    #         # update self.cache_map
    #         self.cache_map[evict_idx]=-1
    #         self.cache_map[load_idx]=write_idx
    #         # update self.cache_counter
    #         if self.cache_counter!=None:
    #             self.cache_counter.zero_()
    #         # return num_load
    #         return len(load_idx)
        
if __name__ == '__main__':
    # init cache
    device=torch.device('cuda:0')
    dim=2
    total_num=20
    capacity=10
    origin_id=torch.arange(start=10,end=20,dtype=torch.long,device=device)
    val=torch.stack([torch.full(size=(dim,),fill_value=int(i),dtype=torch.float32,device=device) for i in range(total_num)])
    cache = GPUCache(capacity,total_num,dim,torch.float32,device,False)
    cache.init_cache(origin_id,None,torch.float32,init_data=val)
    cache._print()
    cache._tensor=val

    # try update
    new_id=torch.arange(start=5,end=15,dtype=torch.long,device=device)
    counter=torch.zeros((total_num,),dtype=torch.long,device=device)
    counter[new_id]+=2
    print('counter=',counter)
    cache.update(4,0,counter)

    # test update result
    cache._print()
    print("cache.get(origin_id)",cache.get(origin_id))
    print("cache.get(new_id)",cache.get(new_id))

    b=cache.clone()
    b.update(4,0,counter)
    