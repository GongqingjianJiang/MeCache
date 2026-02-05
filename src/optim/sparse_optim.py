"""Node embedding optimizers for distributed training"""
import abc
import os
import time
import warnings
from abc import abstractmethod
from os.path import exists
from typing import Optional
from collections import defaultdict
import torch as th
import numpy as np

import dgl

from dgl import backend as F
from ..distributed.dist_tensor import DistTensor
from ..distributed.sparse_emb import DistEmbedding
from dgl.distributed.graph_partition_book import EDGE_PART_POLICY, NODE_PART_POLICY
from .utils import alltoall_cpu, alltoallv_cpu

EMB_STATES = "emb_states"
WORLD_SIZE = "world_size"
IDS = "ids"
PARAMS = "params"
STATES = "states"


class DistSparseGradOptimizer(abc.ABC):
    r"""The abstract dist sparse optimizer.

    Note: dgl dist sparse optimizer only work with dgl.distributed.DistEmbedding

    Parameters
    ----------
    params : list of DistEmbedding
        The list of DistEmbedding.
    lr : float
        The learning rate.
    """

    def __init__(self, params, lr):
        self._params = params
        self._lr = lr
        self._rank = None
        self._world_size = None
        self._shared_cache = {}
        self._clean_grad = False
        self._opt_meta = {}
        self._state = {}
        ## collect all hyper parameters for save
        self._defaults = {}

        self._send_grad_time = 0
        self._pull_time = 0
        self._push_time = 0
        self._h2d_d2h_time = 0
        self._comp_time = 0
        self._tot_time = 0
        self._step_times=0

        print("Init optimizer")

        if th.distributed.is_initialized():
            print("Init optimizer with dist")
            self._rank = th.distributed.get_rank()
            self._world_size = th.distributed.get_world_size()
            self._local_rank = int(os.environ["LOCAL_RANK"])
            self._local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
            print(f"Rank {self._rank} local rank {self._local_rank} world size {self._world_size} local world size {self._local_world_size}")
        else:
            self._rank = 0
            self._world_size = 1
            self._local_rank = 0
            self._local_world_size = 1
        
        print("Init optimizer done")

    def local_state_dict(self):
        """Return the state pertaining to current rank of the optimizer.

        Returns
        -------
        dict
            Local state dict
            Example Dict of Adagrad Optimizer:
            .. code-block:: json

            {
                "params": {
                    "_lr": 0.01,
                    "_eps": "1e-8",
                    "world_size": 2
                },
                "emb_states": {
                    "emb_name1": {
                        "ids": [0, 2, 4, 6 ,8 ,10], ## tensor,
                        "emb_name1_sum": [0.1 , 0.2, 0.5, 0.1, 0.2] ## tensor,
                    },
                    "emb_name2": {
                        "ids": [0, 2, 4, 6 ,8 ,10], ## tensor,
                        "emb_name2_sum": [0.3 , 0.2, 0.4, 0.5, 0.2] ## tensor,
                    }
                }
            }

            :param json: json object

        See Also
        --------
        load_local_state_dict
        """
        local_state_dict = {}
        local_state_dict[EMB_STATES] = {}
        local_state_dict[PARAMS] = {WORLD_SIZE: self._world_size}
        for emb in self._params:
            trainers_per_machine = self._world_size // max(
                1, dgl.distributed.get_num_machines()
            )
            emb_state_dict = {}
            part_policy = (
                emb.part_policy if emb.part_policy else emb.weight.part_policy
            )
            idx = self._get_local_ids(part_policy)
            if trainers_per_machine > 1:
                kv_idx_split = (idx % trainers_per_machine).long()
                local_rank = self._rank % trainers_per_machine
                mask = kv_idx_split == local_rank
                idx = F.boolean_mask(idx, mask)
            emb_state_dict.update({IDS: idx})
            emb_state = {}
            states = (
                list(self._state[emb.name])
                if isinstance(self._state[emb.name], tuple)
                else [self._state[emb.name]]
            )
            emb_state = {state.name: state[idx] for state in states}
            emb_state_dict.update({STATES: emb_state})
            local_state_dict[EMB_STATES].update({emb.name: emb_state_dict})
        local_state_dict[PARAMS].update(self._defaults)
        return local_state_dict

    def load_local_state_dict(self, local_state_dict):
        """Load the local state from the input state_dict,
        updating the optimizer as needed.

        Parameters
        ----------
        local_state_dict : dict
            Optimizer state; should be an object returned
            from a call to local_state_dict().

        See Also
        --------
        local_state_dict
        """
        for emb_name, emb_state in local_state_dict[EMB_STATES].items():
            idx = emb_state[IDS]
            # As state of an embedding of different optimizers can be a single
            # DistTensor(Adagrad) or a tuple(Adam) of that, converting it to list for
            # consistency. The list contains reference(s) to original DistTensor(s).
            states = (
                list(self._state[emb_name])
                if isinstance(self._state[emb_name], tuple)
                else [self._state[emb_name]]
            )
            if len(emb_state[STATES]) != len(states):
                raise ValueError(
                    f"loaded state dict has a different number of states"
                    f" of embedding {emb_name}"
                )
            name_to_index = {
                state.name: index for index, state in enumerate(states)
            }
            for name, state in emb_state[STATES].items():
                if name not in name_to_index:
                    raise ValueError(
                        "loaded state dict contains a state {name}"
                        "that can't be found in the optimizer states"
                    )
                state_idx = name_to_index[name]
                state = state.to(
                    th.device("cpu"), states[name_to_index[name]].dtype
                )
                states[state_idx][idx] = state
        self._defaults.update(local_state_dict[PARAMS])
        self.__dict__.update(local_state_dict[PARAMS])

    def save(self, f):
        """Save the local state_dict to disk on per rank.

        Saved dict contains 2 parts:

        * 'params': hyper parameters of the optimizer.
        * 'emb_states': partial optimizer states, each embedding contains 2 items:
            1. ```ids```: global id of the nodes/edges stored in this rank.
            2. ```states```: state data corrseponding to ```ids```.

        NOTE: This needs to be called on all ranks.

        Parameters
        ----------
        f : Union[str, os.PathLike]
            The path of the file to save to.

        See Also
        --------
        load
        """
        if self._world_size > 1:
            th.distributed.barrier()
        f = f if isinstance(f, str) else str(f, "UTF-8")
        f = f"{f}_{self._rank}"
        th.save(self.local_state_dict(), f)
        if self._world_size > 1:
            th.distributed.barrier()

    def load(self, f):
        """Load the local state of the optimizer from the file on per rank.

        NOTE: This needs to be called on all ranks.

        Parameters
        ----------
        f : Union[str, os.PathLike]
            The path of the file to load from.

        See Also
        --------
        save
        """
        if self._world_size > 1:
            th.distributed.barrier()
        f = f if isinstance(f, str) else str(f, "UTF-8")
        f_attach_rank = f"{f}_{self._rank}"
        # Don't throw error here to support device number scale-out
        # after reloading, but make sure your hyper parameter is same
        # as before because new added local optimizers will be filled
        # in nothing
        if not exists(f_attach_rank):
            warnings.warn(f"File {f_attach_rank} can't be found, load nothing.")
        else:
            old_world_size = self._load_state_from(f_attach_rank)
            # Device number scale-in
            if self._world_size < old_world_size:
                for rank in range(
                    self._rank + self._world_size,
                    old_world_size,
                    self._world_size,
                ):
                    self._load_state_from(f"{f}_{rank}")
        if self._world_size > 1:
            th.distributed.barrier()

    def _load_state_from(self, f):
        local_state_dict = th.load(f)
        world_size = local_state_dict[PARAMS].pop(WORLD_SIZE)
        self.load_local_state_dict(local_state_dict)
        return world_size

    def _get_local_ids(self, part_policy):
        if EDGE_PART_POLICY in part_policy.policy_str:
            return part_policy.partition_book.partid2eids(
                part_policy.part_id, part_policy.type_name
            )
        elif NODE_PART_POLICY in part_policy.policy_str:
            return part_policy._partition_book.partid2nids(
                part_policy.part_id, part_policy.type_name
            )
        else:
            raise RuntimeError(
                "Cannot support policy: %s " % part_policy.policy_str
            )

    def step(self, dist: bool = False, group: Optional[th.distributed.ProcessGroup] = None):
        """The step function.

        The step function is invoked at the end of every batch to push the gradients
        of the embeddings involved in a mini-batch to DGL's servers and update the embeddings.
        """
        self._send_grad_time = 0
        self._pull_time = 0
        self._push_time = 0
        self._h2d_d2h_time = 0
        self._comp_time = 0
        self._tot_time = 0
        start = time.time()
        with th.no_grad():
            local_indics = {emb.weight.name: [] for emb in self._params}
            local_grads = {emb.weight.name: [] for emb in self._params}
            device = th.device("cpu")
            device_dict = {}
            for emb in self._params:
                name = emb.weight.name
                kvstore = emb.weight.kvstore
                trainers_per_server = self._world_size // kvstore.num_servers

                idics = []
                grads = []
                for trace in emb._trace:
                    if trace[1].grad is not None:
                        idics.append(trace[0])
                        grads.append(trace[1].grad.data)
                    else:
                        # assert len(trace[0]) == 0
                        print(f"Rank {self._rank} {name} has no grad")
                        pass
                # If the sparse embedding is not used in the previous forward step
                # The idx and grad will be empty, initialize them as empty tensors to
                # avoid crashing the optimizer step logic.
                #
                # Note: we cannot skip the gradient exchange and update steps as other
                # working processes may send gradient update requests corresponding
                # to certain embedding to this process.
                idics = (
                    th.cat(idics, dim=0)
                    if len(idics) != 0
                    else th.zeros((0,), dtype=th.long, device=th.device(f"cuda:{th.distributed.get_rank()}"))
                )
                grads = (
                    th.cat(grads, dim=0)
                    if len(grads) != 0
                    else th.zeros(
                        (0, emb.embedding_dim),
                        dtype=th.float32,
                        device=th.device(f"cuda:{th.distributed.get_rank()}"),
                    )
                )
                device = grads.device
                device_dict[name] = device

                # will send grad to each corresponding trainer
                if self._world_size > 1:
                    # get idx split from kvstore
                    idx_split = kvstore.get_partid(emb.data_name, idics)
                    if dist:
                        local_machine_id = self._rank // trainers_per_server
                        # all indices should be local 
                        if not th.all(idx_split == local_machine_id):
                            print(f"Rank {self._rank} {name} has remote idx")
                            local_indics[name] = [th.zeros((0,), dtype=th.long, device=th.device("cpu"))]
                            local_grads[name] = [th.zeros((0, emb.embedding_dim), dtype=th.float32, device=th.device("cpu"))]
                            continue

                    idx_split_size = []
                    idics_list = []
                    grad_list = []
                    # split idx and grad first
                    for i in range(kvstore.num_servers):
                        if dist:
                            if i != local_machine_id:
                                continue

                        mask = idx_split == i
                        idx_i = idics[mask]
                        grad_i = grads[mask]

                        if trainers_per_server <= 1:
                            idx_split_size.append(
                                th.tensor([idx_i.shape[0]], dtype=th.int64)
                            )
                            idics_list.append(idx_i)
                            grad_list.append(grad_i)
                        else:
                            kv_idx_split = th.remainder(
                                idx_i, trainers_per_server
                            ).long()
                            # adjustment for replicate caches;
                            # In order to maintain cache consistency, 
                            # add nodes that are cached in caches; adjustment from here
                            # 像数据并行一样先同步梯度不可行! 
                            # 因为optimizer cache是模型并行，更新的时候还要拿optimizer; 这么做更新的时候的读性能下降很多;
                            # mask1=False
                            # if emb.cache != None:
                            #     mask1=emb.cache.cache_flag[idx_i]
                            for j in range(trainers_per_server):
                                mask = kv_idx_split == j
                                # print(f"before = {sum(mask)}, after = {sum(mask | mask1)}")
                                # mask |= mask1
                                # to here. add changes to caches
                                # cost too much, abandoned, use new method
                                idx_j = idx_i[mask]
                                grad_j = grad_i[mask]
                                if th.distributed.get_backend()=='gloo':
                                    idx_split_size.append(
                                        th.tensor([idx_j.shape[0]], dtype=th.int64)
                                    )
                                elif th.distributed.get_backend()=='nccl':
                                    idx_split_size.append(
                                        th.tensor([idx_j.shape[0]], dtype=th.int64, device=device)
                                    )
                                idics_list.append(idx_j)
                                grad_list.append(grad_j)

                    # if one machine launch multiple KVServer, they share the same storage.
                    # For each machine, the pytorch rank is num_trainers *
                    # machine_id + i

                    # use scatter to sync across trainers about the p2p tensor size
                    # Note: If we have GPU nccl support, we can use all_to_all to
                    # sync information here
                    world_size = self._local_world_size if dist else self._world_size
                    rank = self._rank
                    rank_list = list(range(world_size))
                    if dist:
                        rank_list = [rank + local_machine_id * self._local_world_size for rank in rank_list]

                    start = time.time()
                    if th.distributed.get_backend()=='gloo':
                        gather_list = list(
                            th.empty([world_size], dtype=th.int64).chunk(
                                world_size
                            )
                        )
                        alltoall_cpu(
                            rank,
                            rank_list,
                            gather_list,
                            idx_split_size,
                            group=group
                        )
                        # use cpu until we have GPU alltoallv
                        idx_gather_list = [
                            th.empty((int(num_emb),), dtype=idics.dtype)
                            for num_emb in gather_list
                        ]
                        alltoallv_cpu(
                            rank,
                            rank_list,
                            idx_gather_list,
                            idics_list,
                            group=group
                        )
                    
                        local_indics[name] = idx_gather_list
                        grad_gather_list = [
                            th.empty(
                                (int(num_emb), grads.shape[1]), dtype=grads.dtype
                            )
                            for num_emb in gather_list
                        ]
                        alltoallv_cpu(
                            rank,
                            rank_list,
                            grad_gather_list,
                            grad_list,
                            group=group
                        )
                
                        local_grads[name] = grad_gather_list

                    elif th.distributed.get_backend()=='nccl':

                        gather_list = list(
                            th.empty([world_size], dtype=th.int64, device=device).chunk(
                                world_size
                            )
                        )
                        th.distributed.all_to_all(gather_list,idx_split_size,group=group)

                        idx_gather_list = [
                            th.empty((int(num_emb),), dtype=idics.dtype, device=device)
                            for num_emb in gather_list
                        ]
                        th.distributed.all_to_all(idx_gather_list,idics_list,group=group)
                    
                        local_indics[name] = idx_gather_list
                        grad_gather_list = [
                            th.empty(
                                (int(num_emb), grads.shape[1]), dtype=grads.dtype, device=device
                            )
                            for num_emb in gather_list
                        ]
                        th.distributed.all_to_all(grad_gather_list,grad_list,group=group)
                
                        local_grads[name] = grad_gather_list
                    else:
                        raise NotImplementedError("do not support backend other than nccl or gloo!")

                    self._send_grad_time += time.time() - start
                else:
                    local_indics[name] = [idics]
                    local_grads[name] = [grads]

            if self._clean_grad:
                # clean gradient track
                for emb in self._params:
                    emb.reset_trace()
                self._clean_grad = False

            # do local update
            for emb in self._params:
                name = emb.weight.name
                idx = th.cat(local_indics[name], dim=0)
                if len(idx) == 0 and dist:
                    continue
                    
                grad = th.cat(local_grads[name], dim=0)
                # if self._rank==0 and (self._step_times==0 or self._step_times==1):
                #     th.save([idx,grad],f'./log/init_emb/emb_grad_idx_{self._step_times}')

                device = device_dict[name]
                self.update(
                    idx.to(device, non_blocking=True),
                    grad.to(device, non_blocking=True),
                    emb
                )

        self._tot_time = time.time() - start
        # synchronized gradient update
        if self._world_size > 1:
            th.distributed.barrier()
        self._step_times+=1
        
  
    @abstractmethod
    def update(self, idx, grad, emb):
        """Update embeddings in a sparse manner
        Sparse embeddings are updated in mini batches. We maintain gradient states for
        each embedding so they can be updated separately.

        Parameters
        ----------
        idx : tensor
            Index of the embeddings to be updated.
        grad : tensor
            Gradient of each embedding.
        emb : dgl.distributed.DistEmbedding
            Sparse node embedding to update.
        """

    def zero_grad(self):
        """clean grad cache"""
        self._clean_grad = True


def initializer(shape, dtype):
    """Sparse optimizer state initializer

    Parameters
    ----------
    shape : tuple of ints
        The shape of the state tensor
    dtype : torch dtype
        The data type of the state tensor
    """
    arr = th.zeros(shape, dtype=dtype)
    return arr


class SparseAdagrad(DistSparseGradOptimizer):
    r"""Distributed Node embedding optimizer using the Adagrad algorithm.

    This optimizer implements a distributed sparse version of Adagrad algorithm for
    optimizing :class:`dgl.distributed.DistEmbedding`. Being sparse means it only updates
    the embeddings whose gradients have updates, which are usually a very
    small portion of the total embeddings.

    Adagrad maintains a :math:`G_{t,i,j}` for every parameter in the embeddings, where
    :math:`G_{t,i,j}=G_{t-1,i,j} + g_{t,i,j}^2` and :math:`g_{t,i,j}` is the gradient of
    the dimension :math:`j` of embedding :math:`i` at step :math:`t`.

    NOTE: The support of sparse Adagrad optimizer is experimental.

    Parameters
    ----------
    params : list[dgl.distributed.DistEmbedding]
        The list of dgl.distributed.DistEmbedding.
    lr : float
        The learning rate.
    eps : float, Optional
        The term added to the denominator to improve numerical stability
        Default: 1e-10
    """

    def __init__(self, params, lr, eps=1e-10):
        super(SparseAdagrad, self).__init__(params, lr)
        self._eps = eps
        self._defaults = {"_lr": lr, "_eps": eps}
        # We need to register a state sum for each embedding in the kvstore.
        for emb in params:
            assert isinstance(
                emb, DistEmbedding
            ), "SparseAdagrad only supports dgl.distributed.DistEmbedding"

            name = emb.name + "_sum"
            state = DistTensor(
                (emb.num_embeddings, emb.embedding_dim),
                th.float32,
                name,
                init_func=initializer,
                part_policy=emb.part_policy,
                is_gdata=False,
            )
            assert (
                emb.name not in self._state
            ), "{} already registered in the optimizer".format(emb.name)
            self._state[emb.name] = state

    def update(self, idx, grad, emb):
        """Update embeddings in a sparse manner
        Sparse embeddings are updated in mini batches. We maintain gradient states for
        each embedding so they can be updated separately.

        Parameters
        ----------
        idx : tensor
            Index of the embeddings to be updated.
        grad : tensor
            Gradient of each embedding.
        emb : dgl.distributed.DistEmbedding
            Sparse embedding to update.
        """
        eps = self._eps
        clr = self._lr

        state_dev = th.device("cpu")
        exec_dev = grad.device

        # only perform async copies cpu -> gpu, or gpu-> gpu, but block
        # when copying to the cpu, so as to ensure the copy is finished
        # before operating on the data on the cpu
        state_block = state_dev == th.device("cpu") and exec_dev != state_dev

        # the update is non-linear so indices must be unique
        grad_indices, inverse, cnt = th.unique(
            idx, return_inverse=True, return_counts=True
        )
        grad_values = th.zeros(
            (grad_indices.shape[0], grad.shape[1]), device=exec_dev
        )
        grad_values.index_add_(0, inverse, grad)
        grad_values = grad_values / cnt.unsqueeze(1)
        grad_sum = grad_values * grad_values

        # update grad state
        grad_state = self._state[emb.name][grad_indices].to(exec_dev)
        grad_state += grad_sum
        grad_state_dst = grad_state.to(state_dev, non_blocking=True)
        if state_block:
            # use events to try and overlap CPU and GPU as much as possible
            update_event = th.cuda.Event()
            update_event.record()

        # update emb
        std_values = grad_state.sqrt_().add_(eps)
        tmp = clr * grad_values / std_values
        tmp_dst = tmp.to(state_dev, non_blocking=True)

        if state_block:
            std_event = th.cuda.Event()
            std_event.record()
            # wait for our transfers from exec_dev to state_dev to finish
            # before we can use them
            update_event.wait()
        self._state[emb.name][grad_indices] = grad_state_dst

        if state_block:
            # wait for the transfer of std_values to finish before we
            # can use it
            std_event.wait()
        emb._tensor[grad_indices] -= tmp_dst
        # print("adagrad")


class SparseAdam(DistSparseGradOptimizer):
    r"""Distributed Node embedding optimizer using the Adam algorithm.

    This optimizer implements a distributed sparse version of Adam algorithm for
    optimizing :class:`dgl.distributed.DistEmbedding`. Being sparse means it only updates
    the embeddings whose gradients have updates, which are usually a very
    small portion of the total embeddings.

    Adam maintains a :math:`Gm_{t,i,j}` and `Gp_{t,i,j}` for every parameter
    in the embeddings, where
    :math:`Gm_{t,i,j}=beta1 * Gm_{t-1,i,j} + (1-beta1) * g_{t,i,j}`,
    :math:`Gp_{t,i,j}=beta2 * Gp_{t-1,i,j} + (1-beta2) * g_{t,i,j}^2`,
    :math:`g_{t,i,j} = lr * Gm_{t,i,j} / (1 - beta1^t) / \sqrt{Gp_{t,i,j} / (1 - beta2^t)}` and
    :math:`g_{t,i,j}` is the gradient of the dimension :math:`j` of embedding :math:`i`
    at step :math:`t`.

    NOTE: The support of sparse Adam optimizer is experimental.

    Parameters
    ----------
    params : list[dgl.distributed.DistEmbedding]
        The list of dgl.distributed.DistEmbedding.
    lr : float
        The learning rate.
    betas : tuple[float, float], Optional
        Coefficients used for computing running averages of gradient and its square.
        Default: (0.9, 0.999)
    eps : float, Optional
        The term added to the denominator to improve numerical stability
        Default: 1e-8
    """

    def __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-08, cache=None):
        super(SparseAdam, self).__init__(params, lr)
        self._eps = eps
        # We need to register a state sum for each embedding in the kvstore.
        self._beta1 = betas[0]
        self._beta2 = betas[1]
        self._defaults = {
            "_lr": lr,
            "_eps": eps,
            "_beta1": betas[0],
            "_beta2": betas[1],
        }
        self.update_cnt = {}
        self._cache_write_hit_rate = defaultdict(list)
        for emb in params:
            assert isinstance(
                emb, DistEmbedding
            ), "SparseAdam only supports dgl.distributed.DistEmbedding"
            gpu_cache = emb.gpu_cache
            if gpu_cache is not None:
                # drgnn 
                state_mem_gpu_cache = cache[emb.name]
                state_power_gpu_cache = cache[emb.name].clone(write_through=False)
                # print("state_power_gpu_cache.cache_buffer == 0",th.all(state_power_gpu_cache.cache_buffer == 0))
                # print("state_mem_gpu_cache.cache_buffer == 0",th.all(state_mem_gpu_cache.cache_buffer == 0))
                # heta 
                # state_mem_gpu_cache = gpu_cache.clone(write_through=False)
                # state_power_gpu_cache = gpu_cache.clone(write_through=False)
            else:
                state_mem_gpu_cache = None
                state_power_gpu_cache = None

            state_step = DistTensor(
                (emb.num_embeddings,),
                th.float32,
                emb.name + "_step",
                init_func=initializer,
                part_policy=emb.part_policy,
                is_gdata=False,
            )
            state_mem = DistTensor(
                (emb.num_embeddings, emb.embedding_dim),
                th.float32,
                emb.name + "_mem",
                init_func=initializer,
                part_policy=emb.part_policy,
                is_gdata=False,
                gpu_cache=state_mem_gpu_cache,
            )
            state_power = DistTensor(
                (emb.num_embeddings, emb.embedding_dim),
                th.float32,
                emb.name + "_power",
                init_func=initializer,
                part_policy=emb.part_policy,
                is_gdata=False,
                gpu_cache=state_power_gpu_cache
            )
            # for dynamic cache
            if state_power_gpu_cache!=None and state_mem_gpu_cache!=None:
                state_power_gpu_cache._tensor=state_power
                state_mem_gpu_cache._tensor=state_mem

            state = (state_step, state_mem, state_power)
            assert (
                emb.name not in self._state
            ), "{} already registered in the optimizer".format(emb.name)
            self._state[emb.name] = state

            self.update_cnt[emb.name] = 0


    def broadcast_learnable_feature(self, local_idx, local_values, emb):
        if th.distributed.get_backend() != 'nccl':
            print("this cache does the worst in gloo! dont use cache.")
            return
        world_size = th.distributed.get_world_size()
        # choose idx and value need to broadcast, which is in cache.
        mask=emb.cache.cache_flag[local_idx]
        boardcast_idx=local_idx[mask]
        boardcast_values=local_values[mask]
        device=boardcast_values.device
        
        # th.cuda.synchronize()
        # get tensor length
        local_length = th.tensor([boardcast_values.size(0)], dtype=th.int64, device=device)
        all_lengths = [th.empty_like(local_length) for _ in range(world_size)]
        th.distributed.all_gather(all_lengths, local_length)
        # th.cuda.synchronize()

        # all_to_all cached idx and cached values
        idx_list=[th.clone(boardcast_idx) for _ in range(world_size)]
        value_list=[th.clone(boardcast_values) for _ in range(world_size)]
        
        idx_gather_list = [
            th.empty((int(num_emb),), dtype=boardcast_idx.dtype, device=device)
            for num_emb in all_lengths
        ]
        value_gather_list = [
            th.empty(
                (int(num_emb), boardcast_values.shape[1]), dtype=boardcast_values.dtype, device=device
            )
            for num_emb in all_lengths
        ]
        th.distributed.all_to_all(idx_gather_list,idx_list)
        # th.cuda.synchronize()
        th.distributed.all_to_all(value_gather_list,value_list)
        # th.cuda.synchronize()
        
        # get original data and cat
        rank=th.distributed.get_rank()
        idx_gather_list[rank]=local_idx
        value_gather_list[rank]=local_values
        return th.cat(idx_gather_list,dim=0),th.cat(value_gather_list,dim=0)

    def update(self, idx, grad, emb):
        """Update embeddings in a sparse manner
        Sparse embeddings are updated in mini batches. We maintain gradient states for
        each embedding so they can be updated separately.

        Parameters
        ----------
        idx : tensor
            Index of the embeddings to be updated.
        grad : tensor
            Gradient of each embedding.
        emb : dgl.distributed.DistEmbedding
            Sparse embedding to update.
        """
        # 统计没去重的
        # self.update_cnt[emb.name] += len(idx)

        beta1 = self._beta1
        beta2 = self._beta2
        eps = self._eps
        clr = self._lr
        state_step, state_mem, state_power = self._state[emb.name]
        # print(f"rank {self._rank}, state_mem_cache_id {id(state_mem.cache)}")
        state_dev = th.device("cpu")
        exec_dev = grad.device

        # only perform async copies cpu -> gpu, or gpu-> gpu, but block
        # when copying to the cpu, so as to ensure the copy is finished
        # before operating on the data on the cpu
        state_block = state_dev == th.device("cpu") and exec_dev != state_dev

        # the update is non-linear so indices must be unique
        grad_indices, inverse, cnt = th.unique(
            idx, return_inverse=True, return_counts=True
        )
        # 统计去了重的
        self.update_cnt[emb.name] += len(grad_indices)
        # update grad state
        state_idx = grad_indices.to(state_dev)

        # The original implementation will cause read/write contension.
        #    state_step[state_idx] += 1
        #    state_step = state_step[state_idx].to(exec_dev, non_blocking=True)
        # In a distributed environment, the first line of code will send write requests to
        # kvstore servers to update the state_step which is asynchronous and the second line
        # of code will also send read requests to kvstore servers. The write and read requests
        # may be handled by different kvstore servers managing the same portion of the
        # state_step dist tensor in the same node. So that, the read request may read an old
        # value (i.e., 0 in the first iteration) which will cause
        # update_power_corr to be NaN
        start = time.time()
        state_val = state_step[state_idx] + 1
        if state_mem.cache==None:
            orig_mem = state_mem[state_idx]
            orig_power = state_power[state_idx]
        else:
            orig_mem = state_mem[grad_indices]
            orig_power = state_power[grad_indices]
        self._pull_time += time.time() - start
        start = time.time()
        state_step[state_idx] = state_val
        self._push_time += time.time() - start

        start = time.time()
        state_step = state_val.to(exec_dev)
        orig_mem = orig_mem.to(exec_dev)
        orig_power = orig_power.to(exec_dev)
        self._h2d_d2h_time += time.time() - start

        start = time.time()
        grad_values = th.zeros(
            (grad_indices.shape[0], grad.shape[1]), device=exec_dev
        )
        grad_values.index_add_(0, inverse, grad)
        grad_values = grad_values / cnt.unsqueeze(1)
        grad_mem = grad_values
        grad_power = grad_values * grad_values
        update_mem = beta1 * orig_mem + (1.0 - beta1) * grad_mem
        update_power = beta2 * orig_power + (1.0 - beta2) * grad_power
        self._comp_time += time.time() - start

        start = time.time()
        if state_mem.cache==None:
            state_mem[state_idx] = update_mem
            state_power[state_idx] = update_power
        else:
            state_mem[grad_indices] = update_mem
            state_power[grad_indices] = update_power
        self._push_time += time.time() - start - state_mem._h2d_d2h_time - state_power._h2d_d2h_time
        self._h2d_d2h_time += state_mem._h2d_d2h_time + state_power._h2d_d2h_time

        start = time.time()
        update_mem_corr = update_mem / (
            1.0 - th.pow(th.tensor(beta1, device=exec_dev), state_step)
        ).unsqueeze(1)
        update_power_corr = update_power / (
            1.0 - th.pow(th.tensor(beta2, device=exec_dev), state_step)
        ).unsqueeze(1)
        std_values = clr * update_mem_corr / (th.sqrt(update_power_corr) + eps)
        self._comp_time += time.time() - start

        start = time.time()
        # std_values_dst = std_values.to(state_dev, non_blocking=True)
        if emb._tensor.cache==None:
            emb_values = emb._tensor[state_idx]
        else:
            emb_values = emb._tensor[grad_indices]
        self._pull_time += time.time() - start - emb._tensor._h2d_d2h_time
        self._h2d_d2h_time += emb._tensor._h2d_d2h_time

        # orig_emb_values=emb_values.clone()

        start = time.time()
        emb_values = emb_values.to(exec_dev)
        self._h2d_d2h_time += time.time() - start

        start = time.time()
        emb_values -= std_values
        self._comp_time += time.time() - start

        # if self._step_times==0 or self._step_times==1:
        #     th.save(\
        #         {"grad_indices":grad_indices,"grad_values":grad_values,\
        #          "inverse":inverse,"idx":idx,"grad":grad,"cnt":cnt,\
        #          "orig_mem":orig_mem,"orig_power":orig_power,"update_mem":update_mem,\
        #          "update_power":update_power,"std_values":std_values,\
        #             "orig_emb_values":orig_emb_values,"emb_values":emb_values}\
        #                 ,f'./log/init_emb/emb_grad_idx_{self._rank}_{self._step_times}')


        start = time.time()
        if emb._tensor.cache==None:
            emb._tensor[state_idx] = emb_values
        else:
            # emb._tensor[grad_indices] = emb_values
            # broadcast updated learnable features.
            new_idx,new_values=self.broadcast_learnable_feature(grad_indices,emb_values,emb)
            self._h2d_d2h_time+=time.time()-start

            start = time.time()
            emb._tensor[new_idx]=new_values

            # rank=th.distributed.get_rank()
            # if rank==1 or rank==2:
            #     print(f"rank {rank} grad_indices",grad_indices.shape, grad_indices.dtype)
            #     print(f"rank {rank} emb_values",emb_values.shape,emb_values.dtype)
            #     print(f"rank {rank} new_idx",new_idx.shape,new_idx.dtype)
            #     print(f"rank {rank} new_values",new_values.shape,new_values.dtype)
            #     local_idx_not_cached=grad_indices[~emb.cache.cache_flag[grad_indices]]
            #     new_idx_not_cached=new_idx[~emb.cache.cache_flag[new_idx]]
            #     print(f"rank {rank} local_idx not cached",local_idx_not_cached.device, local_idx_not_cached.shape,'\n',
            #         f"rank {rank} all_to_all_idx not cached",new_idx_not_cached.device, new_idx_not_cached.shape)
            #     print(f"rank {rank} th.equal(local_idx_not_cached, new_idx_not_cached)", th.equal(local_idx_not_cached, new_idx_not_cached))
            #     local_value_not_cached=emb_values[~emb.cache.cache_flag[grad_indices]]
            #     new_value_not_cached=new_values[~emb.cache.cache_flag[new_idx]]
            #     print(f"rank {rank} local_value not cached",local_value_not_cached.device, local_value_not_cached.shape,'\n',
            #         f"rank {rank} a2a_value not cached",new_value_not_cached.device, new_value_not_cached.shape)
            #     print(f"rank {rank} th.equal(local_value_not_cached, new_value_not_cached)", th.equal(local_value_not_cached, new_value_not_cached))
            #     dram_value=emb._tensor[new_idx_not_cached]
            #     print(f"rank {rank} dram_value", dram_value.device, dram_value.shape)
            #     print(f"rank {rank} th.equal(local_value_not_cached, dram_value)", th.equal(local_value_not_cached, dram_value))
            #     print(f"rank {rank} th.equal(new_value_not_cached, dram_value)", th.equal(new_value_not_cached, dram_value))
            
            # rank=th.distributed.get_rank()
            # if rank==0 or rank==3:
            #     print(f"rank {rank} grad_indices",grad_indices.shape, grad_indices.dtype)
            #     print(f"rank {rank} emb_values",emb_values.shape,emb_values.dtype)
            #     print(f"rank {rank} new_idx",new_idx.shape,new_idx.dtype)
            #     print(f"rank {rank} new_values",new_values.shape,new_values.dtype)
            #     local_idx_cached=grad_indices[emb.cache.cache_flag[grad_indices]]
            #     new_idx_cached=new_idx[emb.cache.cache_flag[new_idx]]
            #     print(f"rank {rank} local_idx cached",local_idx_cached.device, local_idx_cached.shape,'\n',
            #         f"rank {rank} all_to_all_idx cached",new_idx_cached.device, new_idx_cached.shape)
            #     print(f"rank {rank} th.equal(local_idx_cached, new_idx_cached)", th.equal(local_idx_cached, new_idx_cached))
            #     local_value_cached=emb_values[emb.cache.cache_flag[grad_indices]]
            #     new_value_cached=new_values[emb.cache.cache_flag[new_idx]]
            #     print(f"rank {rank} local_value cached",local_value_cached.device, local_value_cached.shape,'\n',
            #         f"rank {rank} a2a_value cached",new_value_cached.device, new_value_cached.shape)
            #     print(f"rank {rank} th.equal(local_value_cached, new_value_cached)", th.equal(local_value_cached, new_value_cached))
            #     gpu_value=emb._tensor[new_idx_cached]
            #     print(f"rank {rank} gpu_value", gpu_value.device, gpu_value.shape)
            #     print(f"rank {rank} th.equal(local_value_cached, gpu_value)", th.equal(local_value_cached, gpu_value))
            #     print(f"rank {rank} th.equal(new_value_cached, gpu_value)", th.equal(new_value_cached, gpu_value))
            
            # cache=emb._tensor._gpu_cache.cache_buffer
            # world_size = th.distributed.get_world_size()
            # rank = th.distributed.get_rank()
            
            # # 使用 all_gather 收集所有 GPU 上的张量
            # tensor_list = [th.empty_like(cache) for _ in range(world_size)]
            # th.distributed.all_gather(tensor_list,cache)

            # if rank==0:
            #     diff_indices=[]
            #     for i in range(len(tensor_list)):
            #         for j in range(i+1,len(tensor_list)):
            #             if not th.equal(tensor_list[i], tensor_list[j]):
            #                 for idx in range(cache.size(0)):
            #                     if not th.equal(tensor_list[i][idx], tensor_list[j][idx]):
            #                         diff_indices.append((i, j, idx))
            #             else:
            #                 print(i,j,'equals!')
            #     print(diff_indices)

        if state_mem.cache is not None:
            ntype=emb.name.split('node_emb_')[-1]
            # 2 optimizer states, append twice
            self._cache_write_hit_rate[f"{ntype}_opt"].append(state_mem.cache.cache_write_hit_rate)
            self._cache_write_hit_rate[f"{ntype}_opt"].append(state_power.cache.cache_write_hit_rate)
            self._cache_write_hit_rate[ntype].append(emb._tensor.cache.cache_write_hit_rate)
        self._push_time += time.time() - start - emb._tensor._h2d_d2h_time
        self._h2d_d2h_time += emb._tensor._h2d_d2h_time

    # def update_cache(self, cache_counter):
    #     for emb in self._params:
    #         _, state_mem, state_power = self._state[emb.name]
    #         gpu_id=th.distributed.get_rank()
    #         if state_mem.cache!=None and state_power.cache!=None:
    #             state_mem.cache.update(self._local_world_size, gpu_id, cache_counter[emb.name])
    #             state_power.cache.update(self._local_world_size, gpu_id, cache_counter[emb.name])

    @property
    def cache_write_hit_rate(self):
        # print("cache_write_hit_rate",self._cache_write_hit_rate)
        return {
            ntype: f"{np.mean(self._cache_write_hit_rate[ntype])*100:.2f}" for ntype in self._cache_write_hit_rate if len(self._cache_write_hit_rate[ntype]) > 0 
        }
