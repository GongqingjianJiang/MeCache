"""Provide utils for distributed sparse optimizers
"""
from typing import Optional

import torch as th
import torch.distributed as dist


def alltoall_cpu(rank, rank_list, output_tensor_list, input_tensor_list, group: Optional[dist.ProcessGroup] = None):
    """Each process scatters list of input tensors to all processes in a cluster
    and return gathered list of tensors in output list. The tensors should have the same shape.

    Parameters
    ----------
    rank : int
        The rank of current worker
    world_size : int
        The size of the entire
    output_tensor_list : List of tensor
        The received tensors
    input_tensor_list : List of tensor
        The tensors to exchange
    """
    input_tensor_list = [
        tensor.to(th.device("cpu")) for tensor in input_tensor_list
    ]
    for i, r in enumerate(rank_list):
        dist.scatter(
            output_tensor_list[i], input_tensor_list if r == rank else [], src=r, group=group
        )


def alltoallv_cpu(rank, rank_list, output_tensor_list, input_tensor_list, group: Optional[dist.ProcessGroup] = None):
    """Each process scatters list of input tensors to all processes in a cluster
    and return gathered list of tensors in output list.

    Parameters
    ----------
    rank : int
        The rank of current worker
    world_size : int
        The size of the entire
    output_tensor_list : List of tensor
        The received tensors
    input_tensor_list : List of tensor
        The tensors to exchange
    """
    # send tensor to each target trainer using torch.distributed.isend
    # isend is async
    senders = []
    for i, r in enumerate(rank_list):
        if r == rank:
            output_tensor_list[i] = input_tensor_list[i].to(th.device("cpu"))
        else:
            sender = dist.isend(
                input_tensor_list[i].to(th.device("cpu")), dst=r, group=group
            )
            senders.append(sender)

    for i, r in enumerate(rank_list):
        if r != rank:
            dist.recv(output_tensor_list[i], src=r, group=group)

    th.distributed.barrier(group)
