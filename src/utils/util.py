import dgl
import os
import dgl.backend as F
import torch
# from graphviz import Digraph
import torch.distributed as dist
import torch.nn.functional as func
import numpy as np
from .draw import draw_acc_epoch, draw_acc_time,draw_cdf
from tqdm import tqdm
import psutil

from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

# class HookTool: 
#     def __init__(self):
#         self.activations = {}

#     def _write(self,input):
#         if isinstance(input,dict):
#             return {k:v.detach().cpu() for k,v in input.items()}
#         elif isinstance(input,list):
#             return [i.detach().cpu() for i in input]
#         elif isinstance(input,torch.Tensor):
#             return input.detach().cpu()
#         elif isinstance(input,tuple):
#             return [i.detach().cpu() for i in input]
#         else:
#             # print(input,type(input))
#             return input
        
#     def hook_fn(self,module, input, output):
#         module_name = module.__class__.__name__
#         self.activations[f"{module_name}_input"]=self._write(input)
#         self.activations[f"{module_name}_output"]=self._write(output)

# def get_feas_by_hook(model):
#     fea_hooks = []
#     for n, m in model.named_modules():
#         cur_hook = HookTool()
#         m.register_forward_hook(cur_hook.hook_fn)
#         fea_hooks.append(cur_hook)

#     return fea_hooks

# node_id = 0

# def add_node(G: Digraph, input_grad_fn, gradfn_to_viznode:dict,):
#     if input_grad_fn in gradfn_to_viznode:
#         return
#     global node_id
#     gradfn_to_viznode[input_grad_fn] = str(node_id)
#     node_id += 1
#     print(f"adding {str(type(input_grad_fn))}=")
#     if type(input_grad_fn).__name__ == "AccumulateGrad":
#         v = input_grad_fn.variable
#         G.node(gradfn_to_viznode[input_grad_fn], label=f"AccumulateGrad:{list(v.shape)},{str(v.dtype)}", shape='rectangle', style='filled',)
#     else:
#         G.node(gradfn_to_viznode[input_grad_fn], label=f"{input_grad_fn.name()}")

# def _viz_graph(G: Digraph, grad_fn, gradfn_to_viznode: dict, visited: set):

#     if grad_fn is None:
#         print(f"grad_fn is None")
#         return

#     if grad_fn in visited:
#         return
#     visited.add(grad_fn)

#     for (input_grad_fn, index) in grad_fn.next_functions:
#         # 创建节点
#         #input_grad_fn 有3种情况：
#         # 1. AccumulateGrad
#         # 2. grad_fn比如AddBackward0
#         # 3. None
#         if input_grad_fn is None:
#             continue

#         add_node(G, input_grad_fn, gradfn_to_viznode)

#         #创建边
#         G.edge(head_name=gradfn_to_viznode[grad_fn], tail_name=gradfn_to_viznode[input_grad_fn], label=str(index))

#         #如果不是AccumulateGrad，则递归处理
#         if type(input_grad_fn).__name__ == "AccumulateGrad":       
#             continue
#         _viz_graph(G, input_grad_fn, gradfn_to_viznode, visited)

# def viz_graph(t: torch.Tensor) -> Digraph:
#     G = Digraph()
#     if t.grad_fn is not None:
#         global node_id
#         G.node("out", label=f"out:{list(t.shape)},{str(t.dtype)}", shape='rectangle', style='filled', )
#         gradfn_to_viznode = {}
#         add_node(G, t.grad_fn, gradfn_to_viznode)
#         G.edge(head_name="out", tail_name=gradfn_to_viznode[t.grad_fn])
#         _viz_graph(G, t.grad_fn, gradfn_to_viznode, set())
#     else:
#         assert t.grad_fn is not None, f"{t} should have a grad_fn"

#     return G

def nvidia_smi_usage():
	logger = ''
	nvmlInit()
	handle = nvmlDeviceGetHandleByIndex(0)
	info = nvmlDeviceGetMemoryInfo(handle)
	# logger += "\n Nvidia-smi: " + str((info.used) / 1024 / 1024 / 1024) + " GB"
	return (info.used) / 1024 / 1024 / 1024

def gpu_capacity():
    max_device_mem = torch.cuda.mem_get_info()[1]/1024/1024
    max_allocated_mem = torch.cuda.max_memory_allocated()/1024/1024
    nvidia_smi_mem=nvidia_smi_usage()
    print("max_device_mem {:.3f} MB, max_allocated_mem {:.3f} MB, nvidia_smi_mem: {:.3f} GB"
          .format(max_device_mem, max_allocated_mem, nvidia_smi_mem))

def print_graph(g):
    print(f"graph {g} has:")
    print("node:")
    for ntype in g.ntypes:
        for key in g.nodes[ntype].data.keys():
            print(f"ntype:{ntype} has {key}")

    for etype in g.canonical_etypes:
        for key in g.edges[etype].data.keys():
            print(f"etype:{etype} has {key}")
    
    print("canonical_etypes=", g.canonical_etypes)

def write_his_embed(embed_tables, epoch, args, root='log/his_emb'):
    root=os.path.join(os.getcwd(),root)
    print(f'mkdir: {root}')
    os.makedirs(root, exist_ok=True)
    data_dict={}
    for ntype in embed_tables.keys():
        cache_idx = torch.arange(embed_tables[ntype].weight.shape[0], dtype=torch.long)
        data_dict[ntype]=embed_tables[ntype].weight[cache_idx].to(torch.device('cpu'))
        print(ntype,data_dict[ntype].shape,'\n',data_dict[ntype],)
    file_name=f"{args.model}_{args.graph_name}_{args.num_hidden}_{epoch}"
    save_path=(os.path.join(root, file_name))
    torch.save(data_dict, save_path)
    print(f'save_path: {save_path}')

def _cmp_cos_sim(data_dict1, data_dict2):
    cos_dict={}
    for ntype in data_dict1.keys():
        assert data_dict1[ntype].shape==data_dict2[ntype].shape, f'dict {ntype} wrong shape!'
        _len=data_dict1[ntype].shape[0]
        _tensor=torch.zeros((_len))
        for i in range(_len):
            _tensor[i]=func.cosine_similarity(data_dict1[ntype][i], data_dict2[ntype][i], dim=0)
        cos_dict[ntype]=_tensor
    return cos_dict

def _cul_cdf(cos_dict):
    res={}
    # part by ntype
    for ntype in cos_dict.keys():
        _len=cos_dict[ntype].shape[0]
        _tensor=torch.zeros((101))
        for i in range(_len):
            # if idx<=cos_dict[ntype][i]<idx+1, _tensor[idx]+=1
            idx=cos_dict[ntype][i]*100
            # idx = torch.min(torch.tensor(99),idx)   # incase cos=1
            _tensor[torch.floor(idx).int()]+=1
        res[ntype]=_tensor

    # total
    total_tensor=torch.zeros((101))
    for ntype in res.keys():
        total_tensor+=res[ntype]

    def _prefixSumDivSum(t):
        _sum=0
        for i in range(t.shape[0]):
            _sum+=t[i]
            t[i]=_sum
        t=t/_sum
        return t
    
    # prefixSum/sum
    for ntype in res.keys():
        _tensor=res[ntype]
        res[ntype]=_prefixSumDivSum(_tensor)

    total_tensor=_prefixSumDivSum(total_tensor)

    return res, total_tensor

def read_his_embed(graph_name='igb-part-small', step=20, epoch=-1, model='rgcn', num_hidden=64, root='log/his_emb'):
    directory_path=os.path.join(os.getcwd(),root)
    print(directory_path)
    file_list = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            _tmp=file.split('_')
            if len(_tmp)!=4:
                print(file,'pass!')
                continue
            _graph_name, _model, _num_hidden, _epoch=_tmp
            if _graph_name==_graph_name and _model==_model and _num_hidden==_num_hidden:
                file_list.append(file)
    file_list=sorted(file_list, key=lambda x:(int(x.split('_')[-1]),x))
    if epoch==-1:
        epoch=int(file_list[-1].split('_')[-1])+1
        assert len(file_list) == epoch, 'missing epoch!'
    
    total_tensor_dict={}
    for i in range(step, epoch): 
        save_path1=os.path.join(directory_path,file_list[i-step])
        save_path2=os.path.join(directory_path,file_list[i])
        data_dict1 = torch.load(save_path1)
        data_dict2 = torch.load(save_path2)
        print("loaded:", save_path1, save_path2)
        
        cos_dict=_cmp_cos_sim(data_dict1, data_dict2)
        cdf_tensor_dict, total_tensor=_cul_cdf(cos_dict)
        # print(cdf_tensor_dict, total_tensor)
        # for ntype in cdf_tensor_dict.keys():
        #     print(ntype, cdf_tensor_dict[ntype])
        # print('total', total_tensor)
        cdf_tensor_dict['total']=total_tensor
        file_name=f'dict_tensor/{model}_{graph_name}_{num_hidden}_{i-step}_{i}'
        torch.save(cdf_tensor_dict, file_name)
        print(f'tensor {file_name} saved')
        total_tensor_dict[i-step]=total_tensor
        # fig_name=f'output/{model}_{graph_name}_{num_hidden}_{i-step}_{i}.png'
        # draw_cdf(cdf_tensor_dict, fig_name)

    torch.save(total_tensor_dict, f'dict_tensor/{model}_{graph_name}_{num_hidden}_total')
    print(f'tensor {file_name} saved')
    # fig_name=f'output/{model}_{graph_name}_{num_hidden}_total.png'
    # draw_cdf(total_tensor_dict, fig_name)

def _draw_all_cdf(root='dict_tensor'):
    tensor_path=os.path.join(os.getcwd(), 'dict_tensor')
    print(tensor_path)
    for root, dirs, files in os.walk(tensor_path):
        for file in files:
            save_path=os.path.join(tensor_path,file)
            print('loading:', save_path)
            data_dict = torch.load(save_path)
            fig_name=f'output/{file}.png'
            draw_cdf(data_dict, fig_name)

def _test_different_run():
    save_path1='/gf3/home/jgqj/test_code/heta/log/his_emb/rgcn_igb-part-small_64_99'
    save_path2='/gf3/home/jgqj/test_code/heta/log/his_emb/rgcn_igb-part-small_64_99_copy'
    data_dict1 = torch.load(save_path1)
    data_dict2 = torch.load(save_path2)
    print("loaded:", save_path1, save_path2)
    
    cos_dict=_cmp_cos_sim(data_dict1, data_dict2)
    cdf_tensor_dict, total_tensor=_cul_cdf(cos_dict)
    cdf_tensor_dict['total']=total_tensor
    draw_cdf(cdf_tensor_dict, f'epoch_99_in_2run.png')

def mag240_pca(paper_feat, batch_size=615360):
    componentsT=torch.load('ipca.components_.T.pt').half()
    mean=torch.load('ipca.mean_.pt').half()
    def _p(ten):
        print(type(ten),ten.shape,ten.dtype)
    _p(paper_feat)
    _p(componentsT)
    _p(mean)
    min_batch_size=componentsT.shape[1]
    n_samples=paper_feat.shape[0]

    from torchdr import IncrementalPCA
    device=torch.device('cuda')
    mean.to(device)
    componentsT.to(device)
    new_paper_feat=torch.zeros((n_samples,min_batch_size),dtype=torch.float16,device=paper_feat.device)
    for batch in tqdm(IncrementalPCA.gen_batches(n_samples, batch_size, min_batch_size)):
        X_batch = paper_feat[batch].to(device)
        # print(X_batch.shape)
        X_batch = X_batch - mean
        X_batch = X_batch @ componentsT
        new_paper_feat[batch] = X_batch.to(new_paper_feat.device)
    print(new_paper_feat.shape)
    return new_paper_feat


def free_h():
    mem = psutil.virtual_memory()
    # 系统总计内存
    zj = float(mem.total) / 1024 / 1024 / 1024
    # 系统已经使用内存
    ysy = float(mem.used) / 1024 / 1024 / 1024
    # 系统空闲内存
    kx = float(mem.free) / 1024 / 1024 / 1024
    print('系统总计内存:%d.3GB' % zj)
    print('系统已经使用内存:%d.3GB' % ysy)
    print('系统空闲内存:%d.3GB' % kx)

def extract_values_from_file(file_path):
    test_acc_list = []
    epoch_time_list = []

    try:
        with open(file_path, 'r') as file:
            for line in file:
                # 检查是否包含 "Test Acc"，并提取后面的数字
                if "Test Acc" in line:
                    start_index = line.find("Test Acc") + len("Test Acc")
                    # 提取数字部分
                    number = line[start_index:].strip().split()[0][:-1]
                    try:
                        test_acc_list.append(float(number))
                    except ValueError:
                        print(f"Warning: Cannot convert '{number}' to float in 'Test Acc' line.")

                # 检查是否包含 "Part 0, Epoch Time(s): "，并提取后面的数字
                if "Part 0, Epoch Time(s): " in line:
                    start_index = line.find("Part 0, Epoch Time(s): ") + len("Part 0, Epoch Time(s): ")
                    # 提取数字部分
                    number = line[start_index:].strip().split()[0][:-1]
                    try:
                        epoch_time_list.append(float(number))
                    except ValueError:
                        print(f"Warning: Cannot convert '{number}' to float in 'Epoch Time' line.")

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

    return test_acc_list, epoch_time_list

def extract_key_word_from_file(file_path, key_word):
    ret_list = []

    try:
        with open(file_path, 'r') as file:
            for line in file:
                # 检查是否包含 key_word，并提取后面的数字
                if key_word in line:
                    start_index = line.find(key_word) + len(key_word)
                    # 提取数字部分
                    number = line[start_index:].strip().split()[0][:-1]
                    try:
                        ret_list.append(float(number))
                    except ValueError:
                        print(f"Warning: Cannot convert '{number}' to float in {key_word} line.")

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

    return ret_list

def extract_str_from_file(file_path, key_word,end_with=None):
    ret_list = []

    try:
        with open(file_path, 'r') as file:
            for line in file:
                # 检查是否包含 key_word，并提取后面的数字
                if key_word in line:
                    start_index = line.find(key_word) + len(key_word)
                    if end_with!=None:
                        end_index = line.find(end_with,start_index)+1
                        # 提取数字部分
                        number = line[start_index:end_index].strip()
                    else:
                        # 提取数字部分
                        number = line[start_index:].strip()
                    ret_list.append(number)

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

    return ret_list

if __name__=='__main__':
    # 示例使用
    
    # file_path1 = '/gf3/home/jgqj/test_code/heta/log/Heta_rgcn_igb-part-small_miss_penalty_64_1parts_1.log'
    # file_path2 = '/gf3/home/jgqj/test_code/heta/log/Heta_rgcn_igb-part-small_miss_penalty_64_1parts_2.log'
    # file_path3 = '/gf3/home/jgqj/test_code/heta/log/Heta_rgcn_igb-part-small_miss_penalty_64_1parts_3.log'
    # file_path4 = '/gf3/home/jgqj/test_code/heta/log/Heta_rgcn_igb-part-small_miss_penalty_64_1parts_4.log'
    # file_path5 = '/gf3/home/jgqj/test_code/heta/log/Heta_rgcn_igb-part-small_none_64_1parts.log'
    # file_path6 = '/gf3/home/jgqj/test_code/heta/log/Heta_rgcn_igb-part-small_miss_penalty_64_1parts_6.log'
    # file_path7 = '/gf3/home/jgqj/test_code/heta/log/save_log/cache实现测试/Heta_rgcn_igb-part-small_miss_penalty_64_1parts_7.log'
    # file_path8 = '/gf3/home/jgqj/test_code/heta/log/Heta_rgcn_igb-part-small_miss_penalty_64_1parts_8.log'
    # file_path9 = '/gf3/home/jgqj/test_code/heta/log/Heta_rgcn_igb-part-small_miss_penalty_64_1parts_9.log'
    # file_path10 = '/gf3/home/jgqj/test_code/heta/log/save_log/cache实现测试/Heta_rgcn_igb-part-small_miss_penalty_64_1parts_10.log'
    # file_paths=[file_path6,file_path7,file_path8,file_path9,file_path10]
    # file_path1='/gf3/home/jgqj/test_code/heta/log/Heta_rgcn_mag240m_none_64_1parts.log'
    # file_path2='/gf3/home/jgqj/test_code/heta/log/Heta_rgcn_mag240m-pca_none_64_1parts_128_f2cgt.log'
    # file_path1='/gf3/home/jgqj/test_code/heta/log/save_log/Heta_rgcn_igb-full-small_none_64_1parts.log'
    # file_path2='/gf3/home/jgqj/test_code/heta/log/save_log/Heta_rgcn_igb-full-small_none_64_reduction_old_train_nid.log'
    # file_path3='/gf3/home/jgqj/test_code/heta/log/save_log/Heta_rgcn_igb-full-small_none_64_reduction.log'
    file_paths=['/gf3/home/jgqj/test_code/hydro/log/drgnn_rgat_mag240m_miss_penalty_64_4.log',
                '/gf3/home/jgqj/test_code/hydro/exp/exp15/log/save/Heta_rgat_mag240m_none_64_128,8_epoch185.log',
                '/gf3/home/jgqj/test_code/hydro/exp/exp15/log/save/baseline/heta_rgat_mag240m_none_64_acc.log']
    # file_paths=[file_path1,file_path2,file_path3,file_path4]
    # file_paths=[file_path1,file_path2,file_path3,file_path4,file_path5,file_path6,file_path7,file_path8,file_path9]
    tmp_dict1={}
    tmp_dict2={}
    for i in range(len(file_paths)):
        test_acc_values, epoch_time_values=extract_values_from_file(file_paths[i])
        tmp_dict1[str(i+1)]=test_acc_values
        tmp_dict2[str(i+1)]=[test_acc_values, epoch_time_values]

    draw_acc_epoch(tmp_dict1, 185,"acc_epoch.png")

    # draw_acc_time(tmp_dict2,1000,"acc_time.png")
