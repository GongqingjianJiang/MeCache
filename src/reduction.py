import dgl
import os
import torch
from tqdm import tqdm
import dgl.function as fn
from torchdr import IncrementalPCA
import gc, time
import pickle
import math
from .l2pca import L2NormalizedPCA

# 有bug，晚点改
def get_reduction_mask1(hg,predict_category):
    """
    input: dgl heterograph with feat, train_mask(maybe test_mask), topo
    output: mask, indicates which node to use which reduction strategy
    """
    print(hg.ntypes)
    print(hg.etypes)
    for ntype in hg.ntypes:
        print(hg.nodes[ntype].data['feat'].shape)
        print(hg.nodes(ntype))
        num_nodes=hg.nodes[ntype].data['feat'].shape[0]
        hg.nodes[ntype].data['strategy']=torch.full(size=(num_nodes,), fill_value=float('inf'), dtype=torch.float)
        hg.nodes[ntype].data['id']=torch.arange(num_nodes)

    print(hg.nodes[predict_category].data['train_mask'])
    train_mask=hg.nodes[predict_category].data['train_mask']
    hg.nodes[ntype].data['strategy'][train_mask]=0
    # add self loop.
    hg = dgl.remove_self_loop(hg, etype='cites')
    hg = dgl.add_self_loop(hg, etype='cites')
    def _add_one(edges):
        src_id = edges.src['id']
        dst_id = edges.dst['id']
        message = torch.where(
            src_id != dst_id,
            edges.src['strategy'] + 1,
            edges.src['strategy']
        )
        message = torch.where(
            edges.dst['strategy'] == 0,
            0,
            message
        )
        return {'m': message}
    for i in range(10):
        for etype in hg.etypes:
            hg.update_all(
                _add_one, fn.min("m", "strategy"), etype=etype
            )
    for etype in hg.etypes:
        print(f"{etype} strategy=",hg.nodes[ntype].data['strategy'])
    print(hg.nodes[ntype].data['strategy'][train_mask])

# 用seed代替
def _get_reduction_mask(hg, predict_category, reduction_level):
    """
    input: dgl heterograph with feat, train_mask(maybe test_mask), topo
    output: mask, indicates which node to use which reduction strategy,
        mask={ntype:{reduction_level[k]:ntype_k_mask}}
    """
    mask={}
    for ntype in hg.ntypes:
        if 'feat' in hg.nodes[ntype].data.keys():
            num_nodes=len(hg.nodes(ntype))
            mask[ntype]=torch.full(size=(num_nodes,), fill_value=reduction_level[1],dtype=torch.int)
            if predict_category==ntype:
                mask[ntype]=torch.full(size=(num_nodes,), fill_value=reduction_level[0],dtype=torch.int)

    for ntype in hg.ntypes:
        if 'feat' not in hg.nodes[ntype].data.keys():
            continue
        tmp_mask = mask[ntype]
        mask[ntype] = {}
        for dim in reduction_level:
            mask[ntype][dim] = tmp_mask == dim
        print(ntype,mask[ntype])
    
    return mask

def _multi_level_pca(name, feat, mask, reduction_level, batch_size=461520):
    '''
    input: feat=ntype_tensor
        mask={reduction_level[k]:ntype_k_mask}
    return: feat={reduction_level[k]:ntype_k_feat}
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("IncrementalPCA.device=",device)
    ipca=IncrementalPCA(n_components=reduction_level[0], batch_size=batch_size, device=device)
    torch.cuda.reset_peak_memory_stats(device)
    print(f"{name} start IncrementalPCA......")
    start = time.time()
    ipca.fit(feat, check_input=False)
    print(type(ipca.mean_),ipca.mean_.shape,ipca.mean_.dtype)
    print(type(ipca.components_),ipca.components_.shape,ipca.components_.dtype)

    components=ipca.components_.clone().to(torch.float32)
    mean=ipca.mean_.clone().to(torch.float32)

    # Clean up and free GPU memory.
    del ipca
    torch.cuda.empty_cache()
    gc.collect()

    n_samples=feat.shape[0]

    device=torch.device('cuda')
    feat_device=torch.device('cpu')
    mean.to(device)
    ret={}
    for dim in reduction_level:
        componentsT=components[:dim].T
        print("componentsT.shape",componentsT.shape)
        componentsT.to(device)
        min_batch_size=componentsT.shape[1]
        new_feat=torch.zeros((n_samples,min_batch_size),dtype=feat.dtype,device=feat_device)
        for batch in tqdm(IncrementalPCA.gen_batches(n_samples, batch_size, min_batch_size)):
            X_batch = feat[batch].to(device)
            X_batch = X_batch - mean
            X_batch = X_batch @ componentsT
            new_feat[batch] = X_batch.to(new_feat.device)
        
        ret[dim]=new_feat[mask[dim]]
        print(ret[dim].shape)

    ipca_time = time.time() - start
    print(f"IncrementalPCA ends, takes {ipca_time} sec.")
    return ret

def _multi_level_L2pca(name, feat, mask, reduction_level, batch_size=461520):
    '''
    input: feat=ntype_tensor
        mask={reduction_level[k]:ntype_k_mask}
    return: feat={reduction_level[k]:ntype_k_feat}
    只能在小数据集上用, 测试异构图上L2PCA效果
    '''
    start=time.time()
    print(f"{name} start L2PCA......")
    ret={}
    for dim in reduction_level:
        if torch.all(~mask[dim]):
            ret[dim]=torch.tensor([],dtype=torch.float32)
            continue
        l2pca=L2NormalizedPCA(n_components=dim, restore_norms=False)
        l2pca.fit(feat)
        new_feat=l2pca.transform(feat)
        new_feat=torch.from_numpy(new_feat).to(torch.float32)
        ret[dim]=new_feat[mask[dim]]
        print(ret[dim].shape)

    ipca_time = time.time() - start
    print(f"L2PCA ends, takes {ipca_time} sec.")
    return ret

def write_reductioned_feat(hg, predict_category, reduction_level, path, use_L2=False):
    mask=_get_reduction_mask(hg, predict_category, reduction_level)
    for ntype in hg.ntypes:
        if 'feat' not in hg.nodes[ntype].data.keys():
            continue
        if not use_L2:
            feat=_multi_level_pca(ntype, hg.nodes[ntype].data['feat'],
                                    mask[ntype],reduction_level)
        else:
            feat=_multi_level_L2pca(ntype, hg.nodes[ntype].data['feat'],
                                    mask[ntype],reduction_level)
        num_nodes=len(hg.nodes(ntype))
        feat=Feat(num_nodes,mask[ntype],feat,reduction_level,'cpu')
        feat.save(os.path.join(path, f'{ntype}.pkl'))

def _get_reduction_mask_large(hg, predict_category, reduction_level):
    """
    input: dgl heterograph with feat, train_mask(maybe test_mask), topo
    output: mask, indicates which node to use which reduction strategy,
        mask={ntype:{reduction_level[k]:ntype_k_mask}}
    """
    mask={}
    for ntype in hg.ntypes:
        num_nodes=len(hg.nodes(ntype))
        mask[ntype]=torch.full(size=(num_nodes,), fill_value=reduction_level[1],dtype=torch.int)
        if predict_category==ntype:
            mask[ntype]=torch.full(size=(num_nodes,), fill_value=reduction_level[0],dtype=torch.int)

    for ntype in hg.ntypes:
        tmp_mask = mask[ntype]
        mask[ntype] = {}
        for dim in reduction_level:
            mask[ntype][dim] = tmp_mask == dim
        print(ntype,mask[ntype])
    
    return mask

def _gen_large_feat(name, node_num, mask, reduction_level):
    print(f"{name} start gen feat......")
    start = time.time()

    ret={}
    for dim in reduction_level:
        new_feat=torch.zeros((node_num,dim),dtype=torch.float32)
        torch.nn.init.uniform_(new_feat, -1.0, 1.0)
        ret[dim]=new_feat[mask[dim]]
        print(ret[dim].shape)

    print(f"Feat gen ends, takes {time.time() - start} sec.")
    return ret

def generate_reductioned_feat(hg, predict_category, reduction_level, path):
    '''
    generate reductioned feat for igbh-large since no evaluate are needed for evaluation.
    '''
    mask=_get_reduction_mask_large(hg, predict_category, reduction_level)
    for ntype in hg.ntypes:
        num_nodes=len(hg.nodes(ntype))
        feat=_gen_large_feat(ntype,num_nodes,mask[ntype],reduction_level)
        feat=Feat(num_nodes,mask[ntype],feat,reduction_level,'cpu')
        feat.save(os.path.join(path, f'{ntype}.pkl'))

def write_mag_reductioned_feat(hg, predict_category, reduction_level, path):
    mask=_get_reduction_mask(hg, predict_category, reduction_level)
    for ntype in hg.ntypes:
        if 'feat' not in hg.nodes[ntype].data.keys():
            continue
        feat={128:mag240_pca(hg.nodes[ntype].data['feat']),8:torch.tensor([],dtype=torch.float16)}
        num_nodes=len(hg.nodes(ntype))
        feat=Feat(num_nodes,mask[ntype],feat,reduction_level,'cpu')
        feat.save(os.path.join(path, f'{ntype}.pkl'))

def load_reductioned_feat(folder_path):
    ret={}
    for filename in os.listdir(folder_path):
        if filename.endswith('.pkl'):  # 筛选 pickle 文件
            file_path = os.path.join(folder_path, filename)
            # try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                ntype=filename.split('.pkl')[0]
                ret[ntype]=data
                print(f"read file: {filename}")
                print(f"content: {data}")
            # except Exception as e:
            #     print(f"{e} occur when reading {filename}")
    return ret

class Feat:
    '''
    class for reducted feat
    input:  feat: {reduction_level[k]:ntype_k_feat}
            mask: {reduction_level[k]:ntype_k_mask}
    init:   cache_maps:{reduction_level[k]:cache_map}, map node id -> feat index
    '''
    def __init__(self, num_nodes,mask,feat,reduction_level, device):
        self.device=device
        self.reduction_level=reduction_level
        # flag for indicating those cached nodes
        self.mask=mask
        print(self.mask)
        # feat
        self.feat=feat
        print(feat)
        # maps node id -> index
        self.cache_maps={}
        for dim in self.reduction_level:
            cache_map=torch.zeros(num_nodes, dtype=torch.long, device=device) - 1
            part_num_nodes=self.feat[dim].shape[0]
            cache_idx = torch.arange(part_num_nodes, dtype=torch.long, device=self.device)
            mask=self.mask[dim]
            cache_map[mask] = cache_idx
            self.cache_maps[dim]=cache_map

    def get(self,node_id):
        with torch.no_grad():
            ret={}
            cache_masks={}
            for dim in self.reduction_level:
                mask=self.mask[dim]
                cache_mask=mask[node_id]
                cached_id=node_id[cache_mask]
                cache_idx=self.cache_maps[dim][cached_id]
                ret[dim]=self.feat[dim][cache_idx]
                cache_masks[dim]=cache_mask
            return ret,cache_masks
    
    # for cache get, return only mask
    # indicating which should use which mask 
    def get_cache_mask(self,node_id):
        with torch.no_grad():
            cache_masks={}
            for dim in self.reduction_level:
                mask=self.mask[dim]
                cache_mask=mask[node_id]
                cache_masks[dim]=cache_mask
            return cache_masks
    
    def save(self,path):
        with open(path,'wb') as f:
            pickle.dump(self,f, protocol=4)

    def shape(self,dim):
        return self.feat[dim].shape

def choose_dimension(x,y):
    a=pow(2,math.ceil(math.log2(x)))
    b=pow(2,math.floor(math.log2(y)))
    if x<=a<=y:
        return a
    elif x<=b<=y:
        return b
    else:
        return pow(2,math.ceil(math.log2(y)))

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
    print(new_paper_feat.shape,new_paper_feat.dtype)
    return new_paper_feat

# if __name__ == "__main__":
#     out_path='preprocess'
#     root ="/datasets/gnn/IGB"
#     dataset='igb-full-small'
#     level_list=[128,8]

#     from load_graph import load_dataset
#     hg, num_classes, predict_category, list_of_metapaths, _ = load_dataset(dataset,root,True,False)
#     print(num_classes,list_of_metapaths)

#     start =time.time()
#     save_path=os.path.join(out_path, dataset)
#     feats=write_reductioned_feat(hg,predict_category,level_list,save_path)
#     print(f"preprocess takes {time.time()-start} sec.")
#     os.makedirs(save_path, exist_ok=True)
#     feats=load_reductioned_feat('/gf3/home/jgqj/test_code/hydro/preprocess/igb-full-medium')

#     for ntype, feat in feats.items():
#         print(ntype,feat)

#     nids={'author':torch.tensor([10,20,30]),
#         'paper':torch.tensor([400000,400001,600000,600001]),
#         'institute':torch.tensor([10,20,30]),
#         'fos':torch.tensor([10,20,30])}
#     for ntype, nid in nids.items():
#         ret,idx=feats[ntype].get(nid)
#         print(ret,idx)
