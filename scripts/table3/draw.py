import numpy
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter,FuncFormatter
import math
from collections import defaultdict
project_root = '/gf3/home/jgqj/test_code/hydro'
sys.path.append(project_root)

from src import extract_key_word_from_file,extract_str_from_file,draw_color,color_list

SYSTEM=0
MODEL=1
DATASET=2
GPU=4

def split_name(file_name):
    name=file_name.split('.log')[0]
    l=name.split('_')
    if l[3]=='miss' and l[4]=='penalty':
        return {SYSTEM:l[0],MODEL:l[1],DATASET:l[2],GPU:l[6]}
    else:
        return {SYSTEM:l[0],MODEL:l[1],DATASET:l[2],GPU:l[5]}

def evall(l):
    return [eval(i) for i in l]

# for drgnn with miss_penalty only
def drgnn_igb(filename,worker):
    shape_dict={'author': 32, 'conference': 32, 'fos': 32, 'institute': 32, 'journal': 32, 'paper': 512}
    cache_hit_rate_values=extract_str_from_file(filename, 'gpu cache read hit rate: ','}')
    total_cnt_values=extract_str_from_file(filename, 'feature_retrieval_cnt: ','}')
    cache_hit_rate_values =evall(cache_hit_rate_values)
    total_cnt_values =evall(total_cnt_values)
    # print(cache_hit_rate_values,total_cnt_values)

    cache_hit_rate=defaultdict(float)
    total_cnt=defaultdict(int)
    for i in cache_hit_rate_values:
        for ntype,hit_rate in i.items():
            cache_hit_rate[ntype]+=float(hit_rate)/100
    for i in total_cnt_values:
        for ntype,cnt in i.items():
            total_cnt[ntype]+=int(cnt)
    
    # 平均
    cache_hit_rate={ntype:cnt/len(total_cnt_values) for ntype,cnt in cache_hit_rate.items()}
    total_cnt={ntype:cnt/len(total_cnt_values)*worker for ntype,cnt in total_cnt.items()}
    # print(cache_hit_rate,total_cnt)

    ret_dict=defaultdict(float)
    for ntype in total_cnt.keys():
        ret_dict[ntype]=(1-cache_hit_rate[ntype])*shape_dict[ntype]*total_cnt[ntype]/1024/1024/1024
    # print(ret_dict)

    return sum(ret_dict.values())

def heta_igb(filename,worker):
    shape_dict=4096
    cache_hit_rate_values=extract_str_from_file(filename, 'gpu cache hit rate: ','}')
    total_cnt_values=extract_str_from_file(filename, 'feature_retrieval_cnt: ','}')
    cache_hit_rate_values =[eval(i) for i in cache_hit_rate_values]
    total_cnt_values =[eval(i) for i in total_cnt_values]
    # print(cache_hit_rate_values,total_cnt_values)

    cache_hit_rate=defaultdict(float)
    total_cnt=defaultdict(int)
    for i in cache_hit_rate_values:
        for ntype,hit_rate in i.items():
            cache_hit_rate[ntype]+=float(hit_rate)
    for i in total_cnt_values:
        for ntype,cnt in i.items():
            total_cnt[ntype]+=int(cnt)
    cache_hit_rate={ntype:cnt/len(cache_hit_rate_values) for ntype,cnt in cache_hit_rate.items()}
    total_cnt={ntype:cnt/len(total_cnt_values)*worker for ntype,cnt in total_cnt.items()}
    # print(cache_hit_rate,total_cnt)

    ret_dict=defaultdict(float)
    for ntype in total_cnt.keys():
        ret_dict[ntype]=(1-cache_hit_rate[ntype])*shape_dict*total_cnt[ntype]/1024/1024/1024
    # print(ret_dict)

    return sum(ret_dict.values())

def dgl_igb(filename,worker):
    shape_dict=4096
    total_cnt_values=extract_str_from_file(filename, ', #inputs:')
    # print(filename,total_cnt_values)
    total_cnt_values=[int(i.split(',')[0]) for i in total_cnt_values]
    # print(total_cnt_values)

    total_cnt=numpy.mean(total_cnt_values)*worker
    # print(total_cnt)

    return total_cnt*shape_dict/1024/1024/1024

def drgnn_mag(filename,worker, shape_dict=None):
    isDRGNN=False
    if shape_dict==None:
        isDRGNN=True
        if 'ogbn' in filename:
            shape_dict={'author': 256, 'field_of_study': 256, 'institution': 256, 'author_opt': 512, 'field_of_study_opt': 512, 'institution_opt': 512, 'paper': 128}
        else:
            shape_dict={'author': 256, 'institution': 256, 'author_opt': 512, 'institution_opt': 512, 'paper': 512}
    cache_read_hit_rate_values=extract_str_from_file(filename, 'gpu cache read hit rate: ','}')
    feature_cnt_values=extract_str_from_file(filename, 'feature_retrieval_cnt: ','}')

    cache_read_hit_rate_values =evall(cache_read_hit_rate_values)
    feature_cnt_values =evall(feature_cnt_values)
    # print(len(cache_read_hit_rate_values),'\n',len(feature_cnt_values))

    cache_read_hit_rate=defaultdict(float)
    feature_cnt=defaultdict(int)
    for i in cache_read_hit_rate_values:
        for ntype,hit_rate in i.items():
            if isDRGNN:
                cache_read_hit_rate[ntype]+=float(hit_rate)/100
            else:
                cache_read_hit_rate[ntype]+=float(hit_rate)
    for i in feature_cnt_values:
        for ntype,cnt in i.items():
            feature_cnt[ntype]+=int(cnt)
    cache_read_hit_rate={ntype:cnt/len(cache_read_hit_rate_values) for ntype,cnt in cache_read_hit_rate.items()}
    feature_cnt={ntype:cnt/len(feature_cnt_values)*worker for ntype,cnt in feature_cnt.items()}
    # print(cache_hit_rate,total_cnt)

    ret_dict=defaultdict(float)
    for ntype in feature_cnt.keys():
        ret_dict[ntype]=(1-cache_read_hit_rate[ntype])*shape_dict[ntype]*feature_cnt[ntype]/1024/1024/1024
    # print(ret_dict)

    forward_data_vol=sum(ret_dict.values())

    cache_write_hit_rate_values=extract_str_from_file(filename, 'gpu cache write hit rate: ','}')
    embedding_cnt_values=extract_str_from_file(filename, 'emb_update_cnt: ','}')

    cache_write_hit_rate_values =evall(cache_write_hit_rate_values)
    embedding_cnt_values =evall(embedding_cnt_values)

    cache_write_hit_rate=defaultdict(float)
    embedding_cnt=defaultdict(int)
    for i in cache_write_hit_rate_values:
        for ntype,hit_rate in i.items():
            cache_write_hit_rate[ntype]+=float(hit_rate)/100
    for i in embedding_cnt_values:
        for ntype,cnt in i.items():
            embedding_cnt[ntype]+=int(cnt)
    cache_write_hit_rate={ntype:cnt/len(cache_write_hit_rate_values) for ntype,cnt in cache_write_hit_rate.items()}
    embedding_cnt={ntype.split('node_emb_')[-1]:cnt/len(embedding_cnt_values)*worker for ntype,cnt in embedding_cnt.items()}
    l=list(embedding_cnt.keys())
    for ntype in l:
        embedding_cnt[f"{ntype}_opt"]=embedding_cnt[ntype]
    # print(cache_write_hit_rate,embedding_cnt)

    ret_dict=defaultdict(float)
    for ntype in embedding_cnt.keys():
        if isDRGNN or '_opt' in ntype:
            ret_dict[ntype]=2*(1-cache_write_hit_rate[ntype])*shape_dict[ntype]*embedding_cnt[ntype]/1024/1024/1024
        else:
            ret_dict[ntype]=(2-cache_write_hit_rate[ntype])*shape_dict[ntype]*embedding_cnt[ntype]/1024/1024/1024
    # print(ret_dict)

    backward_data_vol=sum(ret_dict.values())

    print("forward_data_vol",forward_data_vol,
          "backward_data_vol",backward_data_vol)
    
    return forward_data_vol+backward_data_vol

def heta_mag(filename,worker):
    if 'ogbn' in filename:
        shape_dict={'author': 256, 'field_of_study': 256, 'institution': 256, 'author_opt': 512, 'field_of_study_opt': 512, 'institution_opt': 512, 'paper': 512}
    else:
        shape_dict={'author': 256, 'institution': 256, 'author_opt': 512, 'institution_opt': 512, 'paper': 1536}
    return drgnn_mag(filename,worker,shape_dict)

def dgl_mag(filename,worker):
    if 'ogbn' in filename:
        shape_dict={'author': 256, 'field_of_study': 256, 'institution': 256, 'author_opt': 512, 'field_of_study_opt': 512, 'institution_opt': 512, 'paper': 512}
    else:
        shape_dict={'author': 256, 'institution': 256, 'author_opt': 512, 'institution_opt': 512, 'paper': 1536}
    feature_cnt_values=extract_str_from_file(filename, 'feature_retrieval_cnt: ','}')

    feature_cnt_values =evall(feature_cnt_values)

    feature_cnt=defaultdict(int)
    for i in feature_cnt_values:
        for ntype,cnt in i.items():
            feature_cnt[ntype]+=int(cnt)
    feature_cnt={ntype:cnt/len(feature_cnt_values)*worker for ntype,cnt in feature_cnt.items()}
    # print(cache_hit_rate,total_cnt)

    ret_dict=defaultdict(float)
    for ntype in feature_cnt.keys():
        ret_dict[ntype]=shape_dict[ntype]*feature_cnt[ntype]/1024/1024/1024
    # print(ret_dict)

    forward_data_vol=sum(ret_dict.values())

    embedding_cnt_values=extract_str_from_file(filename, 'emb_update_cnt: ','}')

    embedding_cnt_values =evall(embedding_cnt_values)

    embedding_cnt=defaultdict(int)
    for i in embedding_cnt_values:
        for ntype,cnt in i.items():
            embedding_cnt[ntype]+=int(cnt)
    embedding_cnt={ntype.split('node_emb_')[-1]:cnt/len(embedding_cnt_values)*worker for ntype,cnt in embedding_cnt.items()}
    l=list(embedding_cnt.keys())
    for ntype in l:
        embedding_cnt[f"{ntype}_opt"]=embedding_cnt[ntype]
    # print(cache_write_hit_rate,embedding_cnt)

    ret_dict=defaultdict(float)
    for ntype in embedding_cnt.keys():
        ret_dict[ntype]=2*shape_dict[ntype]*embedding_cnt[ntype]/1024/1024/1024
    # print(ret_dict)

    backward_data_vol=sum(ret_dict.values())

    print("forward_data_vol",forward_data_vol,
          "backward_data_vol",backward_data_vol)
    
    return forward_data_vol+backward_data_vol

def cul():
    root='./log/all'
    dir_list=['RGCN-MAG','RGCN-ME','RGCN-OM','RGCN-SM','RGCN-LA',
              'RGAT-MAG','RGAT-ME','RGAT-OM','RGAT-SM','RGAT-LA']
    # dir_list=['RGAT-ME','RGAT-OM','RGAT-SM','RGAT-LA',
    #           'RGCN-ME','RGCN-OM','RGCN-SM','RGCN-LA']
    name_transfer={
        'dgl':'DGL',
        'heta':'Heta',
        'drgnn':'DRGNN'
    }
    gpu_num=['1', '2', '4']
    total_dict={subgraph_title:{system:{} for system in name_transfer.values()} for subgraph_title in dir_list}
    # print(total_dict)
    for subgraph_title in dir_list:
        folder_path=os.path.join(root,subgraph_title)
        for filename in os.listdir(folder_path):
            if filename.endswith('.log'):
                name_list=split_name(filename)
                if name_list[GPU] not in gpu_num:
                    # print(filename,name_list,name_list[GPU])
                    continue
                file_path = os.path.join(folder_path, filename)
                # epoch_values=extract_key_word_from_file(file_path, ', feat_copy: ')
                # epoch_time=numpy.mean(epoch_values)
                data_transfer=-1
                data_transfer_dgl=-1
                if name_transfer[name_list[SYSTEM]]=='DRGNN' and 'igb' in name_list[DATASET]:
                    data_transfer=drgnn_igb(file_path,int(name_list[GPU]))
                elif name_transfer[name_list[SYSTEM]]=='DRGNN' and 'igb' not in name_list[DATASET]:
                    data_transfer=drgnn_mag(file_path,int(name_list[GPU]))
                elif name_transfer[name_list[SYSTEM]]=='Heta' and 'igb' in name_list[DATASET]:
                    data_transfer=heta_igb(file_path,int(name_list[GPU]))
                    # data_transfer_dgl=dgl_igb(file_path,int(name_list[GPU]))
                elif name_transfer[name_list[SYSTEM]]=='Heta' and 'igb' not in name_list[DATASET]:
                    data_transfer=heta_mag(file_path,int(name_list[GPU]))
                    data_transfer_dgl=dgl_mag(file_path,int(name_list[GPU]))
                    # print("data_transfer_dgl",data_transfer_dgl)
                elif name_transfer[name_list[SYSTEM]]=='DGL' and 'igb' in name_list[DATASET]:
                    # print(file_path)
                    data_transfer=dgl_igb(file_path,int(name_list[GPU]))
                elif name_transfer[name_list[SYSTEM]]=='DGL' and 'igb' not in name_list[DATASET]:
                    # data_transfer=dgl_mag(file_path,int(name_list[GPU]))
                    continue
                if data_transfer_dgl!=-1:
                    total_dict[subgraph_title]['DGL'][name_list[GPU]]=data_transfer_dgl
                    # print(f"total_dict[{subgraph_title}][{'DGL'}][{name_list[GPU]}]=data_transfer_dgl",total_dict[subgraph_title]['DGL'][name_list[GPU]])
                total_dict[subgraph_title][name_transfer[name_list[SYSTEM]]][name_list[GPU]]=data_transfer
    # total_dict['RGCN-LA']['DGL']={'1': 1, '2': 1, '4': 1}
    # total_dict['RGAT-LA']['DGL']={'1': 1, '2': 1, '4': 1}
    total_dict['RGCN-LA']['Heta']={'1': 1, '2': 1, '4': 1}
    total_dict['RGAT-LA']['Heta']={'1': 1, '2': 1, '4': 1}
    print(total_dict)
    speedup_dict={subgraph_title:{num:{} for num in gpu_num} for subgraph_title in dir_list}
    _min=100
    _max=0
    total_volumn_dgl=[]
    total_volumn_heta=[]
    total_volumn_drgnn=[]
    for subgraph_title in dir_list:
        for gpu in gpu_num:
            speed_up_drgnn=total_dict[subgraph_title]['DRGNN'][gpu]/total_dict[subgraph_title]['DGL'][gpu]*100
            speed_up_heta=total_dict[subgraph_title]['Heta'][gpu]/total_dict[subgraph_title]['DGL'][gpu]*100
            # if _max < max(speed_up_drgnn,speed_up_heta):
            #     _max=max(speed_up_drgnn,speed_up_heta)
            # if _min >min(speed_up_drgnn,speed_up_heta):
            #     _min=min(speed_up_drgnn,speed_up_heta)
            speed_up_against_dgl=total_dict[subgraph_title]['DRGNN'][gpu]/total_dict[subgraph_title]['DGL'][gpu]*100
            speed_up_against_heta=total_dict[subgraph_title]['DRGNN'][gpu]/total_dict[subgraph_title]['Heta'][gpu]*100
            speedup_dict[subgraph_title][gpu]['DRGNN']=speed_up_drgnn
            speedup_dict[subgraph_title][gpu]['Heta']=speed_up_heta
            speedup_dict[subgraph_title][gpu]['DGL']=100 # total_dict[subgraph_title]['DGL'][gpu]
            # total_volumn_dgl.append(total_dict[subgraph_title]['DGL'][gpu])
            # total_volumn_heta.append(total_dict[subgraph_title]['Heta'][gpu])
            # total_volumn_drgnn.append(total_dict[subgraph_title]['DRGNN'][gpu])
            if 'LA' not in subgraph_title:
                total_volumn_dgl.append(speed_up_against_dgl)
                total_volumn_heta.append(speed_up_against_heta)
            print(subgraph_title,gpu,'DGL',speed_up_against_dgl,'Heta',speed_up_against_heta)
    print('average transfer volumn','DGL:',numpy.mean(total_volumn_dgl),'Heta:',numpy.mean(total_volumn_heta),'total',numpy.mean(total_volumn_dgl+total_volumn_heta))
    for model in ['RGCN','RGAT']:
        for system in ['DGL','Heta','DRGNN']:
            print(f"{model}, {system}",end=' & ')
            for dataset in ['OM','MAG','SM','ME','LA']:
                for gpu in ['1','2','4']:
                    print('{:.2f}'.format(speedup_dict[f"{model}-{dataset}"][gpu][system]),end=' & ')
            print('\\\\')
    
    print("total PCIe-GPU Comm.")
    for model in ['RGCN','RGAT']:
        print(model)
        for system in ['DGL','Heta','DRGNN']:
            print(f"{model}, {system}",end=' & ')
            for dataset in ['OM','MAG','SM','ME','LA']:
                for gpu in ['1','2','4']:
                    print('{:.0f}'.format(total_dict[f"{model}-{dataset}"][system][gpu]),end=' & ')
            print('\\\\')
    
if __name__ == '__main__':
    # a=dgl_igb("/gf3/home/jgqj/test_code/hydro/baseline/dgl/log/backup_nccl/dgl_rgat_igb-full-small_none_64_4.log",4)
    # print(a)
    # a=drgnn_igb("/gf3/home/jgqj/test_code/hydro/exp/exp5/log/new/DRGNN/drgnn_rgat_igb-full-small_miss_penalty_64_4.log",4)
    # print(a)
    # a=heta_igb("/gf3/home/jgqj/test_code/hydro/baseline/heta/log/new_cul/heta_rgcn_igb-full-small_miss_penalty_64_4.log",4)
    # print(a)

    # a=dgl_mag("/gf3/home/jgqj/test_code/hydro/exp/exp5/log/new/all/RGAT-OM/heta_rgat_ogbn-mag_miss_penalty_64_4.log",4)
    # print(a)
    # a=drgnn_mag("/gf3/home/jgqj/test_code/hydro/exp/exp5/log/new/all/RGAT-OM/drgnn_rgat_ogbn-mag_miss_penalty_64_4.log",4)
    # print(a)
    # a=heta_mag("/gf3/home/jgqj/test_code/hydro/exp/exp5/log/new/all/RGAT-OM/heta_rgat_ogbn-mag_miss_penalty_64_4.log",4)
    # print(a)
    cul()