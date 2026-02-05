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
    cache_hit_rate={ntype:cnt/len(total_cnt_values) for ntype,cnt in cache_hit_rate.items()}
    total_cnt={ntype:cnt/len(total_cnt_values)*worker for ntype,cnt in total_cnt.items()}
    # print(cache_hit_rate,total_cnt)

    ret_dict=defaultdict(float)
    for ntype in total_cnt.keys():
        # ret_dict[ntype]=(1-cache_hit_rate[ntype])*shape_dict[ntype]*total_cnt[ntype]/1024/1024/1024
        ret_dict[ntype]=shape_dict[ntype]*total_cnt[ntype]/1024/1024/1024
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

root='./all'
dir_list=['RGCN-ME','RGCN-SM','RGCN-LA',
            'RGAT-ME','RGAT-SM','RGAT-LA']
name_transfer={
    'dgl':'DGL',
    'heta':'Heta',
    'drgnn':'DRGNN'
}
gpu_num=['1', '2', '4']

def cul():
    total_dict={subgraph_title:{system:{} for system in name_transfer.values()} for subgraph_title in dir_list}
    total_training_dict={subgraph_title:{system:{} for system in name_transfer.values()} for subgraph_title in dir_list}
    total_epoch_dict={subgraph_title:{system:{} for system in name_transfer.values()} for subgraph_title in dir_list}
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
                epoch_values = extract_key_word_from_file(file_path, 'Epoch Time(s): ')
                sample_values = extract_key_word_from_file(file_path, 'sample: ')
                featcopy_values = extract_key_word_from_file(file_path, 'feat_copy: ')
                forward_values = extract_key_word_from_file(file_path, 'forward: ')
                backward_values = extract_key_word_from_file(file_path, 'backward: ')
                embupdate_values = extract_key_word_from_file(file_path, 'emb update: ')

                # 计算平均值
                epoch_mean = numpy.mean(epoch_values) if epoch_values else 0
                featcopy_mean = numpy.mean(featcopy_values) if featcopy_values else 0
                sample_mean = numpy.mean(sample_values) if sample_values else 0
                forward_mean = numpy.mean(forward_values) if forward_values else 0
                backward_mean = numpy.mean(backward_values) if backward_values else 0
                embupdate_mean = numpy.mean(embupdate_values) if embupdate_values else 0
                total_epoch_dict[subgraph_title][name_transfer[name_list[SYSTEM]]][name_list[GPU]]=epoch_mean-sample_mean
                total_training_dict[subgraph_title][name_transfer[name_list[SYSTEM]]][name_list[GPU]]=forward_mean+backward_mean
                data_transfer=-1
                data_transfer_dgl=-1
                if name_transfer[name_list[SYSTEM]]=='DRGNN' and 'igb' in name_list[DATASET]:
                    data_transfer=drgnn_igb(file_path,int(name_list[GPU]))
                elif name_transfer[name_list[SYSTEM]]=='Heta' and 'igb' in name_list[DATASET]:
                    data_transfer=heta_igb(file_path,int(name_list[GPU]))
                    # data_transfer_dgl=dgl_igb(file_path,int(name_list[GPU]))
                elif name_transfer[name_list[SYSTEM]]=='DGL' and 'igb' in name_list[DATASET]:
                    # print(file_path)
                    data_transfer=dgl_igb(file_path,int(name_list[GPU]))
                total_dict[subgraph_title][name_transfer[name_list[SYSTEM]]][name_list[GPU]]=data_transfer
    # total_dict['RGCN-LA']['DGL']={'1': 1, '2': 1, '4': 1}
    # total_dict['RGAT-LA']['DGL']={'1': 1, '2': 1, '4': 1}
    total_dict['RGCN-LA']['Heta']={'1': 1, '2': 1, '4': 1}
    total_dict['RGAT-LA']['Heta']={'1': 1, '2': 1, '4': 1}
    total_training_dict['RGCN-LA']['Heta']={'1': 1, '2': 1, '4': 1}
    total_training_dict['RGAT-LA']['Heta']={'1': 1, '2': 1, '4': 1}
    total_epoch_dict['RGCN-LA']['Heta']={'1': 1, '2': 1, '4': 1}
    total_epoch_dict['RGAT-LA']['Heta']={'1': 1, '2': 1, '4': 1}
    print(total_dict)
    speedup_dict={subgraph_title:{num:{} for num in gpu_num} for subgraph_title in dir_list}
    _min=100
    _max=0
    for subgraph_title in dir_list:
        for gpu in gpu_num:
            speed_up_drgnn=total_dict[subgraph_title]['DRGNN'][gpu]/total_dict[subgraph_title]['DGL'][gpu]*100
            speed_up_heta=total_dict[subgraph_title]['Heta'][gpu]/total_dict[subgraph_title]['DGL'][gpu]*100
            # if _max < max(speed_up_drgnn,speed_up_heta):
            #     _max=max(speed_up_drgnn,speed_up_heta)
            # if _min >min(speed_up_drgnn,speed_up_heta):
            #     _min=min(speed_up_drgnn,speed_up_heta)
            speedup_dict[subgraph_title][gpu]['DRGNN']=speed_up_drgnn
            speedup_dict[subgraph_title][gpu]['Heta']=speed_up_heta
            speedup_dict[subgraph_title][gpu]['DGL']=100
            print(subgraph_title,gpu,'DRGNN',speed_up_drgnn,'Heta',speed_up_heta,'DGL',100)

    print("speedup:")
    for model in ['RGCN','RGAT']:
        for system in ['DGL','Heta','DRGNN']:
            print(f"{model}, {system}",end=' & ')
            for dataset in ['SM','ME','LA']:
                for gpu in ['1','2','4']:
                    # print('{:.1f}'.format(total_dict[f"{model}-{dataset}"][system][gpu]),end=' & ')
                    print('{:.2f}'.format(speedup_dict[f"{model}-{dataset}"][gpu][system]),end=' & ')
            print('\\\\')
    
    print("total:")
    for model in ['RGCN','RGAT']:
        for system in ['DGL','Heta','DRGNN']:
            print(f"{model}, {system}",end=' & ')
            for dataset in ['SM','ME','LA']:
                for gpu in ['1','2','4']:
                    print('{:.0f}'.format(total_dict[f"{model}-{dataset}"][system][gpu]),end=' & ')
                    # print('{:.2f}'.format(speedup_dict[f"{model}-{dataset}"][gpu][system]),end=' & ')
            print('\\\\')
    return total_dict,total_training_dict,total_epoch_dict

def draw_another():
    print("another")
    folder_path='./DGL+MR'
    for filename in os.listdir(folder_path):
        # print(filename)
        if filename.endswith('.log'):
            name_list=split_name(filename)
            if name_list[GPU] !='4':
                # print(filename,name_list,name_list[GPU])
                continue
            file_path = os.path.join(folder_path, filename)
            epoch_values = extract_key_word_from_file(file_path, 'Epoch Time(s): ')
            sample_values = extract_key_word_from_file(file_path, 'sample: ')
            featcopy_values = extract_key_word_from_file(file_path, 'feat_copy: ')
            forward_values = extract_key_word_from_file(file_path, 'forward: ')
            backward_values = extract_key_word_from_file(file_path, 'backward: ')
            embupdate_values = extract_key_word_from_file(file_path, 'emb update: ')

            # 计算平均值
            epoch_mean = numpy.mean(epoch_values) if epoch_values else 0
            featcopy_mean = numpy.mean(featcopy_values) if featcopy_values else 0
            sample_mean = numpy.mean(sample_values) if sample_values else 0
            forward_mean = numpy.mean(forward_values) if forward_values else 0
            backward_mean = numpy.mean(backward_values) if backward_values else 0
            embupdate_mean = numpy.mean(embupdate_values) if embupdate_values else 0
            print(filename,"train:",forward_mean+backward_mean,"total:",epoch_mean-sample_mean)

def draw(total_dict,total_training_dict,total_epoch_dict):
    # print comm. volume
    print("total:")
    for model in ['RGCN']:
        for system in ['DGL','Heta','DRGNN']:
            print(f"{model}, {system}",end=' & ')
            for dataset in ['SM','ME','LA']:
                for gpu in ['4']:
                    print('{:.0f}'.format(total_dict[f"{model}-{dataset}"][system][gpu]),end=' & ')
                    # print('{:.2f}'.format(speedup_dict[f"{model}-{dataset}"][gpu][system]),end=' & ')
            print('\\\\')
    
    # print training time, only dgl since incorrect in heta logging system
    print("forward+backward:")
    for model in ['RGCN']:
        for system in ['DGL','Heta','DRGNN']:
            print(f"{model}, {system}",end=' & ')
            for dataset in ['SM','ME','LA']:
                for gpu in ['4']:
                    print('{:.0f}'.format(total_training_dict[f"{model}-{dataset}"][system][gpu]),end=' & ')
                    # print('{:.2f}'.format(speedup_dict[f"{model}-{dataset}"][gpu][system]),end=' & ')
            print('\\\\')
    
    print("epoch time-sample time:")
    for model in ['RGCN']:
        for system in ['DGL','Heta','DRGNN']:
            print(f"{model}, {system}",end=' & ')
            for dataset in ['SM','ME','LA']:
                for gpu in ['4']:
                    print('{:.0f}'.format(total_epoch_dict[f"{model}-{dataset}"][system][gpu]),end=' & ')
                    # print('{:.2f}'.format(speedup_dict[f"{model}-{dataset}"][gpu][system]),end=' & ')
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
    # total_dict,total_training_dict,total_epoch_dict=cul()
    # draw(total_dict,total_training_dict,total_epoch_dict)
    draw_another()