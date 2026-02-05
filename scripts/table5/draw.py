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

def drgnn_mag(filename,worker, shape_dict=None):
    isDRGNN=False
    if shape_dict==None:
        isDRGNN=True
        if 'ogbn' in filename:
            shape_dict={'author': 256, 'field_of_study': 256, 'institution': 256, 'author_opt': 512, 'field_of_study_opt': 512, 'institution_opt': 512, 'paper': 512}
        else:
            shape_dict={'author': 256, 'institution': 256, 'author_opt': 512, 'institution_opt': 512, 'paper': 1536}
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
    # print(filename)
    # print(cache_read_hit_rate,feature_cnt)

    ret_dict=defaultdict(float)
    for ntype in feature_cnt.keys():
        ret_dict[ntype]=(1-cache_read_hit_rate[ntype])*shape_dict[ntype]*feature_cnt[ntype]/1024/1024/1024
    # print(ret_dict)

    forward_data_vol=sum(ret_dict.values())

    cache_write_hit_rate_values=extract_str_from_file(filename, 'gpu cache write hit rate: ','}')
    embedding_cnt_values=extract_str_from_file(filename, 'emb_update_cnt: ','}')
    forward_times=extract_key_word_from_file(filename, ', feat_copy:')
    backward_times=extract_key_word_from_file(filename, ', emb update:')
    forward_times=numpy.mean(forward_times)
    backward_times=numpy.mean(backward_times)
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
    consistency_dict=defaultdict(float)
    for ntype in embedding_cnt.keys():
        if isDRGNN or '_opt' in ntype:
            ret_dict[ntype]=2*(1-cache_write_hit_rate[ntype])*shape_dict[ntype]*embedding_cnt[ntype]/1024/1024/1024
        else:
            ret_dict[ntype]=(2-cache_write_hit_rate[ntype])*shape_dict[ntype]*embedding_cnt[ntype]/1024/1024/1024
        # all-to-all代价
        if isDRGNN and ('_opt' not in ntype):
            consistency_dict[ntype]=(worker-1)*cache_write_hit_rate[ntype]*shape_dict[ntype]*embedding_cnt[ntype]/1024/1024/1024
    # print(ret_dict)

    hit_write=defaultdict(float)
    hit_read=defaultdict(float)
    for ntype in embedding_cnt.keys():
        hit_write[ntype]=2*cache_write_hit_rate[ntype]*embedding_cnt[ntype]
    for ntype in feature_cnt.keys():
        hit_read[ntype]=cache_read_hit_rate[ntype]*feature_cnt[ntype]
    average_hit_write=sum(hit_write.values())/(2*sum(embedding_cnt.values()))
    average_hit_read=sum(hit_read.values())/sum(feature_cnt.values())
    average_hit=(sum(hit_write.values())+sum(hit_read.values()))/(2*sum(embedding_cnt.values())+sum(feature_cnt.values()))

    backward_data_vol=sum(ret_dict.values())

    # print("forward_data_vol",forward_data_vol,
    #       "backward_data_vol",backward_data_vol)
    if isDRGNN:
        return (forward_data_vol,backward_data_vol,average_hit_read,average_hit_write,forward_times,backward_times,sum(consistency_dict.values()))
    else:
        return (forward_data_vol,backward_data_vol,average_hit_read,average_hit_write,forward_times,backward_times)

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
    forward_times=extract_key_word_from_file(filename, ', feat_copy:')
    backward_times=extract_key_word_from_file(filename, ', emb update:')
    forward_times=numpy.mean(forward_times)
    backward_times=numpy.mean(backward_times)
    return (forward_data_vol,backward_data_vol,0,0,forward_times,backward_times)

def dgl_get_time(filename):
    forward_times=extract_key_word_from_file(filename, ', feat_copy:')
    backward_times=extract_key_word_from_file(filename, ', emb update:')
    forward_times=numpy.mean(forward_times)
    backward_times=numpy.mean(backward_times)
    return (forward_times,backward_times)

def cul():
    root='./log/save'
    dir_list=['RGAT-MAG','RGAT-OM','RGCN-MAG','RGCN-OM']
    name_transfer={
        'dgl':'DGL',
        'heta':'Heta',
        'drgnn':'DRGNN'
    }
    gpu_num=['2', '4']
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
                if name_transfer[name_list[SYSTEM]]=='DRGNN' and 'igb' not in name_list[DATASET]:
                    data_transfer=drgnn_mag(file_path,int(name_list[GPU]))
                    # data_transfer_dgl=dgl_igb(file_path,int(name_list[GPU]))
                elif name_transfer[name_list[SYSTEM]]=='Heta' and 'igb' not in name_list[DATASET]:
                    data_transfer=heta_mag(file_path,int(name_list[GPU]))
                    data_transfer_dgl=dgl_mag(file_path,int(name_list[GPU]))
                    # print("data_transfer_dgl",data_transfer_dgl)
                elif name_transfer[name_list[SYSTEM]]=='DGL' and 'igb' not in name_list[DATASET]:
                    continue
                if data_transfer_dgl!=-1:
                    total_dict[subgraph_title]['DGL'][name_list[GPU]]=data_transfer_dgl
                    # print(f"total_dict[{subgraph_title}][{'DGL'}][{name_list[GPU]}]=data_transfer_dgl",total_dict[subgraph_title]['DGL'][name_list[GPU]])
                total_dict[subgraph_title][name_transfer[name_list[SYSTEM]]][name_list[GPU]]=data_transfer
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
                if name_transfer[name_list[SYSTEM]]=='DGL' and 'igb' not in name_list[DATASET]:
                    data_transfer=total_dict[subgraph_title][name_transfer[name_list[SYSTEM]]][name_list[GPU]]
                    epoch_time=dgl_get_time(file_path)
                    total_dict[subgraph_title][name_transfer[name_list[SYSTEM]]][name_list[GPU]]=data_transfer[:-2]+epoch_time
                    
    
    print(total_dict)
    forward_dict={subgraph_title:{num:{} for num in gpu_num} for subgraph_title in dir_list}
    backward_dict={subgraph_title:{num:{} for num in gpu_num} for subgraph_title in dir_list}
    for subgraph_title in dir_list:
        for gpu in gpu_num:
            forward_dict[subgraph_title][gpu]['DRGNN']=total_dict[subgraph_title]['DRGNN'][gpu][0]/(total_dict[subgraph_title]['DGL'][gpu][0]+total_dict[subgraph_title]['DGL'][gpu][1])*100
            forward_dict[subgraph_title][gpu]['Heta']=total_dict[subgraph_title]['Heta'][gpu][0]/(total_dict[subgraph_title]['DGL'][gpu][0]+total_dict[subgraph_title]['DGL'][gpu][1])*100
            forward_dict[subgraph_title][gpu]['DGL']=total_dict[subgraph_title]['DGL'][gpu][0]/(total_dict[subgraph_title]['DGL'][gpu][0]+total_dict[subgraph_title]['DGL'][gpu][1])*100
            backward_dict[subgraph_title][gpu]['DRGNN']=total_dict[subgraph_title]['DRGNN'][gpu][1]/(total_dict[subgraph_title]['DGL'][gpu][0]+total_dict[subgraph_title]['DGL'][gpu][1])*100
            backward_dict[subgraph_title][gpu]['Heta']=total_dict[subgraph_title]['Heta'][gpu][1]/(total_dict[subgraph_title]['DGL'][gpu][0]+total_dict[subgraph_title]['DGL'][gpu][1])*100
            backward_dict[subgraph_title][gpu]['DGL']=total_dict[subgraph_title]['DGL'][gpu][1]/(total_dict[subgraph_title]['DGL'][gpu][0]+total_dict[subgraph_title]['DGL'][gpu][1])*100
    #         print(subgraph_title,gpu,'DRGNN',forward_dict[subgraph_title][gpu]['DRGNN'],backward_dict[subgraph_title][gpu]['DRGNN'],'Heta',forward_dict[subgraph_title][gpu]['Heta'],backward_dict[subgraph_title][gpu]['Heta'],'DGL',100,100)
            
    # for model in ['RGAT']:
    #     for system in ['DGL','Heta','DRGNN']:
    #         print(f"{system}",end=' & ')
    #         for dataset in ['OM','MAG']:
    #             for gpu in ['2','4']:
    #                 print('{:.1f}+{:.1f}'.format(forward_dict[f"{model}-{dataset}"][gpu][system],backward_dict[f"{model}-{dataset}"][gpu][system]),end=' & ')
    #         print('\\\\')

    # print("data tranfer volume, forward")
    # for model in ['RGAT']:
    #     for system in ['DGL','Heta','DRGNN']:
    #         print(f"{system}",end=' & ')
    #         for dataset in ['OM','MAG']:
    #             for gpu in ['2','4']:
    #                 print('{:.1f}'.format(total_dict[f"{model}-{dataset}"][system][gpu][0]),end=' & ')
    #         print('\\\\')

    # print("data tranfer volume, backward")
    # for model in ['RGAT']:
    #     for system in ['DGL','Heta','DRGNN']:
    #         print(f"{system}",end=' & ')
    #         for dataset in ['OM','MAG']:
    #             for gpu in ['2','4']:
    #                 print('{:.1f}'.format(total_dict[f"{model}-{dataset}"][system][gpu][1]),end=' & ')
    #         print('\\\\')

    # print("data tranfer volume, total")
    # for model in ['RGAT']:
    #     for system in ['DGL','Heta','DRGNN']:
    #         print(f"{system}",end=' & ')
    #         for dataset in ['OM','MAG']:
    #             for gpu in ['2','4']:
    #                 print('{:.1f}'.format(total_dict[f"{model}-{dataset}"][system][gpu][1]+total_dict[f"{model}-{dataset}"][system][gpu][0]),end=' & ')
    #         print('\\\\')

    # print("cache hit rate, forward")
    # for model in ['RGAT']:
    #     for system in ['DGL','Heta','DRGNN']:
    #         print(f"{system}",end=' & ')
    #         for dataset in ['OM','MAG']:
    #             for gpu in ['2','4']:
    #                 print('{:.1f}\%'.format(total_dict[f"{model}-{dataset}"][system][gpu][2]*100),end=' & ')
    #         print('\\\\')

    # print("cache hit rate, backward")
    # for model in ['RGAT']:
    #     for system in ['DGL','Heta','DRGNN']:
    #         print(f"{system}",end=' & ')
    #         for dataset in ['OM','MAG']:
    #             for gpu in ['2','4']:
    #                 print('{:.1f}\%'.format(total_dict[f"{model}-{dataset}"][system][gpu][3]*100),end=' & ')
    #         print('\\\\')

    # print("all-to-all cost")
    # for model in ['RGAT']:
    #     for system in ['DRGNN']:
    #         print(f"{system}",end=' & ')
    #         for dataset in ['OM','MAG']:
    #             for gpu in ['2','4']:
    #                 print('{:.1f}'.format(total_dict[f"{model}-{dataset}"][system][gpu][-1]),end=' & ')
    #         print('\\\\')
    
    # print("ori_table:")
    # for dataset in ['OM','MAG']:
    #     for system in ['DGL','Heta','DRGNN']:
    #         for gpu in ['2','4']:
    #             if gpu =='4':
    #                 print(f" & {gpu}",end=' & ')
    #             else:
    #                 if system=='DGL':
    #                     print(f"{system} & {gpu}",end=' & ')
    #                 elif system=='DRGNN':
    #                     print("\multirow{2}{*}{DGL+FC}", f"& {gpu}",end=' & ')
    #                 else:
    #                     print("\multirow{2}{*}{"+system+"}", f"& {gpu}",end=' & ')
    #             print('{:.0f} & {:.0f}\% & {:.0f} & {:.0f}\%  & {:.0f}\%'.format(
    #                 total_dict[f"{model}-{dataset}"][system][gpu][0],
    #                 total_dict[f"{model}-{dataset}"][system][gpu][2]*100,
    #                 total_dict[f"{model}-{dataset}"][system][gpu][1],
    #                 total_dict[f"{model}-{dataset}"][system][gpu][3]*100,
    #                 -1 if system=='DRGNN' else 100-100*(total_dict[f"{model}-{dataset}"]['DRGNN'][gpu][0]+total_dict[f"{model}-{dataset}"]['DRGNN'][gpu][1])/(total_dict[f"{model}-{dataset}"][system][gpu][0]+ total_dict[f"{model}-{dataset}"][system][gpu][1])
    #                 ).replace('-1\%','-').replace('-1','-'),end=' ')
    #             print('\\\\')
    #     print()

    # print("new_table:")
    # for dataset in ['OM','MAG']:
    #     for system in ['DGL','Heta','DRGNN']:
    #         for gpu in ['2','4']:
    #             if gpu =='4':
    #                 print(f" & {gpu}",end=' & ')
    #             else:
    #                 if system=='DGL':
    #                     print(f"{system} & {gpu}",end=' & ')
    #                 elif system=='DRGNN':
    #                     print("\multirow{2}{*}{DGL+FC}", f"& {gpu}",end=' & ')
    #                 else:
    #                     print("\multirow{2}{*}{"+system+"}", f"& {gpu}",end=' & ')
    #             print('{:.0f} & {:.0f} & {:.0f} & {:.0f}  & {:.0f}\%'.format(
    #                 total_dict[f"{model}-{dataset}"][system][gpu][0],
    #                 total_dict[f"{model}-{dataset}"][system][gpu][4],
    #                 total_dict[f"{model}-{dataset}"][system][gpu][1],
    #                 total_dict[f"{model}-{dataset}"][system][gpu][5],
    #                 -1 if system=='DRGNN' else 100-100*(total_dict[f"{model}-{dataset}"]['DRGNN'][gpu][0]+total_dict[f"{model}-{dataset}"]['DRGNN'][gpu][1])/(total_dict[f"{model}-{dataset}"][system][gpu][0]+ total_dict[f"{model}-{dataset}"][system][gpu][1])
    #                 ).replace('-1\%','-').replace('-1','-'),end=' ')
    #             print('\\\\')
    #     print()

    for model in ['RGAT','RGCN']:
        print(model)
        for dataset in ['OM','MAG']:
            for gpu in ['2','4']:
                for system in ['DGL','Heta','DRGNN']:
                    if system =='DGL':
                        print("\multirow{3}{*}{"+gpu+"}", f"& {system}",end=' & ')
                    else:
                        if system=='DRGNN':
                            print(f" & DGL+FC",end=' & ')
                        else:
                            print(f" & {system}",end=' & ')
                    print('{:.0f} & {:.0f} & {:.0f}\% & {:.0f} & {:.0f} & {:.0f}\%'.format(
                        total_dict[f"{model}-{dataset}"][system][gpu][4],
                        total_dict[f"{model}-{dataset}"][system][gpu][5],
                        -1 if system=='DRGNN' else 100-100*(total_dict[f"{model}-{dataset}"]['DRGNN'][gpu][4]+total_dict[f"{model}-{dataset}"]['DRGNN'][gpu][5])/(total_dict[f"{model}-{dataset}"][system][gpu][4]+ total_dict[f"{model}-{dataset}"][system][gpu][5]),
                        total_dict[f"{model}-{dataset}"][system][gpu][0],
                        total_dict[f"{model}-{dataset}"][system][gpu][1],
                        -1 if system=='DRGNN' else 100-100*(total_dict[f"{model}-{dataset}"]['DRGNN'][gpu][0]+total_dict[f"{model}-{dataset}"]['DRGNN'][gpu][1])/(total_dict[f"{model}-{dataset}"][system][gpu][0]+ total_dict[f"{model}-{dataset}"][system][gpu][1])
                        ).replace('-1\%','-').replace('-1','-'),end=' ')
                    print('\\\\')
                print("\\midrule")
            print()

if __name__ == '__main__':
    # a=dgl_igb("/gf3/home/jgqj/test_code/hydro/baseline/dgl/log/backup_nccl/dgl_RGAT_igb-full-small_none_64_4.log",4)
    # print(a)
    # a=drgnn_igb("/gf3/home/jgqj/test_code/hydro/exp/exp5/log/new/DRGNN/drgnn_RGAT_igb-full-small_miss_penalty_64_4.log",4)
    # print(a)
    # a=heta_igb("/gf3/home/jgqj/test_code/hydro/baseline/heta/log/new_cul/heta_RGAT_igb-full-small_miss_penalty_64_4.log",4)
    # print(a)

    # a=dgl_mag("/gf3/home/jgqj/test_code/hydro/exp/exp5/log/new/all/RGAT-OM/heta_RGAT_ogbn-mag_miss_penalty_64_4.log",4)
    # print(a)
    # a=drgnn_mag("/gf3/home/jgqj/test_code/hydro/exp/exp5/log/new/all/RGAT-OM/drgnn_RGAT_ogbn-mag_miss_penalty_64_4.log",4)
    # print(a)
    # a=heta_mag("/gf3/home/jgqj/test_code/hydro/exp/exp5/log/new/all/RGAT-OM/heta_RGAT_ogbn-mag_miss_penalty_64_4.log",4)
    # print(a)
    cul()