import numpy
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter,FuncFormatter
import math
from collections import defaultdict
project_root = '/gf3/home/jgqj/test_code/hydro'
sys.path.append(project_root)

from src.utils import extract_key_word_from_file,extract_str_from_file,color_list


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
    isMeCache=False
    if shape_dict==None:
        isMeCache=True
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
            if isMeCache:
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

    opt_dict=defaultdict(float)
    emb_dict=defaultdict(float)
    consistency_dict=defaultdict(float)
    for ntype in embedding_cnt.keys():
        # opt cost is the same in heta and drgnn
        if '_opt' in ntype:
            opt_dict[ntype]=2*(1-cache_write_hit_rate[ntype])*shape_dict[ntype]*embedding_cnt[ntype]/1024/1024/1024
        else:
            # emb cost and consistency cost
            if isMeCache:
                emb_dict[ntype]=2*(1-cache_write_hit_rate[ntype])*shape_dict[ntype]*embedding_cnt[ntype]/1024/1024/1024
                consistency_dict[ntype]=(worker-1)*cache_write_hit_rate[ntype]*shape_dict[ntype]*embedding_cnt[ntype]/1024/1024/1024
            else:
                emb_dict[ntype]=(2-cache_write_hit_rate[ntype])*shape_dict[ntype]*embedding_cnt[ntype]/1024/1024/1024
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

    backward_data_vol={"emb":sum(emb_dict.values()),"opt":sum(opt_dict.values()),"a2a":sum(consistency_dict.values())}

    # print("forward_data_vol",forward_data_vol,
    #       "backward_data_vol",backward_data_vol)
    return (forward_data_vol,backward_data_vol,average_hit_read,average_hit_write,forward_times,backward_times)
    # if isMeCache:
    #     return (forward_data_vol,ret_dict,average_hit_read,average_hit_write,forward_times,backward_times,consistency_dict)
    # else:
    #     return (forward_data_vol,ret_dict,average_hit_read,average_hit_write,forward_times,backward_times)

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
    root='./save'
    dir_list=['RGAT-MAG','RGAT-OM','RGCN-MAG','RGCN-OM']
    name_transfer={
        'dgl':'DGL',
        'heta':'Heta',
        'drgnn':'MeCache'
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
                if name_transfer[name_list[SYSTEM]]=='MeCache' and 'igb' not in name_list[DATASET]:
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
    return total_dict

def draw_comm_with_time(data, pic_name):
    # 提取数据集
    datasets = list(data.keys())
    
    # 创建一行两列的子图
    fig, axes = plt.subplots(1, 2, figsize=(4.25, 1.5))
    
    # 设置子图之间的间距
    plt.subplots_adjust(wspace=0)

    # 定义颜色和hatch模式
    hatch_list = ['//', '\\\\',  'xx', '||','++']
    
    # 为每个数据集创建子图
    for idx, dataset in enumerate(datasets):
        ax1 = axes[idx]  # 左轴（数据传输量）
        ax2 = ax1.twinx()  # 右轴（时间）
        
        # 提取当前数据集的数据
        gpu_counts = sorted(data[dataset].keys())  # ['2', '4']
        frameworks = ['Heta', 'MeCache']
        
        # 设置x轴位置 - 组内间距小，组间间距大
        group_spacing = 1.5  # 组间间距
        bar_width = 0.7  # 单个柱子的宽度
        
        # 计算每个组的位置（每组对应一个框架）
        x_groups = [i * group_spacing for i in range(len(frameworks))]
        
        # 计算每个柱子的位置
        x_positions = []
        for group_x in x_groups:
            # 组内两个柱子的位置（对应2 GPU和4 GPU）
            for j, gpu in enumerate(gpu_counts):
                # 组内居中对称分布
                offset = (j - 0.5) * bar_width
                x_positions.append(group_x + offset)
        
        # 准备堆叠柱状图数据 - 现在按框架分组，组内是GPU数量
        emb_volumes = []
        opt_volumes = []
        a2a_volumes = []
        times = []
        
        for framework in frameworks:
            for gpu in gpu_counts:
                volume_dict = data[dataset][gpu][framework]
                emb_volumes.append(volume_dict['emb'])
                opt_volumes.append(volume_dict['opt'])
                a2a_volumes.append(volume_dict['A2A'])
                times.append(volume_dict['time'])
        
        # 绘制堆叠柱状图（左轴）
        bars1 = ax1.bar(x_positions, opt_volumes, bar_width, 
                       label='Opt. CPU-GPU', 
                       color=color_list[0], 
                       edgecolor='black',
                       hatch=hatch_list[0])
        
        bars2 = ax1.bar(x_positions, emb_volumes, bar_width, 
                       bottom=numpy.array(opt_volumes),
                       label='Emb. GPU-CPU', 
                       color=color_list[1], 
                       edgecolor='black',
                       hatch=hatch_list[1])

        bars3 = ax1.bar(x_positions, a2a_volumes, bar_width, 
                       bottom=numpy.array(opt_volumes)+numpy.array(emb_volumes),
                       label='Emb. A2A', 
                       color=color_list[2], 
                       edgecolor='black',
                       hatch=hatch_list[2])
        
        # 在每根柱子上方添加总数据量文本
        max_volume = max([opt+emb+ + a2a for opt,emb, a2a in zip(opt_volumes, emb_volumes, a2a_volumes)])
        for i, (bar_x, opt,emb, a2a) in enumerate(zip(x_positions, opt_volumes, emb_volumes, a2a_volumes)):
            total_volume = opt+emb+ + a2a
            ax1.text(bar_x, total_volume + max_volume * 0.02, 
                     f'{total_volume:.1f}', 
                     ha='center', va='bottom', fontsize=7)
        
        # 绘制时间标记（右轴）- 使用黑色'+'号
        for i, (time_x, time_val) in enumerate(zip(x_positions, times)):
            ax2.plot(time_x, time_val, marker='v',color='black', markersize=4)
        
        # 设置子图标题和标签
        ax1.set_title(dataset, fontsize=11, fontweight='bold')
        
        if idx == 0:
            ax1.set_ylabel('Comm. Volume (GB)', fontsize=10)
        else:
            ax2.set_ylabel('Time (s)', fontsize=10)
        
        # 设置x轴标签 - 柱子下方显示GPU数量
        gpu_labels = []
        for framework in frameworks:
            for gpu in gpu_counts:
                gpu_labels.append(gpu)
        
        ax1.set_xticks(x_positions,labelpad=-0.5)
        ax1.set_xticklabels(gpu_labels, rotation=0, fontsize=7)
        
        # 缩短x轴tick和x轴的距离
        ax1.tick_params(axis='x', which='major', pad=2)
        
        # 在每组下方中间添加框架名标签
        for i, framework in enumerate(frameworks):
            group_center = x_groups[i]
            # 在x轴下方添加框架名标签
            ax1.text(group_center, -max_volume * 0.5, framework, 
                    ha='center', fontsize=8)
        
        # 添加网格（只保留水平网格线）
        ax1.yaxis.grid(True, linestyle='--', alpha=0.7)
        
        # 设置x轴范围，确保组对称分布
        ax1.set_xlim(x_groups[0] - 0.8, x_groups[-1] + 0.8)
        
        # 设置具体的y轴范围
        if idx == 0:
            ax1.set_ylim(0, 75)
            ax1.set_yticks([0, 50])
            ax2.set_ylim(5, 5+(25-5)*75/50)
            ax2.set_yticks([5, 25])
        else:
            ax1.set_ylim(0, 160)
            ax1.set_yticks([0, 100])
            ax2.set_ylim(15, 15+(40-15)/100*160)
            ax2.set_yticks([15, 40])
    
    # 创建图例
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles1, labels1 = ax1.get_legend_handles_labels()
    
    # 添加时间标记的图例项
    time_handle = plt.Line2D([0], [0], marker='v',color='black', markersize=4,linestyle='None', label='Time')

    # 合并图例
    all_handles = handles1 + [time_handle]
    all_labels = labels1 + ['Time']
    
    # 将图例放在整个图的上方
    fig.legend(all_handles, all_labels, 
               bbox_to_anchor=(0.5, 0.85), fontsize=8, 
               loc='lower center', ncol=2, 
               frameon=False)

    plt.tight_layout()
    
    # 调整布局，为底部图例和框架标签留出空间
    plt.subplots_adjust(bottom=0.25)
    
    # 保存图片
    fig.savefig(f'{pic_name}.png', dpi=300, format="png", bbox_inches='tight')
    fig.savefig(f'{pic_name}.pdf', dpi=300, format="pdf", bbox_inches='tight')
    fig.savefig(f'{pic_name}.svg', dpi=300, format="svg", bbox_inches='tight')
    print(f"图片已保存为：{pic_name}.pdf")

    # 打印数据汇总
    print("\n数据汇总:")
    for dataset in datasets:
        print(f"\n{dataset}:")
        for gpu in sorted(data[dataset].keys()):
            print(f"  {gpu} GPU:")
            for framework in ['Heta', 'MeCache']:
                vol_dict = data[dataset][gpu][framework]
                total_vol = vol_dict['emb'] + vol_dict['opt'] + vol_dict['A2A']
                print(f"    {framework}: Total={total_vol:.1f}GB, "
                      f"emb={vol_dict['emb']:.1f}GB, "
                      f"opt={vol_dict['opt']:.1f}GB, "
                      f"A2A={vol_dict['A2A']:.1f}GB, "
                      f"Time={vol_dict['time']:.1f}s")

if __name__ == '__main__':
    total_dict=cul()
    
    model='RGCN'
    data_dict={dataset:{gpu:{system: 1 for system in ['Heta','MeCache']} for gpu in ['2','4']} for dataset in ['OM','MAG']}
    for dataset in ['OM','MAG']:
        for gpu in ['2','4']:
            for system in ['Heta','MeCache']:
                data_dict[dataset][gpu][system]={"emb":total_dict[f"{model}-{dataset}"][system][gpu][1]["emb"],\
                                                  "opt":total_dict[f"{model}-{dataset}"][system][gpu][1]["opt"],\
                                                    "A2A":total_dict[f"{model}-{dataset}"][system][gpu][1]["a2a"],\
                                                          "time":total_dict[f"{model}-{dataset}"][system][gpu][-1]}
                print(dataset,gpu,system,data_dict[dataset][gpu][system])
    
    # print(data_dict)
    # for dataset in ['OM','MAG']:
    #     for gpu in ['2','4']:
    #         CPU_GPU_bandwidth=data_dict[dataset][gpu]['Heta']["CPU-GPU"]/data_dict[dataset][gpu]['Heta']["time"]
    #         print(f"{gpu} gpu: CPU-GPU PCIe bandwidth",CPU_GPU_bandwidth)
    #         A2A_bandwidth=data_dict[dataset][gpu]['MeCache']["A2A"]/(data_dict[dataset][gpu]['MeCache']["time"]-data_dict[dataset][gpu]['MeCache']["CPU-GPU"]/CPU_GPU_bandwidth)
    #         print(f"{gpu} gpu: A2A PCIe bandwidth",A2A_bandwidth)
    
    # 绘制带加速比的堆叠柱状图
    draw_comm_with_time(data_dict, 'consistency')
