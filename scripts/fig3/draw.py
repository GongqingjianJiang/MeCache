import numpy
import sys
import os
import matplotlib.pyplot as plt
from collections import defaultdict
project_root = '../..'
sys.path.append(project_root)

from src import extract_key_word_from_file, draw_color,color_list
from src.utils import draw_linestyle
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
    
def draw_time_breakdown(data_dict, file_path):
    fig,ax=plt.subplots(figsize=(4.5, 2.2))
    patterns = ['//', '\\\\',  'xx', '||','++']
    # 添加标题和标签
    plt.ylabel('Normalized Execution Time', fontsize=10)

    bar_width = 0.5
    categories = []
    sample = []
    featcopy = []
    emb_update = []
    model_update = []
    train = []
    for dataset in data_dict.values():
        for system, values in dataset.items():
            categories.append([dataset,system])
            sample.append(values[0])
            featcopy.append(values[1])
            emb_update.append(values[2])
            model_update.append(values[3])
            train.append(values[4])
    x_axis=[i for i in range(len(categories))]
    bars1 = plt.bar(x_axis, sample, bar_width, label='Sample', zorder=100,color=color_list[0],edgecolor='black',hatch=patterns[0])
    bars2 = plt.bar(x_axis, featcopy, bar_width, label='Feat Retrieval', zorder=100,color=color_list[1], bottom=sample,edgecolor='black',hatch=patterns[1])
    bars3 = plt.bar(x_axis, emb_update, bar_width, label='Embedding Update', zorder=100,color=color_list[2], bottom=numpy.array(sample) + numpy.array(featcopy),edgecolor='black',hatch=patterns[2])
    bars4 = plt.bar(x_axis, model_update, bar_width, label='Model Update', zorder=100,color=color_list[3], bottom=numpy.array(sample) + numpy.array(featcopy) + numpy.array(emb_update),edgecolor='black',hatch=patterns[3])
    bars5 = plt.bar(x_axis, train, bar_width, label='Train', zorder=100,color=color_list[4], bottom=numpy.array(sample) + numpy.array(featcopy) + numpy.array(emb_update)+ numpy.array(model_update),edgecolor='black',hatch=patterns[4])

    plt.ylim(0,1)
    plt.yticks(list(numpy.linspace(0,1,num=5)))

    ax.set_xticks(x_axis)
    ax.set_xticklabels([i[1] for i in categories], fontsize=7)

    # 添加图例
    plt.legend(bbox_to_anchor=(0.5, 1.3),fontsize=7,loc='upper center', ncol=3,frameon=False)
    plt.tight_layout()

    dataset_axis = list(data_dict.keys())
    offset = 0  # 用于记录当前绘制到哪一个数据集

    # 在每两根柱子下面加上dataset_axis[i]
    for i in range(len(dataset_axis)):
        dataset_start = offset
        dataset_end = offset + len(data_dict[dataset_axis[i]])
        # 计算中间位置
        mid_pos = (dataset_start + dataset_end - 1) / 2
        plt.text(mid_pos, -0.23, dataset_axis[i], ha='center', fontsize=8, transform=ax.transData)
        offset += len(data_dict[dataset_axis[i]])
    
    ax.yaxis.grid(True, linestyle='--', alpha=0.7,zorder=0)

    fig.savefig(f'{file_path}.pdf',dpi=300,format="pdf" ,bbox_inches='tight', pad_inches=0.02)
    fig.savefig(f'{file_path}.png',dpi=300,format="png" ,bbox_inches='tight', pad_inches=0.02)
    fig.savefig(f'{file_path}.svg',dpi=300,format="svg" ,bbox_inches='tight', pad_inches=0.02)
    print(f"图片已保存为文件：{f'{file_path}.pdf'}")

if __name__ == '__main__':
    root='./log/gloo'
    dir_list=['DGL','Heta']
    dataset_transfer={
        'ogbn-mag':'OM',
        'mag240m':'MAG',
        'igb-full-small':'SM',
        'igb-full-medium':'ME'
    }

    data_dict={dataset:{systems:[] for systems in dir_list} for dataset in dataset_transfer.values()}
    for systems in dir_list:
        folder_path=os.path.join(root,systems)
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.log'):  # 筛选 pickle 文件
                file_path = os.path.join(folder_path, file_name)
                sample_values=extract_key_word_from_file(file_path, 'sample: ')
                featcopy_values=extract_key_word_from_file(file_path, 'feat_copy: ')
                forward_values=extract_key_word_from_file(file_path, 'forward: ')
                backward_values=extract_key_word_from_file(file_path, 'backward: ')
                model_update_values=extract_key_word_from_file(file_path, 'model update: ')
                emb_update_values=extract_key_word_from_file(file_path, 'emb update: ')

                # model_update_values=[model_update_values[i] for i in range(len(model_update_values))]
                # emb_update_values=[emb_update_values[i] for i in range(len(emb_update_values))]
                train_values=[forward_values[i]+backward_values[i] for i in range(len(backward_values))]
                
                sample=numpy.mean(sample_values)
                featcopy=numpy.mean(featcopy_values)
                emb_update=numpy.mean(emb_update_values)
                model_update=numpy.mean(model_update_values)
                train=numpy.mean(train_values)

                _sum=sample+featcopy+emb_update+train+model_update

                sample/=_sum
                featcopy/=_sum
                emb_update/=_sum
                model_update/=_sum
                train/=_sum

                name_list=split_name(file_name)
                print(f"{name_list[SYSTEM]},{name_list[DATASET]},[{sample}, {featcopy},{emb_update}, {train}]")
                data_dict[dataset_transfer[name_list[DATASET]]][systems]=[sample, featcopy, emb_update, model_update, train]
        
    print("data_dict",data_dict)
    heta_total=[]
    heta_featrue=[]
    heta_embedding=[]
    dgl_total=[]
    dgl_featrue=[]
    dgl_embedding=[]
    for dataset, systems_dict in data_dict.items():
        for system, values in systems_dict.items():
            print(dataset,system)
            print("sample",values[0],end=', ')
            print("featcopy",values[1],end=', ')
            print("update",values[2],end=', ')
            print("train",values[3])
            if system=='DGL':
                dgl_total.append(values[1]+values[2])
                dgl_featrue.append(values[1])
                dgl_embedding.append(values[2])
            else:
                heta_total.append(values[1]+values[2])
                heta_featrue.append(values[1])
                heta_embedding.append(values[2])
    draw_time_breakdown(data_dict,'normalized_epoch')
    print('average DGL: total:',numpy.mean(dgl_total),'feature retrieval: ',numpy.mean(dgl_featrue),'embedding update',numpy.mean(dgl_embedding))
    print('average Heta: total:',numpy.mean(heta_total),'feature retrieval: ',numpy.mean(heta_featrue),'embedding update',numpy.mean(heta_embedding))

    print('max DGL: total:',numpy.max(dgl_total),'feature retrieval: ',numpy.max(dgl_featrue),'embedding update',numpy.max(dgl_embedding))
    print('max Heta: total:',numpy.max(heta_total),'feature retrieval: ',numpy.max(heta_featrue),'embedding update',numpy.max(heta_embedding))