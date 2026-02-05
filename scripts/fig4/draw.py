import numpy
import sys
import os
import matplotlib.pyplot as plt
import math
from matplotlib.lines import Line2D
import numpy as np
project_root = '/gf3/home/jgqj/test_code/hydro'
sys.path.append(project_root)

from src import extract_key_word_from_file,color_list

def draw(acc_data_dict, file_path):
    fig, ax1 = plt.subplots(figsize=(4.25, 1.5))

    # 存储原始 x 值用于标注
    original_x_values = {}
    
    # get ylim
    for i, label in enumerate(acc_data_dict):
        x, y = acc_data_dict[label]
        original_x_values[label] = x  # 保存原始 x 值
        acc_data_dict[label] = [[math.log(i, 2) for i in x], y]

    # 添加标题和标签
    ax1.set_xlabel('Reduction Ratio', fontsize=8)
    ax1.set_ylabel('Accuracy (%)', fontsize=8)
    ax1.set_ylim(73.5, 79)

    # baseline marker
    ax1.plot(0.01, 78.5, marker=">", color='black', markersize=3, transform=ax1.get_yaxis_transform())
    # ax1.hlines(y=78.8, xmin=0, xmax=0.02, color=color_list[-1], linestyle='-', linewidth=1, transform=ax1.get_yaxis_transform())

    plt.yticks(list(numpy.linspace(74, 79, num=6)), fontsize=7)
    log4 = math.log(300, 2)
    ax1.set_xlim(0, log4)
    plt.xticks(ticks=[i for i in range(math.floor(log4 + 1))], 
               labels=[2**i for i in range(math.floor(log4 + 1))], fontsize=7)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7, zorder=0)

    line_style = ['-', '--']
    markers = ['o', 's']  # 添加不同的标记
    for i, label in enumerate(acc_data_dict):
        x, y = acc_data_dict[label]
        # 绘制线条和标记
        line = ax1.plot(x, y, linestyle=line_style[0], marker=markers[i], markersize=3,
                        label=label, color=color_list[i])
        
        # 在每个数据点上添加 x 值标注
        # for j, (x_val, y_val) in enumerate(zip(x, y)):
        #     # 使用原始 x 值进行标注
        #     original_x = original_x_values[label][j]
        #     ax1.annotate(f'{original_x:.1f}', 
        #                  xy=(x_val, y_val), 
        #                  xytext=(5, 5), 
        #                  textcoords='offset points',
        #                  fontsize=2,
        #                  color=color_list[i])

    # 添加图例
    fig.legend(bbox_to_anchor=(0.12, 0.32), fontsize=8, loc='lower left', ncol=1)
    
    fig.tight_layout()
    
    fig.savefig(f'{file_path}.pdf', dpi=300, format="pdf", bbox_inches='tight', pad_inches=0.02)
    fig.savefig(f'{file_path}.png', dpi=300, format="png", bbox_inches='tight', pad_inches=0.02)
    fig.savefig(f'{file_path}.svg', dpi=300, format="svg", bbox_inches='tight', pad_inches=0.02)
    print(f"图片已保存为文件：{f'{file_path}.pdf'}")

def get_acc(acc):
    # acc=numpy.mean(acc[-20:])
    acc=max(acc)
    # acc=acc[81]
    # acc=max(acc[:300])
    return acc

def get_base_acc(dir_name):
    x_axis=[]
    y_axis=[]
    for filename in sorted(os.listdir(dir_name)):
        if filename.endswith('.log'):
            file_path = os.path.join(dir_name, filename)
            dim0,dim1=filename.split('.log')[0].split('_')[-1].split(',')
            if dim0!=dim1:
                continue
            acc=extract_key_word_from_file(file_path, 'Test Acc ')
            acc=get_acc(acc)
            # print([dim0,dim1], acc)
            x_axis.append(1024/int(dim0))
            y_axis.append(acc*100)
    # print(x_axis,y_axis)
    paired=list(zip(x_axis,y_axis))
    sorted_paired_list = sorted(paired, key=lambda x: x[0])
    x, y = zip(*sorted_paired_list)
    # print(x,y)
    return [x,y]

def get_base_time(dir_name):
    x_axis=[]
    y_axis=[]
    for filename in sorted(os.listdir(dir_name)):
        if filename.endswith('.log'):
            file_path = os.path.join(dir_name, filename)
            dim0,dim1=filename.split('.log')[0].split('_')[-1].split(',')
            if dim0!=dim1:
                continue
            epoch_time=extract_key_word_from_file(file_path, 'Epoch Time(s): ')
            epoch_time=numpy.mean(epoch_time)
            # print([dim0,dim1], epoch_time)
            x_axis.append(1024/int(dim0))
            y_axis.append(epoch_time)
    # print(x_axis,y_axis)
    paired=list(zip(x_axis,y_axis))
    sorted_paired_list = sorted(paired, key=lambda x: x[0])
    x, y = zip(*sorted_paired_list)
    # print(x,y)
    return [x,y]

# specilized for igbh-small
def get_reduction_ratio(dim0,dim1):
    ret=3147758/((3147758-2147758)*(int(dim0)/1024)+2147758*(int(dim1)/1024))
    print(f'{dim0},{dim1}',ret)
    return ret

# no_good_method=[['512','128'],['512','32'],['512','16'],['512','8'],['512','4'],['512','2'],['512','1'],
#                 ['256','64'],
#                 ['128','4'],['128','2'],['128','1']]
good_method=[['128', '32'] ,['64', '32'],['64', '16'],['64', '4'],['32', '8'],['32', '2'],['16', '4'],['8', '4'],['8', '2']]

def get_my_acc(dir_name):
    x_axis=[]
    y_axis=[]
    for filename in sorted(os.listdir(dir_name)):
        if filename.endswith('.log'):
            file_path = os.path.join(dir_name, filename)
            dim0,dim1=filename.split('.log')[0].split('_')[-1].split(',')
            if dim0==dim1:
                continue
            # if [dim0,dim1] in no_good_method:
            #     continue
            if [dim0,dim1] not in good_method:
                continue
            acc=extract_key_word_from_file(file_path, 'Test Acc ')
            acc=get_acc(acc)
            print([dim0,dim1], acc)
            x_axis.append(get_reduction_ratio(dim0,dim1))
            y_axis.append(acc*100)
    # print(x_axis,y_axis)
    paired=list(zip(x_axis,y_axis))
    sorted_paired_list = sorted(paired, key=lambda x: x[0])
    x, y = zip(*sorted_paired_list)
    # print(x,y)
    return [x,y]

def get_my_time(dir_name):
    x_axis=[]
    y_axis=[]
    for filename in sorted(os.listdir(dir_name)):
        if filename.endswith('.log'):
            file_path = os.path.join(dir_name, filename)
            dim0,dim1=filename.split('.log')[0].split('_')[-1].split(',')
            if dim0==dim1:
                continue
            # if [dim0,dim1] in no_good_method:
            #     continue
            epoch_time=extract_key_word_from_file(file_path, 'Epoch Time(s): ')
            epoch_time=numpy.mean(epoch_time)
            # print([dim0,dim1], epoch_time)
            x_axis.append(get_reduction_ratio(dim0,dim1))
            y_axis.append(epoch_time)
    # print(x_axis,y_axis)
    paired=list(zip(x_axis,y_axis))
    sorted_paired_list = sorted(paired, key=lambda x: x[0])
    x, y = zip(*sorted_paired_list)
    # print(x,y)
    return [x,y]

def cut(dual,low_bar,high_bar):
    x_axis,y_axis=dual
    paired=list(zip(x_axis,y_axis))
    ret=[]
    for i in paired:
        if low_bar<=i[0]<=high_bar:
            ret.append(i)
    # print(paired)
    # print(ret)
    x,y=zip(*ret)
    return [x,y]

if __name__ == '__main__':
    base_acc=get_base_acc('./log/pca_acc_19_linear')
    # base_time=get_base_time('./log/pca_acc_19')
    my_acc=get_my_acc('./log/pca_acc_19_linear')
    # my_time=get_my_time('./log/pca_acc_19')

    high_bar=280
    low_bar=2
    mid_bar=16
    base_acc=cut(base_acc,low_bar,high_bar)
    # base_time=cut(base_time,bar)
    # my_acc=cut(my_acc,mid_bar,high_bar)
    # my_time=cut(my_time,bar)
    print('base_acc',base_acc)
    # print('base_time',base_time)
    print('my_acc',my_acc)
    # print('my_time',my_time)
    acc_data_dict={
        'PCA':base_acc,
        'Meta Reduction':my_acc
    }
    # time_data_dict={
    #     'PCA':base_time,
    #     'MetaReduction':my_time
    # }

    draw(acc_data_dict,'pca_drawback')
