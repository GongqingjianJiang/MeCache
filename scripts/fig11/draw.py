import numpy
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter,FuncFormatter,FormatStrFormatter,MultipleLocator

project_root = '/gf3/home/jgqj/test_code/hydro'
sys.path.append(project_root)

from src.utils import extract_key_word_from_file,draw_linestyle, color_list

def generate_number(x):
    # 将数字转换为字符串以便处理
    x_str = str(int(x))
    length = len(x_str)
    
    # 获取前两位数字
    a = x_str[:2]

    a=str(int(a)+1)
    
    # 生成与x相同位数的数字，以a开头，后面补零
    new_num = a + '0' * (length - 2)
    
    return int(new_num)

def draw_acc_time(subplot, test_acc_dict, ylim, file_path):
    # 添加标题和标签
    subplot.set_title(file_path, fontsize=11)
    subplot.set_xlabel('Time (s)', fontsize=10)

    x_max=0
    for j, (label, tensor) in enumerate(test_acc_dict.items()):
        test_acc_list = tensor[0]
        epoch_time_values = tensor[1]
        cumulative_time = [sum(epoch_time_values[:i+1]) for i in range(len(epoch_time_values))]
        subplot.plot(cumulative_time, test_acc_list, 
                     label=label,color=color_list[j],zorder=100)
        print(label,cumulative_time[-1])
        if cumulative_time[-1]>x_max:
            x_max=cumulative_time[-1]

    x_max=generate_number(x_max)
    plt.ylim(ylim[0],ylim[1])
    if file_path == 'RGAT-ME':
        subplot.set_ylabel('Test Accuracy (%)', fontsize=10)
        x_max=4000
    else:
        x_max=generate_number(x_max)
        ylim[1]=0.7
    plt.xlim(0,x_max)
    # formatter = ScalarFormatter(useMathText=True)
    # formatter.set_powerlimits((0, 2))  # 设置指数显示的阈值
    # subplot.xaxis.get_offset_text().set_fontsize(8)
    # subplot.xaxis.set_major_formatter(formatter)
    subplot.set_xticks(list(numpy.linspace(0,x_max,num=3)))
    subplot.set_yticks(list(numpy.linspace(ylim[0],ylim[1],num=3)))

    # 添加图例
    subplot.legend(fontsize=8,loc='lower right')

    # # 添加网格
    # subplot.grid(True)

if __name__ == '__main__':
    igb_baseline='./log/final/baseline/heta_rgat_igb-full-medium_none_64_acc.log'
    igb_drgnn_acc='./log/final/acc/drgnn_rgat_igb-full-medium_miss_penalty_64_128,8.log'
    igb_drgnn_speed='./log/final/speed/drgnn_rgat_igb-full-medium_miss_penalty_64_128,8.log'

    mag_baseline='./log/final/baseline/Heta_rgcn_mag240m_none_64_1parts.log'
    mag_baseline_speed='./log/final/speed/dgl_rgcn_mag240m_none_64_4.log'
    mag_drgnn_acc='./log/final/acc/Heta_rgcn_mag240m-pca_miss_penalty_64_128,8.log'
    mag_drgnn_speed='./log/final/speed/drgnn_rgcn_mag240m_miss_penalty_64_4.log'
    
    fig=plt.figure(figsize=(4.25, 1.8))
    data_dict2={}
    epoch_time_values=extract_key_word_from_file(igb_drgnn_speed, 'Epoch Time(s):')
    test_acc_values=extract_key_word_from_file(igb_drgnn_acc, 'Test Acc')[:13]
    epoch_time_values=numpy.mean(epoch_time_values)
    epoch_time_values=[epoch_time_values for _ in range(13)]
    my_time=sum(epoch_time_values)
    epoch_time_values.insert(0,0)
    test_acc_values.insert(0,0)
    data_dict2['MeCache']=[test_acc_values, epoch_time_values]

    epoch_time_values=extract_key_word_from_file(igb_baseline, 'Part 0, Epoch Time(s):')[:23]
    test_acc_values=extract_key_word_from_file(igb_baseline, 'Test Acc')[:23]
    epoch_time_values.insert(0,0)
    base_time=sum(epoch_time_values)
    test_acc_values.insert(0,0)
    data_dict2['DGL']=[test_acc_values, epoch_time_values]
    draw_acc_time(fig.add_subplot(121),data_dict2,(0.60,0.80),"RGAT-ME")
    # plt.grid(axis='y', linestyle='--', alpha=0.7,zorder=0)

    print("igbn-medium acc speed up:",base_time/my_time)

    data_dict2={}
    epoch_time_values=extract_key_word_from_file(mag_drgnn_speed, 'Epoch Time(s):')
    test_acc_values=extract_key_word_from_file(mag_drgnn_acc, 'Test Acc')[:413]
    epoch_time_values=numpy.mean(epoch_time_values)
    epoch_time_values=[epoch_time_values for _ in range(413)]
    my_time=sum(epoch_time_values)
    epoch_time_values.insert(0,0)
    test_acc_values.insert(0,0)
    data_dict2['MeCache']=[test_acc_values, epoch_time_values]

    epoch_time_values=extract_key_word_from_file(mag_baseline_speed, 'Part 0, Epoch Time(s):')
    test_acc_values=extract_key_word_from_file(mag_baseline, 'Test Acc')[:425]
    epoch_time_values=numpy.mean(epoch_time_values)
    epoch_time_values=[epoch_time_values for _ in range(425)]
    base_time=sum(epoch_time_values)
    epoch_time_values.insert(0,0)
    test_acc_values.insert(0,0)
    data_dict2['DGL']=[test_acc_values, epoch_time_values]
    draw_acc_time(fig.add_subplot(122),data_dict2,[0.5,0.73],"RGCN-MAG")

    print("mag acc speed up:",base_time/my_time)
    # should be 6.285894868279108, 

    plt.tight_layout()
    # plt.grid(axis='y', linestyle='--', alpha=0.7,zorder=0)
    fig.savefig(f'time_to_acc.pdf',dpi=300,format="pdf",bbox_inches='tight',pad_inches=0.02)
    fig.savefig(f'time_to_acc.png',dpi=300,format="png",bbox_inches='tight',pad_inches=0.02)
    fig.savefig(f'time_to_acc.svg',dpi=300,format="svg",bbox_inches='tight',pad_inches=0.02)
    print(f"图片已保存为文件：{'time_to_acc.pdf'}")

# RGAT-MAG 4 DGL 5.485379555489134 
# 66.0 epoch 00164
# 65.72 Epoch 00169
# RGCN-ME 4 DGL 5.4543865053094835 
# 77.52, 19
# 77.11, 18