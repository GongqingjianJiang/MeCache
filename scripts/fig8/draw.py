import numpy
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter,FuncFormatter,FormatStrFormatter,MultipleLocator
import math
project_root = '/gf3/home/jgqj/test_code/hydro'
sys.path.append(project_root)

from src import extract_key_word_from_file, draw_color,color_list

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

def draw_time(total_dict, pic_name):
    patterns = ['//', '\\\\',  'xx', '||','++']
    # 创建一个 2x5 的子图布局
    fig, axs = plt.subplots(2, 5, figsize=(8.5, 2.8))

    # 定义x轴的坐标
    x_labels = ['1', '2', '4']
    x = [0, 1, 2]

    # 定义每个柱子的宽度
    bar_width = 0.2

    # 定义每个系统的偏移量
    offsets = [-0.25, 0, 0.25]

    # 颜色列表
    colors = [color for color in draw_color.values()]

    # 遍历每个子图
    for i, (name, systems) in enumerate(total_dict.items()):
        row = i // 5
        col = i % 5
        ax = axs[row, col]
        
        # 遍历每个系统
        max_y=0
        for j, (system, values) in enumerate(systems.items()):
            # 准备数据
            y = [values[i] for i in x_labels]
            if max_y<max(y):
                max_y=max(y)
            # 计算每个柱子的位置
            positions = [xi+ offsets[j] for xi in x]
            # 绘制柱状图
            bars = ax.bar(positions, y, width=bar_width, label=system, color=colors[j],
                          edgecolor='black', zorder=100,hatch=patterns[j])
            
            # 找出值为1的柱子并标记
            for bar in bars:
                height = bar.get_height()
                if height == 1:
                    ax.text(bar.get_x() + bar.get_width()/2., 0.1,
                            'OOM', ha='center', va='bottom', color='black',rotation='vertical',fontsize=6)

        # 设置x轴刻度
        ax.set_xticks(x,fontsize=8)
        ax.set_xticklabels(x_labels,fontsize=8)
        
        # 设置标题和标签
        ax.set_title(name,fontsize=8,fontweight='bold')
        # fig.text(col/(5.0+1), 1 - row/(3.0), name, ha='center', va='center', fontsize=9, fontweight='bold')
        # fig.text(col / 5.0 + 0.5 / 5.0, row / 2.0 - 0.15, name, 
        #          ha='center', va='top', fontsize=9, fontweight='bold')
        if row ==1:
            ax.set_xlabel('#GPU',fontsize=8,labelpad=-0.5)
        if col ==0:
            ax.set_ylabel('Throughput (seed/s)',fontsize=8)
    
        ax.tick_params(axis='y', labelsize=8)
        # ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((0, 2))  # 设置指数显示的阈值
        ax.yaxis.get_offset_text().set_fontsize(8)
        ax.yaxis.set_major_formatter(formatter)
        gap=5000
        ax.set_ylim(0,math.ceil(max_y/gap)*gap)
        ax.yaxis.set_major_locator(MultipleLocator(5000))
        # y_ticks = ax.get_yticks()
        # y_ticks = [f"{yt:.2f}" for yt in y_ticks]
        # ax.set_yticklabels(y_ticks, fontsize=12)
        ax.yaxis.grid(True, linestyle='--', alpha=0.7,zorder=0)

        # 添加图例
        ax.legend(fontsize=6,loc='upper left')

    # 调整布局
    plt.tight_layout(h_pad=0.1,w_pad=0.2)
    # plt.subplots_adjust(hspace=0.4, wspace=0.4)

    # 显示图表
    # plt.show()
    fig.savefig(f'{pic_name}.pdf',dpi=300,format="pdf",bbox_inches='tight',pad_inches=0.02)
    fig.savefig(f'{pic_name}.png',dpi=300,format="png",bbox_inches='tight',pad_inches=0.02)
    fig.savefig(f'{pic_name}.svg',dpi=300,format="svg",bbox_inches='tight',pad_inches=0.02)
    print(f"图片已保存为文件：{pic_name}.pdf")

if __name__ == '__main__':
    root='./log/new/all'
    # dir_list=['RGAT-MAG','RGAT-ME','RGAT-OM','RGAT-SM','RGAT-LA',
    #           'RGCN-MAG','RGCN-ME','RGCN-OM','RGCN-SM','RGCN-LA']
    dir_list=['RGCN-OM','RGCN-MAG','RGCN-SM','RGCN-ME','RGCN-LA',
              'RGAT-OM','RGAT-MAG','RGAT-SM','RGAT-ME','RGAT-LA']
    seed_dict={
        'RGAT-MAG':1112392,
        'RGCN-MAG':1112392,
        'RGAT-ME':1000000,
        'RGCN-ME':1000000,
        'RGAT-OM':629571,
        'RGCN-OM':629571,
        'RGAT-SM':600000,
        'RGCN-SM':600000,
        'RGCN-LA':10000000,
        'RGAT-LA':10000000,
    }
    name_transfer={
        'dgl':'DGL',
        'heta':'Heta',
        'drgnn':'MeCache'
    }
    gpu_num=['1', '2', '4']
    total_dict={subgraph_title:{system:{} for system in name_transfer.values()} for subgraph_title in dir_list}
    # print(total_dict)
    for subgraph_title in dir_list:
        folder_path=os.path.join(root,subgraph_title)
        for filename in os.listdir(folder_path):
            if filename.endswith('.log'):  # 筛选 pickle 文件
                name_list=split_name(filename)
                if name_list[GPU] not in gpu_num:
                    # print(filename,name_list,name_list[GPU])
                    continue
                file_path = os.path.join(folder_path, filename)
                epoch_values=extract_key_word_from_file(file_path, 'Epoch Time(s):')
                epoch_time=numpy.mean(epoch_values)
                seed_num=seed_dict[subgraph_title]
                total_dict[subgraph_title][name_transfer[name_list[SYSTEM]]][name_list[GPU]]=seed_num/epoch_time
    total_dict['RGCN-LA']['DGL']={'1': 1, '2': 1, '4': 1}
    total_dict['RGAT-LA']['DGL']={'1': 1, '2': 1, '4': 1}
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
        if subgraph_title in ['RGCN-LA','RGAT-LA']:
            continue
        for gpu in gpu_num:
            speed_up_dgl=total_dict[subgraph_title]['MeCache'][gpu]/total_dict[subgraph_title]['DGL'][gpu]
            speed_up_heta=total_dict[subgraph_title]['MeCache'][gpu]/total_dict[subgraph_title]['Heta'][gpu]
            if _max<max(speed_up_dgl,speed_up_heta):
                _max=max(speed_up_dgl,speed_up_heta)
            if _min>min(speed_up_dgl,speed_up_heta):
                _min=min(speed_up_dgl,speed_up_heta)
            speedup_dict[subgraph_title][gpu]['DGL']=speed_up_dgl
            speedup_dict[subgraph_title][gpu]['Heta']=speed_up_heta
            # total_volumn_dgl.append(total_dict[subgraph_title]['DGL'][gpu])
            # total_volumn_heta.append(total_dict[subgraph_title]['Heta'][gpu])
            # total_volumn_drgnn.append(total_dict[subgraph_title]['MeCache'][gpu])
            total_volumn_dgl.append(speed_up_dgl)
            total_volumn_heta.append(speed_up_heta)
            # total_volumn_drgnn.append(total_dict[subgraph_title]['MeCache'][gpu])
            print(subgraph_title,gpu,'DGL',speed_up_dgl,'Heta',speed_up_heta)
    print("max,",_max,"min,",_min)
    # print('average speed up','DGL:',sum(total_volumn_drgnn)/sum(total_volumn_dgl),'Heta:',sum(total_volumn_drgnn)/sum(total_volumn_heta))
    print('average speed up','DGL:',numpy.mean(total_volumn_dgl),'Heta:',numpy.mean(total_volumn_heta),'total',numpy.mean(total_volumn_dgl+total_volumn_heta))

    scalability={
        'DGL':[],
        'Heta':[],
        'MeCache':[]
    }
    for subgraph_title in dir_list:
        # if subgraph_title in ['RGCN-LA','RGAT-LA']:
        #     continue
        for systems in scalability.keys():
            scalability[systems].append(total_dict[subgraph_title][systems]['4']/total_dict[subgraph_title][systems]['1']/4)
            scalability[systems].append(total_dict[subgraph_title][systems]['2']/total_dict[subgraph_title][systems]['1']/2)
    for key, value in scalability.items():
        print('scalability', key, numpy.mean(value))
    draw_time(total_dict,'overall_performance')
    
