import numpy
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter,FuncFormatter
import math
project_root = '/gf3/home/jgqj/test_code/hydro'
sys.path.append(project_root)

from src.utils import extract_key_word_from_file,color_list

def draw_time_stacked_with_speedup(data, categories, pic_name):
    # 提取数据 - 现在每个数据点包含4个部分
    patterns = ['//', '\\\\',  'xx', '||','++']
    datasets = list(data.keys())
    
    # 准备堆叠数据
    sample_values = []
    featcopy_values = []
    train_values = []
    update_values = []
    
    # 计算总时间和加速比
    total_times = {}
    speedup_ratios = {}
    
    for dataset in datasets:
        total_times[dataset] = {}
        speedup_ratios[dataset] = {}
        
        # 计算每个方法的总时间
        for method in categories:
            if method in data[dataset]:
                time_dict = data[dataset][method]
                total_time = time_dict['Sample'] + time_dict['Feat Retrieval'] + time_dict['Train'] + time_dict['Embedding Update']
                total_times[dataset][method] = total_time
        
        # 计算加速比（相对于第一个方法，即DGL）
        baseline = total_times[dataset][categories[0]]
        for method in categories:
            if method in total_times[dataset]:
                speedup_ratios[dataset][method] = baseline / total_times[dataset][method]
            else:
                speedup_ratios[dataset][method] = 1.0  # 如果没有数据，加速比为1
    
    # 创建一行两列的子图
    fig, axes = plt.subplots(1, 2, figsize=(4.25, 1.5))
    
    # 设置子图之间的间距
    plt.subplots_adjust(wspace=0.2)
    
    # 为每个数据集创建子图
    for idx, dataset in enumerate(datasets):
        ax1 = axes[idx]
        ax2 = ax1.twinx()
        
        # 提取当前数据集的数据
        dataset_sample = []
        dataset_featcopy = []
        dataset_train = []
        dataset_update = []
        dataset_total = []
        
        for cat in categories:
            if cat in data[dataset]:
                time_dict = data[dataset][cat]
                dataset_sample.append(time_dict['Sample'])
                dataset_featcopy.append(time_dict['Feat Retrieval'])
                dataset_train.append(time_dict['Train'])
                dataset_update.append(time_dict['Embedding Update'])
                dataset_total.append(time_dict['Sample'] + time_dict['Feat Retrieval'] + 
                                   time_dict['Train'] + time_dict['Embedding Update'])
            else:
                dataset_sample.append(0)
                dataset_featcopy.append(0)
                dataset_train.append(0)
                dataset_update.append(0)
                dataset_total.append(0)
        
        # 设置x轴位置
        x = numpy.arange(len(categories))
        width = 0.7  # 柱状图的宽度

        # 绘制堆叠柱状图（左轴）
        bars1 = ax1.bar(x, dataset_sample, width, label='Sample', zorder=100, 
                       color=color_list[0], edgecolor='black', hatch=patterns[0])
        bars2 = ax1.bar(x, dataset_featcopy, width, label='Feat Retrieval', zorder=100, 
                       bottom=numpy.array(dataset_sample), 
                       color=color_list[1], edgecolor='black', hatch=patterns[1])  
        bars3 = ax1.bar(x, dataset_update, width, label='Embedding Update', zorder=100, 
                       bottom=numpy.array(dataset_sample) + numpy.array(dataset_featcopy), 
                       color=color_list[2], edgecolor='black', hatch=patterns[2])
        bars4 = ax1.bar(x, dataset_train, width, label='Train', zorder=100, 
                       bottom=numpy.array(dataset_sample) + numpy.array(dataset_featcopy) + numpy.array(dataset_update), 
                       color=color_list[3], edgecolor='black', hatch=patterns[3])

        # 在每根柱子上方添加epoch时间文本
        max_value = max(dataset_total) if dataset_total else 1
        if idx==0:
            ax1.set_ylim(0, 666)
            ax1.set_yticks([0,200,400,600])
            ax2.set_ylim(0, 6.66)
            ax2.set_yticks([0,2,4,6])
        else:
            ax1.set_ylim(0, 444)
            ax1.set_yticks([0,200,400])
            ax2.set_ylim(0, 6.66)
            ax2.set_yticks([0,3,6])
        
        for i, (bar_x, total_time) in enumerate(zip(x, dataset_total)):
            if total_time > 0:  # 只在有数据的柱子上添加文本
                ax1.text(bar_x, total_time + max_value * 0.02, f'{total_time:.1f}s', 
                        ha='center', va='bottom', fontsize=7, rotation=0, zorder=200)
        
        # 设置子图标题和标签
        ax1.set_title(dataset, fontsize=11, fontweight='bold')
        
        if idx==0:
            ax1.set_ylabel('Epoch Time (s)', fontsize=10)
        else:
            ax2.set_ylabel('Speedup', fontsize=10)
        
        # 设置x轴标签
        x_labels = []
        for category in categories:
            x_labels.append(f"{category}")
        
        ax1.set_xticks(x)
        ax1.set_xticklabels(x_labels, fontsize=7)
        
        ax1.yaxis.grid(True, linestyle='--', alpha=0.7, zorder=0)
        
        # 设置右轴范围
        dataset_speedups = [speedup_ratios[dataset][m] for m in categories if m in speedup_ratios[dataset]]
        max_speedup = max(dataset_speedups) if dataset_speedups else 1
        
        # 绘制加速比折线图（右轴）
        marker = ['^','v'][idx]  # 不同数据集的标记
        dataset_speedup = [speedup_ratios[dataset][method] for method in categories]
        
        # 绘制折线图
        line = ax2.plot(x, dataset_speedup,
                label=f'Speedup', linewidth=1, marker='v',markersize=4,
                color=color_list[-1])

        # 在加速比不为1的点上添加文本
        for j, (x_pos, speedup) in enumerate(zip(x, dataset_speedup)):
            if speedup == 1:
                continue    # 跳过加速比=1的baseline
            # offset = max_speedup * 0.05 if speedup < max_speedup * 0.5 else -max_speedup * 0.05
            ax2.text(x_pos, speedup + max_speedup * 0.16, f'{speedup:.1f}x', 
                    ha='center', va='top', 
                    fontsize=7, zorder=200)
    
    # # 创建统一的图例
    # # 从第一个子图获取堆叠部分的图例句柄和标签
    # handles1, labels1 = axes[0].get_legend_handles_labels()
    # # 从两个子图获取加速比部分的图例句柄和标签
    # handles2 = []
    # labels2 = []
    # for idx, dataset in enumerate(datasets):
    #     # 获取加速比线的句柄（每个子图的第二条线）
    #     line_handles, line_labels = axes[idx].get_legend_handles_labels()
    #     # 加速比线是第5个元素（前面有4个堆叠部分）
    #     if len(line_handles) > 4:
    #         handles2.append(line_handles[4])
    #         labels2.append(line_labels[4])
    
    # # 合并图例，去除重复的标签
    # unique_labels = []
    # unique_handles = []
    # for handle, label in zip(handles1 + handles2, labels1 + labels2):
    #     if label not in unique_labels:
    #         unique_labels.append(label)
    #         unique_handles.append(handle)
    # 直接用第二个图的图例，所有图的图例都是一样的
    handles1, labels1 = ax1.get_legend_handles_labels()
    # 右轴图例（加速比）
    handles2, labels2 = ax2.get_legend_handles_labels()
    
    # 合并图例
    unique_handles = handles1 + handles2
    unique_labels = labels1 + labels2
    
    # 将图例放在整个图的上方中央
    fig.legend(unique_handles, unique_labels, 
               bbox_to_anchor=(0.5, 0.95), fontsize=8, 
               loc='lower center', ncol=3, 
               frameon=False)

    plt.tight_layout()
    
    # 调整子图布局，为顶部图例留出空间
    plt.subplots_adjust(top=0.85)
    
    fig.savefig(f'{pic_name}.png', dpi=300, format="png", bbox_inches='tight', pad_inches=0.02)
    fig.savefig(f'{pic_name}.pdf', dpi=300, format="pdf", bbox_inches='tight', pad_inches=0.02)
    fig.savefig(f'{pic_name}.svg', dpi=300, format="svg", bbox_inches='tight', pad_inches=0.02)
    print(f"图片已保存为文件：{pic_name}.pdf")
    
    # 打印加速比信息
    print("\n加速比信息:")
    for dataset in datasets:
        print(f"{dataset}:")
        for method in categories:
            if method in speedup_ratios[dataset]:
                print(f"  {method}: {speedup_ratios[dataset][method]:.2f}x")

def get_time_dict(file_path):
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

    time_dict = {}
    time_dict['Sample'] = epoch_mean - featcopy_mean - forward_mean - backward_mean - embupdate_mean
    time_dict['Train'] = forward_mean + backward_mean
    # time_dict['Sample'] = sample_mean
    # time_dict['Train'] = epoch_mean - featcopy_mean - sample_mean - embupdate_mean
    time_dict['Feat Retrieval'] = featcopy_mean
    time_dict['Embedding Update'] = embupdate_mean
    
    # 确保所有值都是非负的
    for key in time_dict:
        if time_dict[key] < 0:
            time_dict[key] = 0
    
    return time_dict

if __name__ == '__main__':
    dgl_me='./log/cikm/dgl_rgcn_igb-full-medium_none_64_4.log'
    dgl_mag='./log/drgnn_rgcn_mag240m_none_64_nolinear.log'
    drgnn_nocache_me='./log/cikm/drgnn_rgcn_igb-full-medium_none_64_4.log'
    drgnn_nocache_mag='./log/drgnn_rgcn_mag240m_none_64_128,8_linear.log'
    drgnn_me='./log/cikm/drgnn_rgcn_igb-full-medium_miss_penalty_64_4.log'
    drgnn_mag='./log/drgnn_rgcn_mag240m_miss_penalty_64_4.log'

    add = ['DGL', '+MR', '+MR+FC']
    data_dict = {'ME': {}, 'MAG': {}}

    # 获取ME数据集的时间分解数据
    data_dict['ME'][add[0]] = get_time_dict(dgl_me)
    data_dict['ME'][add[1]] = get_time_dict(drgnn_nocache_me)
    data_dict['ME'][add[2]] = get_time_dict(drgnn_me)

    # 获取MAG数据集的时间分解数据
    data_dict['MAG'][add[0]] = get_time_dict(dgl_mag)
    data_dict['MAG'][add[1]] = get_time_dict(drgnn_nocache_mag)
    data_dict['MAG'][add[2]] = get_time_dict(drgnn_mag)

    print("时间分解数据:")
    for dataset in data_dict:
        print(f"{dataset}:")
        for method in data_dict[dataset]:
            print(f"  {method}: {data_dict[dataset][method]}")

    # 绘制带加速比的堆叠柱状图
    draw_time_stacked_with_speedup(data_dict, add, 'ablation')