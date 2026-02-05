import numpy
import sys
import os
import matplotlib.pyplot as plt

project_root = '/gf3/home/jgqj/test_code/hydro'
sys.path.append(project_root)

from src import extract_key_word_from_file,color_list

def draw(data_dict, file_path):
    # fig=plt.figure(figsize=(4.5, 2.5))
    patterns = ['///', '\\\\\\',  'xxx', '|||','+++']
    fig, ax = plt.subplots(figsize=(4.25, 1.5))
    # 添加标题和标签
    plt.ylabel('Data Trans. Time (s)', fontsize=9)

    bar_width = 0.6
    categories = [label for label, tensor in data_dict.items()]
    featcopy = [tensor[0] for label, tensor in data_dict.items()]
    update = [tensor[1] for label, tensor in data_dict.items()]
    speedup = [tensor[2] for label, tensor in data_dict.items()]
    print(categories)
    print(featcopy)
    print(update)
    bars1 = ax.bar(categories, featcopy, bar_width, label='Feat retrieval',
                    color=color_list[0], zorder=100,hatch=patterns[0],edgecolor='black')
    bars2 = ax.bar(categories, update, bar_width, label='Embedding update'
                   , color=color_list[1], bottom=featcopy, zorder=100,hatch=patterns[1],edgecolor='black')

    # 在每个柱子上方添加 speedup 文字
    for i, bar in enumerate(bars1):
        # 计算柱子的 top 位置
        height = bar.get_height()
        total_height = height + bars2[i].get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, total_height, f'{speedup[i]:.2f}x', ha='center', va='bottom', fontsize=7)

    plt.ylim(0,170)
    plt.yticks(list(numpy.linspace(0,160,num=3)))

    # 添加图例
    # plt.legend(bbox_to_anchor=(0.5, 1.15),fontsize=7,loc='upper center', ncol=4,frameon=False)
    plt.legend(fontsize=7,loc='upper right', ncol=1,frameon=True)
    plt.tight_layout()
    plt.xticks(rotation=15,fontsize=9)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7,zorder=0)

    fig.savefig(f'{file_path}.pdf',dpi=300,format="pdf",bbox_inches='tight',pad_inches=0.02)
    fig.savefig(f'{file_path}.png',dpi=300,format="png",bbox_inches='tight',pad_inches=0.02)
    fig.savefig(f'{file_path}.svg',dpi=300,format="svg",bbox_inches='tight',pad_inches=0.02)
    print(f"图片已保存为文件：{f'{file_path}.pdf'}")



if __name__ == '__main__':
    dgl='./log/drgnn_rgcn_mag240m_none_64_2.log'
    drgnn='./log/drgnn_rgcn_mag240m_miss_penalty_64_2_drgnn.log'
    # drgnn_worst='./log/small/drgnn_rgcn_igb-full-small_miss_penalty_64_4_2.log'
    drgnn_readonly='./log/drgnn_rgcn_mag240m_miss_penalty_64_2_R.log'
    drgnn_learnable='./log/drgnn_rgcn_mag240m_miss_penalty_64_2_E.log'
    drgnn_optimizer='./log/drgnn_rgcn_mag240m_miss_penalty_64_2_O.log'
    drgnn_embedding_and_optimizer='./log/drgnn_rgcn_mag240m_miss_penalty_64_2_EO.log'
    # heta='./log/small/drgnn_rgcn_igb-full-small_miss_penalty_64_4_6.log'

    title_dict={
        dgl:"DGL+MR",
        drgnn:"MeCache",
        # drgnn_worst:"MeCache-W",          # MeCache-worst
        drgnn_readonly:"MeCache-R",    # MeCache-readonly
        drgnn_learnable:"MeCache-E",  # MeCache-embeddings
        drgnn_optimizer:"MeCache-O",  # MeCache-optimizer
        drgnn_embedding_and_optimizer:"MeCache-EO" # MeCache-embeddings and optimizer
        # heta:"Heta"
    }

    data_dict={}
    _max=0
    for file_name in [dgl,drgnn_readonly,drgnn_learnable,drgnn_optimizer,drgnn_embedding_and_optimizer,drgnn]:
        featcopy_values=extract_key_word_from_file(file_name, ', feat_copy:')
        update_values=extract_key_word_from_file(file_name, ', emb update:')

        update_values=numpy.mean(update_values)
        featcopy_values=numpy.mean(featcopy_values)
        # update_values=update_values[-1]
        # featcopy_values=featcopy_values[-1]
        _sum=featcopy_values+update_values
        if _sum>_max:
            _max=_sum
        speedup=_max/_sum
        print(title_dict[file_name], f"featcopy_values: {featcopy_values}, update_values: {update_values}, speedup: {speedup}")

        data_dict[title_dict[file_name]]=[featcopy_values,update_values,speedup]

    draw(data_dict,'cost_model_efficiency')