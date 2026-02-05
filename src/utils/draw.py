import torch
import matplotlib.pyplot as plt
import numpy as np

draw_linestyle={
    'DGL':'--',
    'DRGNN':'-',
    'Heta':'-.',
}

draw_color={
    'DGL':'#8ECFC9',
    'MeCache':'#FA7F6F',
    'DRGNN':'#FA7F6F',
    'Heta':'#FFBE7A',
}

color_list=['#8ECFC9','#FA7F6F','#FFBE7A','#82B0D2','#BEB8DC','#E7DAD2','#999999']

def draw_cdf(tensor_dict, file_path):
    """
    根据输入的 {str: torch.tensor} 绘制曲线图。

    参数:
        tensor_dict (dict): 键为字符串，值为 torch.tensor 的字典
    """
    # 创建一个新的图形
    plt.figure(figsize=(10, 6))
    
    # 遍历字典中的每个条目
    for label, tensor in tensor_dict.items():
        # 将张量转换为 NumPy 数组
        y = tensor.numpy()
        
        # 绘制曲线
        plt.plot(y, label=label)
    
    # 添加图例
    plt.legend()
    
    # 添加标题和轴标签
    plt.title(f"{file_path.split('/')[-1]}")
    plt.xlabel("similarity")
    plt.ylabel("ratio")
    
    # 显示网格
    plt.grid(True)
    plt.savefig(file_path)
    print(f"图片已保存为文件：{file_path}")

    plt.close()

def draw_acc_epoch(test_acc_dict, epochs, file_path):
    # 生成 x 轴的刻度（假设每个列表的长度相同）
    for label, tensor in test_acc_dict.items():
        epochs=min(len(tensor),epochs)
        plt.plot(range(1, epochs + 1), tensor[:epochs], label=label)

    # 添加标题和标签
    plt.title('Test Accuracy Comparison', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Test Accuracy', fontsize=12)
    plt.ylim(60,67)

    # 添加图例
    plt.legend()
    
    # 显示网格
    plt.grid(True)
    plt.savefig(file_path)
    print(f"图片已保存为文件：{file_path}")

def draw_acc_time(test_acc_dict, epochs, file_path):
    # 绘制折线图
    plt.figure(figsize=(10, 6))  # 设置图形大小

    # 添加标题和标签
    plt.title('Test Accuracy vs. Time', fontsize=14)
    plt.xlabel('Cumulative Time (s)', fontsize=12)
    plt.ylabel('Test Accuracy', fontsize=12)

    for label, tensor in test_acc_dict.items():
        epochs=min(len(tensor[0]),epochs)
        test_acc_list = tensor[0][:epochs]
        epoch_time_values = tensor[1][:epochs]
        cumulative_time = [sum(epoch_time_values[:i+1]) for i in range(len(epoch_time_values))]

        plt.plot(cumulative_time, test_acc_list, label=label)

    # 添加图例
    plt.legend()

    # 添加网格
    plt.grid(True)

    # 显示图形
    plt.show()
    plt.savefig(file_path)
    print(f"图片已保存为文件：{file_path}")

if __name__ == "__main__":
    data = {
        "A": torch.rand(101),  # 随机张量 (0, 1) 范围
        "B": torch.rand(101),
        "C": torch.rand(101)
    }
    
    # 绘制曲线
    draw_cdf(data, './tmp.png')
