import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, file_path=None, num_samples=None):
        """
        初始化数据集，可以从文件加载数据或随机生成数据
        :param file_path: 数据文件路径 (如果存在，则从文件加载数据)
        :param num_samples: 样本数量 (仅在未提供文件路径时随机生成数据)
        """
        if file_path and os.path.exists(file_path):
            # 从文件中加载数据
            print(f"Loading data from {file_path}...")
            data = np.load(file_path)
            self.img_data = data['img']
            self.state_data = data['state']
            self.action_data = data['action']
        elif num_samples:
            # 随机生成数据
            print("Generating random data...")
            self.img_data = np.random.rand(num_samples, 8, 256, 256).astype(np.float32)  # img: (8, 256, 256)
            self.state_data = np.random.rand(num_samples, 16).astype(np.float32)       # state: (2, 8)
            self.action_data = np.random.rand(num_samples, 256).astype(np.float32)        # action: (4,)
        else:
            raise ValueError("Either file_path or num_samples must be provided.")

    def __len__(self):
        """
        返回数据集的样本数量
        """
        return len(self.img_data)

    def __getitem__(self, index):
        """
        根据索引返回一个数据样本
        :param index: 数据样本的索引
        :return: 一个包含 img, state, action 的字典
        """
        img = torch.tensor(self.img_data[index])      # 转换为 torch.Tensor
        state = torch.tensor(self.state_data[index])  # 转换为 torch.Tensor
        action = torch.tensor(self.action_data[index])# 转换为 torch.Tensor

        return {"img": img, "state": state, "action": action}

    def save_to_file(self, file_path):
        """
        将数据保存到文件中
        :param file_path: 保存文件路径
        """
        print(f"Saving data to {file_path}...")
        np.savez(file_path, img=self.img_data, state=self.state_data, action=self.action_data)
        print("Data saved successfully.")

# 创建并保存数据集
def create_and_save_dataset(file_path, num_samples=100):
    """
    创建一个随机数据集并保存到文件
    :param file_path: 保存文件路径
    :param num_samples: 样本数量
    """
    dataset = CustomDataset(num_samples=num_samples)
    dataset.save_to_file(file_path)

# 测试加载和使用数据集
def load_and_test_dataset(file_path, batch_size=4):
    """
    从文件加载数据集并测试 DataLoader
    :param file_path: 数据文件路径
    :param batch_size: DataLoader 的批次大小
    """
    dataset = CustomDataset(file_path=file_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for batch_idx, batch in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}:")
        print(f"  img shape: {batch['img'].shape}")      # 期望 (batch_size, 8, 256, 256)
        print(f"  state shape: {batch['state'].shape}")  # 期望 (batch_size, 2, 8)
        print(f"  state: {batch['state']}")  
        print(f"  action shape: {batch['action'].shape}")# 期望 (batch_size, 4)
        break  # 只打印第一个批次

    return dataloader

# # 文件路径
# data_file_path = "custom_dataset.npz"

# # 创建并保存数据集
# create_and_save_dataset(data_file_path, num_samples=100)

# # 从文件加载并测试数据集
# load_and_test_dataset(data_file_path, batch_size=4)