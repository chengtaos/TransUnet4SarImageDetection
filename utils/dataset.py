import os
import cv2
import numpy as np
import torch
import scipy.io as sio
from torch.utils.data import Dataset
from utils.config import cfg
import matplotlib.pyplot as plt

class MyDataset(Dataset):
    output_size = cfg.transunet.img_dim  # 输出图像大小

    def __init__(self, path, transform=None):
        super().__init__()

        self.transform = transform

        # 设置图像和掩码的文件夹路径
        img_folder = os.path.join(path, 'img')
        mask_folder = os.path.join(path, 'mask')

        self.img_paths = []
        self.mask_paths = []
        for p in os.listdir(img_folder):
            name = p.split('.')[0]  # 获取文件的主名（不含扩展名）

            # 将 .mat 文件路径添加到 img_paths 和 mask_paths 列表中
            self.img_paths.append(os.path.join(img_folder, name + '.mat'))
            self.mask_paths.append(os.path.join(mask_folder, name + '.mat'))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 获取图像和掩码的 .mat 文件路径
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]

        # 读取 .mat 文件
        img_data = sio.loadmat(img_path)  # 读取图像数据，假设存储在 'img' 键中
        mask_data = sio.loadmat(mask_path)  # 读取掩码数据，假设存储在 'mask' 键中
        # print(img_data.keys())
        # print(mask_data.keys())
        # 访问 img 和 mask 数据，根据你的实际键名称替换 'img' 和 'mask'
        MyImg = img_data['img']  # 假设 img 是复数数据
        MyMask = mask_data['mask']  # 掩码是单通道实数数据

        MyImg=(MyImg-np.min(np.abs(MyImg)))/(np.max(np.abs(MyImg))-np.min(np.abs(MyImg)))# 根据复数图像的最大值进行归一化
        # 将复数图像分为实部和虚部，分别作为两个通道
        img_real = np.real(MyImg)  # 实部
        img_imag = np.imag(MyImg)  # 虚部
        MyImg = np.stack((img_real, img_imag), axis=-1)  # 将实部和虚部沿最后一个维度（通道维度）叠加，形成二通道数据

        # 调整图像和掩码大小
        MyImg = cv2.resize(MyImg, (self.output_size, self.output_size))
        MyMask = cv2.resize(MyMask, (self.output_size, self.output_size), interpolation=cv2.INTER_NEAREST)

        # 如果掩码是二维数据，扩展为三维
        if MyMask.ndim == 2:
            MyMask = np.expand_dims(MyMask, axis=-1)

        sample = {'img': MyImg, 'mask': MyMask}

        # 如果有数据增强（transform），则应用
        if self.transform:
            sample = self.transform(sample)

        MyImg, MyMask = sample['img'], sample['mask']

        # 将图像和掩码数据归一化到 [0, 1] 范围
        MyImg = MyImg.transpose((2, 0, 1))  # 调整维度为 (C, H, W)
        MyImg = torch.from_numpy(MyImg.astype('float32'))

        MyMask = MyMask.transpose((2, 0, 1))  # 调整维度为 (C, H, W)
        MyMask = torch.from_numpy(MyMask.astype('float32'))

        return {'img': MyImg, 'mask': MyMask}

    def __len__(self):
        return len(self.img_paths)


if __name__ == '__main__':
    mydataset = MyDataset('D:/code/Data/MyData/train')  # 替换为你的数据路径

    for sample in mydataset:
        # 打印图像和掩码的形状
        print(sample['img'].shape)  # 打印图像的维度 (2, H, W) -> 两个通道 (实部和虚部)
        print(sample['mask'].shape)  # 打印掩码的维度 (1, H, W) -> 单通道

        # 将 img 转换为适合计算幅值的 NumPy 数组
        img = sample['img'].numpy()  # 转换为 NumPy 数组
        img_real = img[0, :, :]  # 实部
        img_imag = img[1, :, :]  # 虚部

        # 计算幅值 sqrt(real^2 + imag^2)
        magnitude = np.sqrt(np.square(img_real) + np.square(img_imag))
        # 将 mask 转换为适合显示的 NumPy 数组
        mask = sample['mask'].numpy()  # 转换为 NumPy 数组
        mask = np.squeeze(mask, axis=0)  # [1, H, W] -> [H, W]

        # 使用 matplotlib 显示图像的幅值和掩码
        plt.figure(figsize=(10, 5))

        # 显示幅值图像
        plt.subplot(1, 2, 1)
        plt.imshow(magnitude)
        plt.title('Magnitude Image')
        plt.colorbar()

        # 显示掩码
        plt.subplot(1, 2, 2)
        plt.imshow(mask)
        plt.title('Mask Image')
        plt.colorbar()

        # 展示图像
        plt.show()

        break  # 只显示一个样本后退出