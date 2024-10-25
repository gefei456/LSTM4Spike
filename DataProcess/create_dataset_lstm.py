#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time:         16:32
# @Author:      WGF
# @File:        create_dataset.py
# @Description:

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import scipy.io as sio
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle


class CustomDataset(Dataset):
    def __init__(self, df_feature, df_pos, df_vel, df_tgpos, device):

        assert len(df_feature) == len(df_pos)
        assert len(df_feature) == len(df_vel)
        assert len(df_feature) == len(df_tgpos)

        # df_feature = df_feature.reshape(df_feature.shape[0], df_feature.shape[1] // 6, df_feature.shape[2] * 6)
        # self.df_feature = df_feature.to(torch.float32)
        # self.df_vel = df_vel.to(torch.float32)
        # self.df_pos = df_pos.to(torch.float32)
        # self.df_feature = df_feature
        # self.df_vel = df_vel
        # self.df_pos = df_pos
        # self.df_feature = torch.tensor(
        #     self.df_feature, dtype=torch.float32)
        # self.df_vel = torch.tensor(
        #     self.df_vel, dtype=torch.float32)
        # self.df_pos = torch.tensor(
        #     self.df_pos, dtype=torch.float32)
        # 不定长 list
        self.df_feature = [torch.tensor(x, dtype=torch.float32).to(device) for x in df_feature]
        self.df_vel = [torch.tensor(x, dtype=torch.float32).to(device) for x in df_vel]
        self.df_pos = [torch.tensor(x, dtype=torch.float32).to(device) for x in df_pos]
        self.df_tgpos = [torch.tensor(x, dtype=torch.float32).to(device) for x in df_tgpos]

    def __getitem__(self, index):
        sample, pos, vel, tgpos = self.df_feature[index], self.df_pos[index], self.df_vel[index], self.df_tgpos[index]
        return sample, pos, vel, tgpos

    def __len__(self):
        return len(self.df_feature)


def createDataset_trial(dict_file, device, ratio):
    # file_to_save = {"truth_traj": X_Truth, "spike_list": X_Feature}
    assert len(dict_file['spike_list']) == len(dict_file['truth_traj'])
    lst = list(range(len(dict_file['spike_list'])))
    split_index = int(len(lst) * ratio[0])  # 计算切分索引
    train_feat_list = dict_file['spike_list'][:split_index]
    test_feat_list = dict_file['spike_list'][split_index:]
    train_traj_list = dict_file['truth_traj'][:split_index]
    test_traj_list = dict_file['truth_traj'][split_index:]
    train_tgpos_list = dict_file['tgpos_list'][:split_index]
    test_tgpos_list = dict_file['tgpos_list'][split_index:]
    for i, traj in enumerate(train_traj_list):
        train_tgpos_list[i] = train_tgpos_list[i].reshape(1, -1).repeat(traj.shape[0], axis=0)
    for i, traj in enumerate(test_traj_list):
        test_tgpos_list[i] = test_tgpos_list[i].reshape(1, -1).repeat(traj.shape[0], axis=0)
    len_state = train_traj_list[0].shape[1]
    len_seq = train_feat_list[0].shape[0]
    len_ch = train_feat_list[0].shape[1]
    if len(train_feat_list[0].shape) == 3:
        len_time = train_feat_list[0].shape[2]
    else:
        len_time = 1
    train_feat = torch.tensor(np.stack(train_feat_list, axis=0).reshape(-1, len_seq, len_ch * len_time)).to(device)
    test_feat = torch.tensor(np.stack(test_feat_list, axis=0).reshape(-1, len_seq, len_ch * len_time)).to(device)
    train_traj = torch.tensor(np.stack(train_traj_list, axis=0).reshape(-1, len_seq, len_state)).to(device)
    test_traj = torch.tensor(np.stack(test_traj_list, axis=0).reshape(-1, len_seq, len_state)).to(device)
    train_tgpos = torch.tensor(np.stack(train_tgpos_list, axis=0)).to(device)
    test_tgpos = torch.tensor(np.stack(test_tgpos_list, axis=0)).to(device)
    train_dataset = CustomDataset(train_feat, train_traj[:, :, :2], train_traj[:, :, 2:], train_tgpos, device)
    test_dataset = CustomDataset(test_feat, test_traj[:, :, :2], test_traj[:, :, 2:], test_tgpos, device)

    return train_dataset, test_dataset


def createDataset_point(dict_file, ratio, device):
    # file_to_save = {"truth_traj": X_Truth, "spike_list": X_Feature}
    assert len(dict_file['spike_list']) == len(dict_file['truth_traj'])
    lst = list(range(len(dict_file['spike_list'])))  # 创建长度为1298的列表
    split_index = int(len(lst) * ratio[0])  # 计算切分索引
    train_feat_list = dict_file['spike_list'][:split_index]
    test_feat_list = dict_file['spike_list'][split_index:]
    train_traj_list = dict_file['truth_traj'][:split_index]
    test_traj_list = dict_file['truth_traj'][split_index:]
    train_tgpos_list = dict_file['tgpos_list'][:split_index]
    test_tgpos_list = dict_file['tgpos_list'][split_index:]
    for i, traj in enumerate(train_traj_list):
        train_tgpos_list[i] = train_tgpos_list[i].reshape(1, -1).repeat(traj.shape[0], axis=0)
    for i, traj in enumerate(test_traj_list):
        test_tgpos_list[i] = test_tgpos_list[i].reshape(1, -1).repeat(traj.shape[0], axis=0)
    train_feat = torch.tensor(np.concatenate(train_feat_list, axis=0)).to(device)
    test_feat = torch.tensor(np.concatenate(test_feat_list, axis=0)).to(device)
    train_traj = torch.tensor(np.concatenate(train_traj_list, axis=0)).to(device)
    test_traj = torch.tensor(np.concatenate(test_traj_list, axis=0)).to(device)
    train_tgpos = torch.tensor(np.concatenate(train_tgpos_list, axis=0)).to(device)
    test_tgpos = torch.tensor(np.concatenate(test_tgpos_list, axis=0)).to(device)
    train_dataset = CustomDataset(train_feat.transpose(1, 2), train_traj[:, :2], train_traj[:, 2:], train_tgpos, device)
    test_dataset = CustomDataset(test_feat.transpose(1, 2), test_traj[:, :2], test_traj[:, 2:], test_tgpos, device)
    return train_dataset, test_dataset


if __name__ == '__main__':
    session_name = '20230209-3'
    bin_size = 0.05
    spike_lag = -0.14
    isAlign = 'Align'
    with open(f"../{session_name}_{bin_size}_{isAlign}_{spike_lag}_ReNN.pkl", "rb") as f:
        data_RefitNN = pickle.load(f)

    train_dataset, test_dataset = createDataset(data_RefitNN)
    print('train_dataset:', train_dataset.df_feature.shape)
    print('test_dataset:', test_dataset.df_feature.shape)
