#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time:         16:30
# @Author:      WGF
# @File:        main_gf003.py
# @Description: point->point

import torch
from torch.optim import lr_scheduler
import torch.optim as optim
import numpy as np
import torch.nn as nn
from DataProcess.create_dataset_seq2seq import createDataset_point
import pickle
import time
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from pytorch_model_summary import summary
import sklearn.decomposition as skd
from torch.utils.tensorboard import SummaryWriter
import shutil
from Module.seq2seq_offical import GRU_Seq2Seq
from DataProcess._utils import R2_score, final_dist
from _utils import calculate_mean_variance
import scipy.io as scio
import hdf5storage as hdf5
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
import random
import math


def train_GRU(dataloader, model, optimizer, loss_fn, label_state):
    size = len(dataloader.dataset)
    num_batches = 0
    total_loss = 0
    model.train()
    for i, (X, pos, vel) in enumerate(dataloader):
        optimizer.zero_grad()
        pred = model(X)[:, -1, :]
        if label_state == 'vel':
            loss = loss_fn(pred, vel)
        elif label_state == 'pos':
            loss = loss_fn(pred, pos)
        else:
            loss = loss_fn(pred, pos + vel)
        # loss = loss_fn(
        #     decoder_outputs.view(-1, decoder_outputs.size(-1)),
        #     vel.view(-1, decoder_outputs.size(-1))
        # )
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
        if i % 5 == 0:
            loss, current = loss.item(), (i + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    if num_batches != 0:
        train_loss = total_loss / num_batches
    else:
        train_loss = total_loss
    return train_loss


def test_GRU(dataloader, model, loss_fn, label_state):
    model.eval()
    # size = len(dataloader.dataset)
    num_batches = 0
    # batch_size = dataloader.batch_size
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for i, (X, pos, vel) in enumerate(dataloader):
            pred = model(X)[:, -1, :]
            if label_state == 'vel':
                loss = loss_fn(pred, vel)
            elif label_state == 'pos':
                loss = loss_fn(pred, pos)
            else:
                loss = loss_fn(pred, pos + vel)
            test_loss += loss.item()
            num_batches += 1
    if num_batches != 0:
        test_loss /= num_batches
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")
    return test_loss


def train_seq2seq(train_dataloader, test_dataloader, model, optimizer, loss_fn, epochs, best_train_loss, scheduler):
    for t in range(epochs):
        scheduler.step()
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loss = train_GRU(train_dataloader, model, optimizer, loss_fn, label_state)
        val_loss = test_GRU(test_dataloader, model, loss_fn, label_state)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        writer.add_scalars('Loss', {'trainLoss': train_loss, 'valLoss': val_loss}, t)
        # early stop
        # num_bad_epochs = 0
        if (best_train_loss - train_loss) - (1e-2 * train_loss) > 0:
            best_train_loss = train_loss
            num_bad_epochs = 0
        else:
            num_bad_epochs += 1
            print("num_bad_epochs:", num_bad_epochs)
        if num_bad_epochs >= patience:
            break
    print("Done!")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    isPkl = False
    lr = 1e-3
    inputSize = 256
    PCA_ncomp = inputSize
    hiddenSize = 64
    label_state = 'vel'
    if label_state == 'vel' or 'pos':
        numStates = 2
    else:
        numStates = 4
    batchSize = 1024
    encoder_layers = 2
    decoder_layers = 2
    session_name = 'qianqian_bj-co-mind-control_24-07-24_7_SBP_3_bip'
    bin_size = 0.05
    useBottleneck = False
    only4Test = True
    time_str = 'GRU_2024-08-17--08_19_15_0.05_3_-0.0_noAlign_13'
    # time_str = 'good_GRU_2023-09-24--16_50_47'
    # time_str = 'good_GRU_2023-09-22--17_50_33'
    seed = 13
    torch.manual_seed(seed)
    spike_lag = -0.0
    isAlign = 'noAlign'
    isMerged = False
    isPCA = False
    epochs = 10000
    patience = 500
    lr_step = 1500
    path_head = 'E:/Data/PKL/'
    # path_head = './Data/'
    pre_bin_num = 3
    drop_out = 0.8

    noisy_chans = [0, 1, 2, 3, 5, 7, 8, 9, 16, 59, 89, 93, 99, 122, 124, 126, 127, 134, 142, 149, 177, 179, 183, 185,
                   187, 189, 191, 232, 240, 252]

    if isPkl:
        # load Pkl
        if isMerged:
            session_name1 = '20230209-2'
            session_name2 = '20230209-3'
            with open(path_head + f'{session_name1}_{bin_size}_{isAlign}_{spike_lag}_{pre_bin_num}_ReNN.pkl', 'rb') as f1:
                data_RefitNN1 = pickle.load(f1)
            with open(path_head + f'{session_name2}_{bin_size}_{isAlign}_{spike_lag}_{pre_bin_num}_ReNN.pkl', 'rb') as f2:
                data_RefitNN2 = pickle.load(f2)

            data_spike_merge = data_RefitNN1['spike_list'] + data_RefitNN2['spike_list']
            data_label_merge = data_RefitNN1['label_list'] + data_RefitNN2['label_list']
            data_traj_merge = data_RefitNN1['truth_traj'] + data_RefitNN2['truth_traj']
            if isAlign == 'Align':
                trial_length = np.array([x.shape[0] for x in data_spike_merge])
                trial_max_length = np.max(trial_length)
                data_spike_merge = np.array(
                    [np.pad(x, ((trial_max_length - x.shape[0], 0), (0, 0), (0, 0)), 'constant') for x in data_spike_merge])
                data_traj_merge = np.array(
                    [np.pad(x, ((trial_max_length - x.shape[0], 0), (0, 0)), 'constant') for x in data_traj_merge])
            data_RefitNN = {'truth_traj': list(data_traj_merge), 'spike_list': data_spike_merge,
                            'label_list': data_label_merge}
            session_name = session_name1 + '_' + session_name2
        else:
            with open(path_head + f'{session_name}_{bin_size}_{isAlign}_{spike_lag}_{pre_bin_num}_ReNN.pkl', 'rb') as f:
                data_RefitNN = pickle.load(f)
    else:
        # load mat
        file_path = path_head + session_name + '.mat'
        try:
            # 尝试使用 scipy.io.loadmat 读取文件（适用于7.3以下版本）
            data_mat = scio.loadmat(file_path)  # 从文件中加载数据
        except NotImplementedError:
            # 如果 scipy.io.loadmat 不支持该文件，尝试使用 hdf5storage 读取（适用于7.3及以上版本）
            data_mat = hdf5.loadmat(file_path)  # 从文件中加载数据
        label_list = np.ravel(data_mat['label_list'])
        spike_list = np.ravel(data_mat['spike_list'])
        # 噪声通道清 0
        for i in range(len(spike_list)):
            spike_list[i][:, noisy_chans] = 0

        tgpos_list = np.ravel(data_mat['tgpos_list'])
        truth_traj = np.ravel(data_mat['truth_traj'])
        data_RefitNN = {'truth_traj': list(truth_traj), 'spike_list': spike_list,
                        'label_list': label_list}

    # shuffle data
    shuffle_seed = 13
    random.seed(shuffle_seed)
    shuffle_index = np.arange(0, len(data_RefitNN['label_list']))
    random.shuffle(shuffle_index)
    for key in data_RefitNN.keys():
        data_RefitNN[key] = [data_RefitNN[key][i] for i in shuffle_index]

    local_time = time.strftime("%Y-%m-%d--%H_%M_%S", time.localtime())
    print(local_time)
    exp_path = f'./{session_name}_EXP/GRU_{local_time}_{bin_size}_{pre_bin_num}_{spike_lag}_{isAlign}_{seed}/'
    writer = SummaryWriter(exp_path + 'loss')

    if isPCA:
        ##############################  PCA  #################################
        pca = skd.PCA(n_components=PCA_ncomp)
        orig_index_list = np.cumsum([x.shape[0] for x in data_spike_merge])
        orig_index_list = np.insert(orig_index_list, 0, 0)
        stacked_data_spike_merge = np.concatenate(data_spike_merge, axis=0)
        stacked_data_spike_merge = stacked_data_spike_merge.reshape(len(stacked_data_spike_merge), -1)
        pca.fit(stacked_data_spike_merge)
        print('total explained variance ratio: ', np.cumsum(pca.explained_variance_ratio_)[-1])
        data_spike_merge_extr = pca.transform(stacked_data_spike_merge)
        data_RefitNN['spike_list'] = [data_spike_merge_extr[orig_index_list[i]:orig_index_list[i + 1], :]
                                      for i, _ in enumerate(orig_index_list[:-1])]
        ######################################################################

    if not only4Test:
        # define the model, loss function and optimizer
        # model = BasicLSTM(input_size=inputSize, hidden_size=hiddenSize, num_states=numStates).to(device)
        model = GRU_Seq2Seq(input_size=inputSize, hidden_size=hiddenSize, output_size=numStates,
                            n_encoder_layers=encoder_layers, n_decoder_layers=decoder_layers, device=device,
                            dropout_p=drop_out).to(device)
        # criterion = torch.nn.HuberLoss(reduction='mean', delta=40)
        # criterion = nn.SmoothL1Loss(beta=0.5)
        # criterion = nn.L1Loss()
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
        # optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
        # optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)
        # optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        # 定义学习率衰减策略
        scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=0.7)  # 每10个周期将学习率减小为原来的一半

        # 切分比例，第一部分占80%，第二部分占20%
        ratio = [0.8, 0.2]
        train_dataset, test_dataset = createDataset_point(data_RefitNN, device, ratio)
        train_dataloader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True, drop_last=False)
        test_dataloader = DataLoader(test_dataset, batch_size=batchSize, shuffle=True, drop_last=False)

        best_train_loss = float('inf')
        num_bad_epochs = 0
        train_loss_list = []
        val_loss_list = []
        # train the model
        train_seq2seq(train_dataloader, test_dataloader, model, optimizer, loss_fn, epochs, best_train_loss, scheduler)

        # plot
        plt.figure(1)
        plt.plot(range(len(train_loss_list)), train_loss_list, 'r', '--')
        plt.plot(range(len(val_loss_list)), val_loss_list, 'b', '--')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()
        # save loss to local
        writer.close()
        model_path = exp_path + 'GRU-Model.pth'
        state_dict_path = exp_path + 'GRU-state_dict.pt'
        # torch.save(model.state_dict(), model_path)
        torch.save(model, model_path)
        torch.save({
            'learning_rate': lr,
            'inputSize': inputSize,
            'hiddenSize': hiddenSize,
            'batchSize': batchSize,
            'encoderLayers': encoder_layers,
            'decoderLayers': decoder_layers,
            'bin_size': bin_size,
            'label_state': label_state,
            'patience': patience,
            'isAlign': isAlign,
            'drop_out': drop_out,
            # 'optimizer_param_groups': optimizer.param_groups,
            'loss_type': type(loss_fn).__name__,
            'optimizer_type': type(optimizer).__name__
        }, state_dict_path)
    else:
        if isPkl:
            model_path = '20230209-2_20230209-3_EXP/' + time_str + '/GRU-Model.pth'
            state_dict_path = '20230209-2_20230209-3_EXP/' + time_str + '/GRU-state_dict.pt'
            shutil.rmtree(exp_path)
        else:
            model_path = session_name + '_EXP/' + time_str + '/GRU-Model.pth'
            state_dict_path = session_name + '_EXP/' + time_str + '/GRU-state_dict.pt'
            shutil.rmtree(exp_path)

    model = torch.load(model_path).to(device)
    # print(model.state_dict())
    for name, parameters in model.named_parameters():
        print(name, ':', parameters.size())
    checkpoints = torch.load(state_dict_path)
    print('checkpoints:', checkpoints)
    model.eval()
    model.requires_grad_(False)
    # print(model.__module__)
    print(model)

    # trial_num = len(data_RefitNN['truth_traj'])
    # trial_sec_start = trial_num - 40
    # trial_sec_end = trial_num - 0
    # test_traj_list = data_RefitNN['truth_traj'][trial_sec_start:trial_sec_end]
    # test_feat_list = data_RefitNN['spike_list'][trial_sec_start:trial_sec_end]
    # test_label_list = data_RefitNN['label_list'][trial_sec_start:trial_sec_end]

    # plot
    plot_method = 2
    if plot_method == 1:
        trial_every_dir = 1
        trial_end_index = 5
        trial_num = len(data_RefitNN['truth_traj'])
        print('total trial num:', trial_num)
        # trial_sec_end = 32
        # trial_sec_start = 0
        trial_sec_end = trial_num - trial_end_index
        trial_sec_start = trial_num - (trial_every_dir * 6 + trial_end_index)
        test_traj_list = data_RefitNN['truth_traj'][trial_sec_start:trial_sec_end]
        test_feat_list = data_RefitNN['spike_list'][trial_sec_start:trial_sec_end]
        test_label_list = data_RefitNN['label_list'][trial_sec_start:trial_sec_end]
    elif plot_method == 2:
        ratio = 0.8
        trial_every_dir = 5
        trial_num = len(data_RefitNN['truth_traj'])
        print('total trial num:', trial_num)
        label_count = np.zeros(8)
        label_len = len(data_RefitNN['label_list'])
        start_index = math.floor(label_len * ratio)
        label_mask = np.full((label_len,), False, dtype=bool)
        for i, label in enumerate(data_RefitNN['label_list'][start_index:],
                                  start=start_index):
            label_count[int(label) - 1] += 1
            if label_count[int(label) - 1] < trial_every_dir + 1:
                label_mask[i] = True
        mask_index = np.where(label_mask)[0]
        test_traj_list = [data_RefitNN['truth_traj'][_mask] for _mask in mask_index]
        test_feat_list = [data_RefitNN['spike_list'][_mask] for _mask in mask_index]
        test_label_list = [data_RefitNN['label_list'][_mask] for _mask in mask_index]

    plt.figure(3)
    axis_range = 12
    colorlist = list(matplotlib.colors.TABLEAU_COLORS)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    linetypelist = ['-', '--', '-.', ':']
    markerlist = ['.', ',', 'o', '2', '1']
    R2_Mat = [[] for _ in range(8)]
    FD_Mat = [[] for _ in range(8)]
    CC_Mat = [[] for _ in range(8)]
    TD_Mat = [[] for _ in range(8)]
    R2_Mat = [[] for _ in range(8)]
    dist_Mat = [[] for _ in range(8)]
    for i, feat in enumerate(test_feat_list):
        X = torch.tensor(feat.transpose(0, 2, 1), dtype=torch.float32).to(device)
        pred_vel = torch.tensor([])
        X_feat = torch.tensor([]).to(device)
        for j, xx in enumerate(X):
            # stime4test = time.time_ns()
            pred_vel = torch.cat([pred_vel, model(xx.unsqueeze(0))[:, -1, :].to('cpu')], dim=0)
            # print("运算所需时间", (time.time_ns() - stime4test) / 1e9)
        if label_state == 'vel':
            true_traj = np.cumsum(test_traj_list[i][:, 2:], axis=0) * bin_size
            pred_traj = np.cumsum(pred_vel.numpy(), axis=0) * bin_size
        elif label_state == 'pos':
            true_traj = test_traj_list[i][:, :2]
            pred_traj = pred_vel.numpy()
        else:
            # true_traj = np.cumsum(test_traj_list[i][:, 2:], axis=0) * bin_size
            # pred_traj = np.cumsum(pred_vel.numpy(), axis=0) * bin_size
            true_traj = test_traj_list[i][:, :2]
            pred_traj = pred_vel.numpy()
        # print('pretraj:', pred_traj)
        R2_Mat[int(test_label_list[i]) - 1].append(R2_score(true_traj, pred_traj))
        FD_Mat[int(test_label_list[i]) - 1].append(final_dist(true_traj, pred_traj))
        # CC_Mat[int(test_label_list[i]) - 1].append(np.sqrt(R2_score(true_traj, pred_traj)))
        CC_Mat[int(test_label_list[i]) - 1].append(np.corrcoef(true_traj.flatten(), pred_traj.flatten())[0, 1])
        TD_Mat[int(test_label_list[i]) - 1].append(final_dist(true_traj, pred_traj))
        plt.plot(true_traj[:, 0], true_traj[:, 1], colorlist[int(test_label_list[i]) - 1], linestyle='-',
                 linewidth=0.7)
        plt.plot(pred_traj[:, 0], pred_traj[:, 1], colorlist[int(test_label_list[i]) - 1], linestyle='--',
                 linewidth=0.7)
    plt.xlim(-axis_range, axis_range)
    plt.ylim(-axis_range, axis_range)
    x_scale = 8
    y_scale = 8
    x_pos = [x_scale, x_scale / 1.414, 0, -x_scale / 1.414, -x_scale, -x_scale / 1.414, 0, x_scale / 1.414]
    y_pos = [0, y_scale / 1.414, y_scale, y_scale / 1.414, 0, -y_scale / 1.414, -y_scale, -y_scale / 1.414]
    r2_all = 0
    fd_all = 0
    for i in range(8):
        r2 = sum(R2_Mat[i]) / 5
        fd = sum(FD_Mat[i]) / 5
        plt.text(x_pos[i], y_pos[i], f"{r2:.2f} | {fd: .2f}")
        r2_all += r2
        fd_all += fd
    r2_all /= 8
    fd_all /= 8
    # R2_mean, R2_std = calculate_mean_variance(np.array(R2_Mat))
    # FD_mean, FD_std = calculate_mean_variance(np.array(FD_Mat))
    # CC_mean, CC_std = calculate_mean_variance(np.array(CC_Mat))
    # TD_mean, TD_std = calculate_mean_variance(np.array(TD_Mat))
    # print('R2_mean:', R2_mean)
    # print('R2_std:', R2_std)
    # print('FD_mean:', FD_mean)
    # print('FD_std:', FD_std)
    # print('CC_mean:', CC_mean)
    # print('CC_std:', CC_std)
    # print('TD_mean:', TD_mean)
    # print('TD_std:', TD_std)
    plt.title(f'GRU-{session_name}-{local_time}-r2:{r2_all: .3f}-dist:{fd_all: .3f}')
    plt.show()
