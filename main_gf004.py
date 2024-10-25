#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time:         21:02
# @Author:      WGF
# @File:        main_gf004.py
# @Description: point->point + tgpos + teaching force

import torch
import torch.optim as optim
import numpy as np
import torch.nn as nn
from DataProcess.create_dataset_seq2seq import createDataset_tgpos
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
from DataProcess._utils import R2_score, final_dist, refitVelocity


def train_GRU(dataloader, model, optimizer, loss_fn, label_state):
    size = len(dataloader.dataset)
    num_batches = 0
    total_loss = 0
    model.train()
    for i, (X, pos, vel, tpos) in enumerate(dataloader):
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
        for i, (X, pos, vel, tpos) in enumerate(dataloader):
            pred = model(X)[:,-1,:]
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


def train_seq2seq(train_dataloader, test_dataloader, model, optimizer, loss_fn, epochs, best_train_loss):
    for t in range(epochs):
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
            return best_train_loss, val_loss
    print("train model Done!")
    return best_train_loss, val_loss


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    lr = 1e-4
    inputSize = 256
    PCA_ncomp = inputSize
    hiddenSize = 64
    label_state = 'vel'
    if label_state == 'vel' or 'pos':
        numStates = 2
    else:
        numStates = 4
    batchSize = 512
    encoder_layers = 2
    decoder_layers = 2
    session_name = '20230209-3'
    bin_size = 0.02
    useBottleneck = False
    only4Test = True
    time_str = 'good_GRU_2023-09-22--17_50_33'
    seed = 13
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    spike_lag = -0.0
    isAlign = 'noAlign'
    isMerged = True
    isPCA = False
    epochs = 20000
    patience = 200
    path_head = './Data/'
    pre_bin_num = 7
    drop_out = 0.8
    alpha = 1.0

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
        data_tpos_merge = data_RefitNN1['tgpos_list'] + data_RefitNN2['tgpos_list']
        if isAlign == 'Align':
            trial_length = np.array([x.shape[0] for x in data_spike_merge])
            trial_max_length = np.max(trial_length)
            data_spike_merge = np.array([np.pad(x, ((trial_max_length - x.shape[0], 0), (0, 0), (0, 0)), 'constant')
                                         for x in data_spike_merge])
            data_traj_merge = np.array([np.pad(x, ((trial_max_length - x.shape[0], 0), (0, 0)), 'constant')
                                        for x in data_traj_merge])
        data_RefitNN = {'truth_traj': list(data_traj_merge), 'spike_list': data_spike_merge,
                        'label_list': data_label_merge, 'tgpos_list': data_tpos_merge}
        session_name = session_name1 + '_' + session_name2
    else:
        with open(path_head + f'{session_name}_{bin_size}_{isAlign}_{spike_lag}_{pre_bin_num}_ReNN.pkl', 'rb') as f:
            data_RefitNN = pickle.load(f)
    local_time = time.strftime("%Y-%m-%d--%H_%M_%S", time.localtime())
    print(local_time)
    exp_path = f'./{session_name}_EXP/GRU_refit_{local_time}_{bin_size}_{pre_bin_num}_{spike_lag}_{isAlign}_{seed}/'
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
        data_RefitNN['spike_list'] = [data_spike_merge_extr[orig_index_list[i]:orig_index_list[i+1], :]
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

        # 切分比例，第一部分占80%，第二部分占20%
        ratio = [0.5, 0.5]
        train_dataset, test_dataset = createDataset_tgpos(data_RefitNN, device, ratio)
        train_dataloader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True, drop_last=False)
        test_dataloader = DataLoader(test_dataset, batch_size=batchSize, shuffle=True, drop_last=False)

        best_train_loss = float('inf')
        num_bad_epochs = 0
        train_loss_list = []
        val_loss_list = []
        # train the model
        best_train_loss, val_loss = train_seq2seq(train_dataloader, test_dataloader, model, optimizer, loss_fn, epochs, best_train_loss)

        #plot
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
            'best_train_loss': best_train_loss,
            'final_val_loss': val_loss,
            # 'optimizer_param_groups': optimizer.param_groups,
            'loss_type': type(loss_fn).__name__,
            'optimizer_type': type(optimizer).__name__
        }, state_dict_path)
    else:
        model_path = '20230209-2_20230209-3_EXP/' + time_str + '/GRU-Model.pth'
        state_dict_path = '20230209-2_20230209-3_EXP/' + time_str + '/GRU-state_dict.pt'
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
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params}")

    trial_num = len(data_RefitNN['truth_traj'])
    trial_sec_start = trial_num - 40
    trial_sec_end = trial_num - 0
    test_traj_list = data_RefitNN['truth_traj'][trial_sec_start:trial_sec_end]
    test_feat_list = data_RefitNN['spike_list'][trial_sec_start:trial_sec_end]
    test_label_list = data_RefitNN['label_list'][trial_sec_start:trial_sec_end]

    # plot
    plt.figure(2)
    colorlist = list(matplotlib.colors.TABLEAU_COLORS)
    linetypelist = ['-', '--', '-.', ':']
    markerlist = ['.', ',', 'o', '2', '1']
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
        dist_Mat[int(test_label_list[i]) - 1].append(final_dist(true_traj, pred_traj))
        plt.plot(true_traj[:, 0], true_traj[:, 1], colorlist[int(test_label_list[i]) - 1], linestyle='-', linewidth=0.7)
        plt.plot(pred_traj[:, 0], pred_traj[:, 1], colorlist[int(test_label_list[i]) - 1], linestyle='--', linewidth=0.7)
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    x_scale = 8
    y_scale = 8
    x_pos = [x_scale, x_scale / 1.414, 0, -x_scale / 1.414, -x_scale, -x_scale / 1.414, 0, x_scale / 1.414]
    y_pos = [0, y_scale / 1.414, y_scale, y_scale / 1.414, 0, -y_scale / 1.414, -y_scale, -y_scale / 1.414]
    r2_all = 0
    dist_all = 0
    for i in range(8):
        r2 = sum(R2_Mat[i]) / 5
        dist = sum(dist_Mat[i]) / 5
        plt.text(x_pos[i], y_pos[i], f"{r2:.2f} | {dist: .2f}")
        r2_all += r2
        dist_all += dist
    r2_all /= 8
    dist_all /= 8
    plt.title(f'GRU+refit-{session_name}-{local_time}-r2:{r2_all: .3f}-dist:{dist_all: .3f}')
    plt.show()
