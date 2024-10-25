#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time:         16:08
# @Author:      WGF
# @File:        gru_module.py
# @Description:

import torch
import torch.nn as nn


class basic_GRU(nn.Module):
    def __init__(self, use_bottleneck=False, bottleneck_width=256, n_input=128, n_hiddens=[64, 64], n_output=6, dropout=0.0,
                 len_seq=9, model_type='BasicGRU', trans_loss='mmd'):
        super(basic_GRU, self).__init__()
        self.use_bottleneck = use_bottleneck
        self.n_input = n_input
        self.num_layers = len(n_hiddens)
        self.hiddens = n_hiddens
        self.n_output = n_output
        self.model_type = model_type
        self.trans_loss = trans_loss
        self.len_seq = len_seq
        in_size = self.n_input

        features = nn.ModuleList()
        for hidden in n_hiddens:
            rnn = nn.GRU(
                input_size=in_size,
                num_layers=1,
                hidden_size=hidden,
                batch_first=True,
                dropout=dropout
            )
            features.append(rnn)
            in_size = hidden
        self.features = nn.Sequential(*features)

        if self.use_bottleneck:  # finance
            self.bottleneck = nn.Sequential(
                nn.Linear(n_hiddens[-1], bottleneck_width),
                nn.Linear(bottleneck_width, bottleneck_width),
                nn.BatchNorm1d(bottleneck_width),
                nn.ReLU(),
                nn.Dropout(),
            )
            self.bottleneck[0].weight.data.normal_(0, 0.005)
            self.bottleneck[0].bias.data.fill_(0.1)
            self.bottleneck[1].weight.data.normal_(0, 0.005)
            self.bottleneck[1].bias.data.fill_(0.1)
            self.fc = nn.Linear(bottleneck_width, n_output)
            nn.init.xavier_normal_(self.fc.weight)
        else:
            self.fc_out = nn.Linear(n_hiddens[-1], self.n_output)

        if self.model_type == 'BasicGRU':
            gate = nn.ModuleList()
            for i in range(len(n_hiddens)):
                gate_weight = nn.Linear(
                    len_seq * self.hiddens[i] * 2, len_seq)
                gate.append(gate_weight)
            self.gate = gate

            bnlst = nn.ModuleList()
            for i in range(len(n_hiddens)):
                bnlst.append(nn.BatchNorm1d(len_seq))
            self.bn_lst = bnlst
            self.softmax = nn.Softmax(dim=0)
            self.init_layers()

    def init_layers(self):
        for i in range(len(self.hiddens)):
            self.gate[i].weight.data.normal_(0, 0.05)
            self.gate[i].bias.data.fill_(0.0)

    def gru_features(self, x, predict=False):
        x_input = x
        out = None
        out_lis = []
        out_weight_list = [] if (
             self.model_type == 'BasicGRU') else None
        for i in range(self.num_layers):
            out, _ = self.features[i](x_input.float())
            x_input = out
            out_lis.append(out)
            if self.model_type == 'BasicGRU' and predict == False:
                out_gate = self.process_gate_weight(x_input, i)
                out_weight_list.append(out_gate)
        return out, out_lis, out_weight_list

    def forward(self, x):
        out = self.gru_features(x)
        fea = out[0]
        if self.use_bottleneck:
            fea_bottleneck = self.bottleneck(fea[:, -1, :])
            fc_out = self.fc(fea_bottleneck).squeeze()
        else:
            fc_out = self.fc_out(fea[:, -1, :]).squeeze()
        return fc_out

    def process_gate_weight(self, out, index):
        x_s = out[0: int(out.shape[0]//2)]
        x_t = out[out.shape[0]//2: out.shape[0]]
        x_all = torch.cat((x_s, x_t), 2)
        x_all = x_all.view(x_all.shape[0], -1)
        weight = torch.sigmoid(self.bn_lst[index](
            self.gate[index](x_all.float())))
        weight = torch.mean(weight, dim=0)
        res = self.softmax(weight).squeeze()
        return res
