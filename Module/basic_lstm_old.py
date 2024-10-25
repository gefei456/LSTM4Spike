#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time:         10:21
# @Author:      WGF
# @File:        basic_lstm.py
# @Description:

import torch.nn as nn


class BasicLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_states):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_states)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


class TwoLayerLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_states, num_layers, use_bottleneck, bottleneck_width=256):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, dropout=0.0)
        self.use_bottleneck = use_bottleneck
        if self.use_bottleneck == True:  # finance
            self.bottleneck = nn.Sequential(
                nn.Linear(hidden_size, bottleneck_width),
                nn.Linear(bottleneck_width, bottleneck_width),
                nn.BatchNorm1d(bottleneck_width),
                nn.ReLU(),
                nn.Dropout(),
            )
            self.bottleneck[0].weight.data.normal_(0, 0.005)
            self.bottleneck[0].bias.data.fill_(0.1)
            self.bottleneck[1].weight.data.normal_(0, 0.005)
            self.bottleneck[1].bias.data.fill_(0.1)
            self.fc = nn.Linear(bottleneck_width, num_states)
            nn.init.xavier_normal_(self.fc.weight)
        else:
            self.fc_out = nn.Linear(hidden_size, num_states)

    def forward(self, x):
        out, _ = self.lstm(x)
        if self.use_bottleneck == True:
            fea_bottleneck = self.bottleneck(out[:, -1, :])
            fc_out = self.fc(fea_bottleneck).squeeze()
        else:
            fc_out = self.fc_out(out[:, -1, :])
        return fc_out
