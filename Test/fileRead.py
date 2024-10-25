#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time:        2023/9/27 14:16
# @Author:      WGF
# @File:        fileRead.py
# @Description: read mode state dict

import torch

if __name__ == '__main__':
    time_str = 'good_GRU_2023-09-24--16_50_47'
    model_path = '../20230209-2_20230209-3_EXP/' + time_str + '/GRU-Model.pth'
    state_dict_path = '../20230209-2_20230209-3_EXP/' + time_str + '/GRU-state_dict.pt'

    model = torch.load(model_path)
    print(model.state_dict())
    for name, parameters in model.named_parameters():
        print(name, ':', parameters.size())
    checkpoints = torch.load(state_dict_path)
    print('checkpoints:', checkpoints)
