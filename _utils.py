#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time:         16:39
# @Author:      WGF
# @File:        _utils.py
# @Description:

import numpy as np
import math as mt


def pos_screenPIX2stdCM(Pos, screen_W, screen_H, cm_W, cm_H):
    Pos_new = np.zeros(2,)
    Pos_new[0] = Pos[0] - screen_W / 2
    Pos_new[1] = -Pos[1] + screen_H / 2
    Pos_new[0] = Pos_new[0] / screen_W * cm_W
    Pos_new[1] = Pos_new[1] / screen_H * cm_H
    return Pos_new


def pos_screen2std(Pos, screen_W, screen_H):
    Pos_new = np.zeros(2,)
    Pos_new[0] = Pos[0] - screen_W / 2
    Pos_new[1] = -Pos[1] + screen_H / 2
    return Pos_new

def trial_cell_to_dict(expData):
    multi_cell_array = expData["trialData"]
    dictList = []
    for row in multi_cell_array:
        for element in row:
            dict = {}
            for name in element.dtype.names:
                if element[name][0][0].size > 0:
                    dict[name] = element[name][0][0][0]
                else:
                    dict[name] = element[name][0][0]
            dictList.append(dict)
    return dictList


def R2_score(targts, preds):
    SS_res = np.sum((targts - preds) ** 2)
    # print(SS_res)
    SS_tot = np.sum((targts - np.mean(targts, axis=0)) ** 2)
    # print(SS_tot)
    r2 = 1 - SS_res / SS_tot
    # print('R^2: ', r2)
    return r2


def final_dist(targts, preds):
    fd = np.sqrt(np.max([np.sum(targts[-1, :] ** 2) + np.sum(preds[-1, :] ** 2) \
                         - 2 * targts[-1, :].dot(preds[-1, :]), 0]))

    return fd


def trial_dist(targets, preds):
    dist = 0
    for i in range(len(targets)):
        dist += final_dist(targets[i, :].reshape(1, -1), preds[i, :].reshape(1, -1))
    mean_dist = dist / len(targets)
    return mean_dist


def calculate_mean_variance(matrix):
    mean = np.mean(matrix)
    variance = np.var(matrix)
    return mean, variance


def refitVelocity(target_pos, X, weight):
    '''
    refit计算旋转角度
    :param target_pos: 目标位置坐标
    :param x_state_true: 输入状态
    :param weight: 旋转权重
    :return: 更新后速度向量
    '''
    # 预测的位置和速度
    p_loc = X[0:2].reshape(-1)
    p_vel = X[2:4].reshape(-1)
    speed = np.sqrt(sum((p_vel ** 2)))
    correctVect = target_pos - p_loc
    # 计算target方向的tvx tvy
    vRot = speed * correctVect / np.linalg.norm(correctVect)
    # dot 点乘求和, target方向与预测速度的角度
    angle = np.arccos((vRot @ p_vel) / (np.linalg.norm(vRot) * np.linalg.norm(p_vel)))
    # 根据叉乘判断target方向与预测速度的旋转方向
    # 设矢量P = (x1, y1) ，Q = (x2, y2), P × Q = x1 * y2 - x2 * y1 。若P × Q > 0, 则P
    # 在Q的顺时针方向.若 P × Q < 0, 则P在Q的逆时针方向；若P × Q = 0, 则P与Q共线，但可能同向也可能反向；
    pos = p_vel[0] * vRot[1] - p_vel[1] * vRot[0]
    if pos < 0:
        angle = -angle
    # 根据权重计算需要旋转的角度
    angle = angle * weight
    # 坐标转换矩阵
    TranM = [[mt.cos(angle), mt.sin(angle)],
             [-mt.sin(angle), mt.cos(angle)]]
    # 向量旋转angle角度
    new_vel = p_vel @ TranM
    # new_vel = TranM @ p_vel
    x_new = np.vstack((X.reshape(-1,1)[0:2, :], new_vel.reshape(-1,1)))
    return x_new
