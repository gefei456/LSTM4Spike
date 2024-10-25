#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time:         16:39
# @Author:      WGF
# @File:        _utils.py
# @Description:


import numpy as np

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

import math as mt
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
