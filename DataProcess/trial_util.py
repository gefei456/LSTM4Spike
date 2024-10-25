'''
@Software: PyCharm
@Time:2023/4/7 13:47
@Author: gwq
'''
import scipy.io as scio

def trial_cell_to_dict_ori(expData):
    multi_cell_array = expData["trialData"]
    dictList = []
    for row in multi_cell_array:
        for element in row:
            dict = {}
            for name in element.dtype.names:
                if name == "center":
                    center = {}
                    for centerEle in element[name][0]:
                        for cell_name in centerEle[0].dtype.names:
                            center[cell_name] = centerEle[cell_name][0][0][0]
                    dict[name] = center
                elif name == "target":
                    target = {}
                    for centerEle in element[name][0]:
                        for cell_name in centerEle[0].dtype.names:
                            target[cell_name] = centerEle[cell_name][0][0][0]
                    dict[name] = target
                else:
                    dict[name] = element[name][0][0][0]
            dictList.append(dict)
    return dictList

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

if __name__ == '__main__':
    expData = scio.loadmat(r"D:\log\baize_bj-co-offline_04-17_47\trialData.mat")  # 从文件中加载数据
    trials = trial_cell_to_dict(expData)
    print(trials)