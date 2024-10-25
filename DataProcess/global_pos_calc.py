import time
import numpy as np
# def screen_pos_center(pos_center, global_pos)
#
#     return

def high_precision_delay(delay_time):
    ''' Function to provide accurate time delay in millisecond
    '''
    _ = time.perf_counter() + delay_time
    while time.perf_counter() < _:
        pass
class PoseCalc():

    #O坐标系原点， Px X轴单位向量 Py Y轴单位向量
    def matrixSwitch(self, Po, Px, Py):

        X = (Px - Po) / np.linalg.norm(Px - Po)
        Y = (Py - Po) / np.linalg.norm(Py - Po)
        Z = np.cross(X,Y)
        Znorm = np.linalg.norm(Z)
        z_ = Z / Znorm
        #选择矩阵
        R = np.matrix([X, Y, Z]).transpose()
        #平移矩阵
        T = np.matrix(Po).transpose()
        return R, T

#平面方程 ax+by+cz+d=0
def getPanel(p1, p2, p3):
    a = (p2[1] - p1[1]) * (p3[2] - p1[2]) - (p2[2] - p1[2]) * (p3[1] - p1[1])
    b = (p2[2] - p1[2]) * (p3[0] - p1[0]) - (p2[0] - p1[0]) * (p3[2] - p1[2])
    c = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
    d = 0 - (a * p1[0] + b * p1[1] + c* p1[2])
    return a,b,c,d

#平面法向量
def getNormal(p1,p2,p3):
    a = (p2[1] - p1[1]) * (p3[2] - p1[2]) - (p2[2] - p1[2]) * (p3[1] - p1[1])
    b = (p2[2] - p1[2]) * (p3[0] - p1[0]) - (p2[0] - p1[0]) * (p3[2] - p1[2])
    c = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
    return (a,b,c)

#点到平面距离
def disPoint2Panel(pt,a,b,c,d):
    return np.abs(a*pt[0] + b*pt[1] + c*pt[2] + d) / np.sqrt(a**2 + b**2 + c**2)


# def pos_screenPIX2stdCM(Pos, screen_W, screen_H, cm_W, cm_H):
#     Pos_new = np.zeros(2,)
#     Pos_new[0] = Pos[0] - screen_W / 2
#     Pos_new[1] = -Pos[1] + screen_H / 2
#     Pos_new[0] = Pos_new[0] / screen_W * cm_W
#     Pos_new[1] = Pos_new[1] / screen_H * cm_H
#     return Pos_new

# 空间坐标点转换接口
# pos_LU, pos_RU, pos_LD, pos_RD 分别为显示器左上右上左下右下角点空间坐标
# screen_width, screen_height为显示区域的宽和高
# global_pos为动捕设备发送的坐标
# 返回值screen_Pos_cur[0], screen_Pos_cur[1]分别为转换后的横纵坐标
def pos_screen2std(Pos, screen_W, screen_H):
    Pos_new = np.zeros(2,)
    Pos_new[0] = Pos[0] - screen_W / 2
    Pos_new[1] = -Pos[1] + screen_H / 2
    return Pos_new
def screen_pos_calc(pos_LU, pos_RU, pos_LD, pos_RD, screen_width, screen_height, global_pos):
    Po = 1 / 2 * (pos_LU + pos_RD)
    Px_Norm = np.linalg.norm(pos_RD - pos_LD)
    Py_Norm = np.linalg.norm(pos_LU - pos_LD)
    Px = Po + (pos_RD - pos_LD) / Px_Norm
    Py = Po + (pos_LU - pos_LD) / Py_Norm
    Sx = Px_Norm / screen_width
    Sy = Py_Norm / screen_height
    # 求转移矩阵
    pos = PoseCalc()
    R, T = pos.matrixSwitch(Po, Px, Py)
    S = np.array((Sx, Sy, 1))
    S = np.diag(S)
    MM = np.append(np.append((R @ S), T, axis=1), np.array([0, 0, 0, 1]).reshape(1, -1), axis=0)
    # a,b,c,d = getPanel(pos_LU, pos_LD, pos_RD)
    # MM[2] = [a,b,c,d]
    global_pos = np.matrix(np.append(global_pos, 1)).transpose()
    screen_Pos_cur = np.array(np.linalg.inv(MM) @ global_pos)
    return screen_Pos_cur[0], screen_Pos_cur[1]

    # Po = 1 / 2 * (pos_LU + pos_RD)
    # Px_Norm = np.linalg.norm(pos_RD - pos_LD)
    # Py_Norm = np.linalg.norm(pos_LU - pos_LD)
    # Px = Po + (pos_RD - pos_LD) / Px_Norm
    # Py = Po + (pos_LU - pos_LD) / Py_Norm
    # Sx = Px_Norm / screen_width
    # Sy = Py_Norm / screen_height
    # # 求转移矩阵
    # pos = pose_calc.PoseCalc()
    # R, T = pos.matrixSwitch(Po, Px, Py)
    # S = np.array((Sx,Sy,1),dtype=float)
    # S = np.diag(S)
    # # A = np.matrix([[1,2,3],[7,8,9],[4,5,6]])
    # # B = np.array([[3,2,1],[4,3,2],[4,3,4]])
    # # MM=[]
    # # startTime = time.time()
    # # MM = np.dot(A,B)
    # # R = np.array(R)
    # # MM = np.matrix([R@S, T])
    # # for i in range(10):
    # # MM = np.dot(R, S)
    # # Sx = Px_Norm / screen_width
    # # MM = np.dot(R, A)
    # # Sx = Px_Norm / screen_width
    # # MM = np.dot(A, S)
    # # Sx = Px_Norm / screen_width
    # # MM = np.dot(S, A)
    # # MM = np.dot(R, B)
    # # MM = np.dot(B, R)
    # # Sx = Px_Norm / screen_width
    # # MM = np.dot(S, R)
    # MM = np.append(np.append((R @ S), T, axis=1), np.array([0, 0, 0, 1]).reshape(1, -1), axis=0)
    # # a,b,c,d = getPanel(pos_LU, pos_LD, pos_RD)
    # # MM[2] = [a,b,c,d]
    # # endTime = time.time()
    # # print('endTime - startTime:', (endTime - startTime) * 1000)
    # # print('startTime:', startTime, '\n')
    # # print('endTime:', endTime, '\n')
    # # print('MM', MM)
    # global_pos = np.matrix(np.append(global_pos, 1)).transpose()
    # screen_Pos_cur = np.array(np.linalg.inv(MM) @ global_pos)
    # return screen_Pos_cur[0], screen_Pos_cur[1]
    # return MM[0],MM[1]

global_pos = np.array([49.687503814697266, -208.86325073242188, 1506.2757568359375])

zs = np.array([473.7781677246094, -438.68621826171875, 1794.763671875])
zx = np.array([424.4982604980469, -452.0476379394531, 1101.83154296875])
ys = np.array([-289.5498962402344, 11.695173263549805, 1845.4090576171875])
yx = np.array([-342.166015625, -0.7464830875396729, 1149.016357421875])
global_pos = zs
A = np.matrix([[1, 2, 3], [7, 8, 9], [4, 5, 6]], dtype=float)
B = np.array([[3, 2, 1], [4, 3, 2], [4, 3, 4]], dtype=float)
NN = np.dot(A, B)
startTime0 = time.time_ns()
# startTime1 = time.time_ns()
for i in range(60):
    # xx, yy = screen_pos_calc(zs, ys, zx, yx, 400, 300, global_pos)
    # time.sleep(0.05)
    # startTime2 = time.time_ns()
    # startTime1 = time.time_ns() + 50*1e6
    # while time.time_ns() < startTime1:
    #     pass
    high_precision_delay(0.05)
endTime = time.time_ns()
# print('startTime:', startTime, '\n')
# print('endTime:', endTime, '\n')
print('endTime - startTime:', (endTime - startTime0)/1e6)
# print(xx,yy)