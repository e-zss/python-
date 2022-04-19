# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import geatpy as ea

"""
带时间窗、容量约束且行驶速度可变的路径优化问题（Capacited vehicle routing problem with time-windows and variable speeds,CVRPTWVS）;
带分区传输时间、拣货准备时间（融合在拣货时间矩阵内）、打包时间的订单分区串行拣选问题（Zone picking with convey time,ZPC）
逆向调度
1.CVRPTWVS
2.ZPC
"""

class MyProblem(ea.Problem): # 继承Problem父类
    def __init__(self):
        name = 'MyProblem' # 初始化name（函数名称，可以随意设置）
        M = 1 # 初始化M（目标维数）
        maxormins = [1] # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        num=25 #订单量
        url = "D://Onedrive/SEU/Paper_chen/Paper_2/data_figure_table/cus_data_origin.csv"
        rawdata=pd.read_csv(url,nrows =num+1, header=None)
        # 坐标
        X = list(rawdata.iloc[:, 1])
        Y = list(rawdata.iloc[:, 2])
        # 最早到达时间
        eh = list(rawdata.iloc[:, 4])
        # 最晚到达时间
        lh = list(rawdata.iloc[:, 5])
        time_windows = \
             []
        for i in range(len(rawdata)):
            time_windows.append((eh[i], lh[i]))
        Dim = num+1 # 初始化Dim（决策变量维数）
        varTypes = [1] * Dim # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        lb = [0] * Dim # 决策变量下界
        ub = [num] * Dim # 决策变量上界
        lbin = [1] * Dim # 决策变量下边界
        ubin = [1] * Dim # 决策变量上边界
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
        # 新增一个属性存储旅行地坐标
        _locations = \
          []
        for i in range(len(rawdata)):
             _locations.append((X[i], Y[i]))

        self.places = np.array([(l[0] *300, l[1] * 300) for l in _locations])
        self.num_locations=len(self.places)
        num_vehicles=50
        self.num_vehicles= num_vehicles
        self.depot= 0
        self.demands = list(rawdata.iloc[:, 3])
        self.MaxLoad = 12
        capacities = []
        for i in range(num_vehicles):
            capacities.append(self.MaxLoad)
        self.vehicle_capacities = capacities
        self.time_windows = time_windows #min
        self.ServiceTime = 1 #每个订单服务时间
        self.vehicle_speed = 500 #m/min 即30km/h
    
    def aimFunc(self, pop): # 目标函数
        x = pop.Phen # 得到决策变量矩阵
        # 添加从0地出发且最后回到出发地
        X = np.hstack([np.zeros((x.shape[0], 1)), x, np.zeros((x.shape[0], 1))]).astype(int)
        
        ObjV = [] # 存储所有种群个体对应的总路程
        for i in range(X.shape[0]):
            journey = self.places[X[i], :] # 按既定顺序到达的地点坐标
            distance = np.sum(np.sqrt(np.sum(np.diff(journey.T)**2, 0))) # 计算总路程
            ObjV.append(distance)
        pop.ObjV = np.array([ObjV]).T # 把求得的目标函数值赋值给种群pop的ObjV
        # 找到违反约束条件的个体在种群中的索引，保存在向量exIdx中（如：若0、2、4号个体违反约束条件，则编程找出他们来）
        exIdx1 = np.where(np.where(x == 3)[1] - np.where(x == 6)[1] < 0)[0]
        exIdx2 = np.where(np.where(x == 4)[1] - np.where(x == 5)[1] < 0)[0]
        exIdx = np.unique(np.hstack([exIdx1, exIdx2]))
        pop.CV[exIdx] = 1 # 把求得的违反约束程度矩阵赋值给种群pop的CV
    