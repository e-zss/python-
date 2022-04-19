## 环境设定
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from deap import base, tools, creator, algorithms
import random
import multiprocessing
from time import time
from Routing_plot import plot



params = {
    'font.family': 'serif',
    # 'figure.dpi': 300,
    # 'savefig.dpi': 300,
    'font.size': 12,
    # 'text.usetex': True, #用latex渲染
    'legend.fontsize': 'small'
    
}
plt.rcParams.update(params)

from copy import deepcopy
#-----------------------------------
## 问题定义
creator.create('FitnessMin', base.Fitness, weights=(-1.0,)) # 最小化问题
# 给个体一个routes属性用来记录其表示的路线
creator.create('Individual', list, fitness=creator.FitnessMin) 

#-----------------------------------
## 个体编码
# 用字典存储所有参数 -- 配送中心坐标、顾客坐标、顾客需求、到达时间窗口、服务时间、车型载重量
dataDict = {}
# 节点坐标，节点0是配送中心的坐标
num=25
url="D://Onedrive/SEU/Paper_chen/Paper_2/data_figure_table/cus_data_origin.csv"
rawdata=pd.read_csv(url,nrows =num+1, header=None)
# 坐标
X = list(rawdata.iloc[:, 1])
Y = list(rawdata.iloc[:, 2])
# 最早到达时间
eh = list(rawdata.iloc[:, 4])
# 最晚到达时间
lh = list(rawdata.iloc[:, 5])
# Locations in block units
_locations = \
          []
for i in range(len(rawdata)):
    _locations.append((X[i], Y[i]))

dataDict['NodeCoor'] = [(l[0], l[1]) for l in _locations]
# 将配送中心的需求设置为0
dataDict['Demand'] = list(rawdata.iloc[:, 3])
# 将配送中心的服务时间设置为0
time_windows = \
             []
for i in range(len(rawdata)):
    time_windows.append((eh[i], lh[i]))
dataDict['Timewindow'] = time_windows
dataDict['MaxLoad'] = 12
dataDict['ServiceTime'] = 1
dataDict['Velocity'] = 500 # 车辆的平均行驶速度

def genInd(dataDict = dataDict):
    '''生成个体， 对我们的问题来说，困难之处在于车辆数目是不定的'''
    nCustomer = len(dataDict['NodeCoor']) - 1 # 顾客数量
    perm = np.random.permutation(nCustomer) + 1 # 生成顾客的随机排列,注意顾客编号为1--n
    pointer = 0 # 迭代指针
    lowPointer = 0 # 指针指向下界
    permSlice = []
    # 当指针不指向序列末尾时
    while pointer < nCustomer -1:
        vehicleLoad = 0
        # 当不超载时，继续装载
        while (vehicleLoad < dataDict['MaxLoad']) and (pointer < nCustomer -1):
            vehicleLoad += dataDict['Demand'][perm[pointer]]
            pointer += 1
        if lowPointer+1 < pointer:
            tempPointer = np.random.randint(lowPointer+1, pointer)
            permSlice.append(perm[lowPointer:tempPointer].tolist())
            lowPointer = tempPointer
            pointer = tempPointer
        else:
            permSlice.append(perm[lowPointer::].tolist())
            break
    # 将路线片段合并为染色体
    ind = [0]
    for eachRoute in permSlice:
        ind = ind + eachRoute + [0]
    return ind
#-----------------------------------
## 评价函数
# 染色体解码
def decodeInd(ind):
    '''从染色体解码回路线片段，每条路径都是以0为开头与结尾'''
    indCopy = np.array(deepcopy(ind)) # 复制ind，防止直接对染色体进行改动
    idxList = list(range(len(indCopy)))
    zeroIdx = np.asarray(idxList)[indCopy == 0]
    routes = []
    for i,j in zip(zeroIdx[0::], zeroIdx[1::]):
        routes.append(ind[i:j]+[0])
    return routes

def calDist(pos1, pos2,h1=0.3,h2=0.15):
    '''计算距离的辅助函数，根据给出的坐标pos1和pos2，返回两点之间的距离
    输入： 
    pos1, pos2 -- (x,y)元组 ;
    h1:caution intensity when depart from depot ;
    h2:caution intensity in-transit

    输出： 曼哈顿距离'''
    if pos1==(5,5):
        return (abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]))*300/(1-h1)
    elif pos2==(5,5):
        return (abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]))*300/(1-h2)
    else:
        return (abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]))*300

#
def loadPenalty(routes):
    '''辅助函数，因为在交叉和突变中可能会产生不符合负载约束的个体，需要对不合要求的个体进行惩罚'''
    penalty = 0
    # 计算每条路径的负载，取max(0, routeLoad - maxLoad)计入惩罚项
    for eachRoute in routes:
        routeLoad = np.sum([dataDict['Demand'][i] for i in eachRoute])
        penalty += max(0, routeLoad - dataDict['MaxLoad'])
    return penalty

def calcRouteServiceTime(route, dataDict = dataDict):
    '''辅助函数，根据给定路径，计算到达该路径上各顾客的时间'''
    # 初始化serviceTime数组，其长度应该比eachRoute小2
    serviceTime = [0] * (len(route) - 2)
    # 从仓库到第一个客户时不需要服务时间
    arrivalTime = calDist(dataDict['NodeCoor'][0], dataDict['NodeCoor'][route[1]])/dataDict['Velocity']
    arrivalTime = max(arrivalTime, dataDict['Timewindow'][route[1]][0])
    serviceTime[0] = arrivalTime
    arrivalTime += dataDict['ServiceTime'] # 在出发前往下个节点前完成服务
    for i in range(1, len(route)-2):
        # 计算从路径上当前节点[i]到下一个节点[i+1]的花费的时间
        arrivalTime += calDist(dataDict['NodeCoor'][route[i]], dataDict['NodeCoor'][route[i+1]])/dataDict['Velocity']
        arrivalTime = max(arrivalTime, dataDict['Timewindow'][route[i+1]][0])
        serviceTime[i] = arrivalTime
        arrivalTime += dataDict['ServiceTime'] # 在出发前往下个节点前完成服务
    return serviceTime

def timeTable(distributionPlan, dataDict = dataDict):
    '''辅助函数，依照给定配送计划，返回每个顾客受到服务的时间'''
    # 对于每辆车的配送路线，第i个客户受到服务的时间serviceTime[i]是min(TimeWindow[i][0], arrivalTime[i])
    # arrivalTime[i] = serviceTime[i-1] + 服务时间 + distance(i,j)/averageVelocity
    timeArrangement = [] #容器，用于存储每个顾客受到服务的时间
    for eachRoute in distributionPlan:
        serviceTime = calcRouteServiceTime(eachRoute)
        timeArrangement.append(serviceTime)
    # 将数组重新组织为与基因编码一致的排列方式
    realignedTimeArrangement = [0]
    for routeTime in timeArrangement:
        realignedTimeArrangement = realignedTimeArrangement + routeTime + [0]
    return realignedTimeArrangement

def timePenalty(ind, routes):
    '''辅助函数，对不能按服务时间到达顾客的情况进行惩罚'''
    timeArrangement = timeTable(routes) # 对给定路线，计算到达每个客户的时间
    # 索引给定的最迟到达时间
    desiredTime = [dataDict['Timewindow'][ind[i]][1] for i in range(len(ind))]
    # 如果最迟到达时间大于实际到达客户的时间，则延迟为0，否则延迟设为实际到达时间与最迟到达时间之差
    timeDelay = [max(timeArrangement[i]-desiredTime[i],0) for i in range(len(ind))]
    return np.sum(timeDelay)/len(timeDelay)

def calRouteLen(routes,dataDict=dataDict):
    '''辅助函数，返回给定路径的总长度'''
    totalDistance = 0 # 记录各条路线的总长度
    for eachRoute in routes:
        # 从每条路径中抽取相邻两个节点，计算节点距离并进行累加
        for i,j in zip(eachRoute[0::], eachRoute[1::]):
            totalDistance += calDist(dataDict['NodeCoor'][i], dataDict['NodeCoor'][j])    
    return totalDistance



def evaluate(ind, c0=3,c1=0.005,c2=9999.0, c3=9999.0):
    '''评价函数，返回解码后路径的总长度，c0,c1,c1, c2分别为为单位派车成本、单位行驶成本、单位运载惩罚、单位时间惩罚成本'''
    routes = decodeInd(ind) # 将个体解码为路线
    totalDistance = calRouteLen(routes)
    
    return (c0*len(routes)+c1*totalDistance + c2*loadPenalty(routes) + c3*timePenalty(ind,routes)),
#-----------------------------------
## 交叉操作
def genChild(ind1, ind2, nTrail=5):
    '''参考《基于电动汽车的带时间窗的路径优化问题研究》中给出的交叉操作，生成一个子代'''
    # 在ind1中随机选择一段子路径subroute1，将其前置
    routes1 = decodeInd(ind1) # 将ind1解码成路径
    numSubroute1 = len(routes1) # 子路径数量
    subroute1 = routes1[np.random.randint(0, numSubroute1)]
    # 将subroute1中没有出现的顾客按照其在ind2中的顺序排列成一个序列
    unvisited = set(ind1) - set(subroute1) # 在subroute1中没有出现访问的顾客
    unvisitedPerm = [digit for digit in ind2 if digit in unvisited] # 按照在ind2中的顺序排列
    # 多次重复随机打断，选取适应度最好的个体
    bestRoute = None # 容器
    bestFit = np.inf
    for _ in range(nTrail):
        # 将该序列随机打断为numSubroute1-1条子路径
        breakPos = [0]+random.sample(range(1,len(unvisitedPerm)),numSubroute1-2) # 产生numSubroute1-2个断点
        breakPos.sort()
        breakSubroute = []
        for i,j in zip(breakPos[0::], breakPos[1::]):
            breakSubroute.append([0]+unvisitedPerm[i:j]+[0])
        breakSubroute.append([0]+unvisitedPerm[j:]+[0])
        # 更新适应度最佳的打断方式
        # 将先前取出的subroute1添加入打断结果，得到完整的配送方案
        breakSubroute.append(subroute1)
        # 评价生成的子路径
        routesFit = calRouteLen(breakSubroute) + loadPenalty(breakSubroute)
        if routesFit < bestFit:
            bestRoute = breakSubroute
            bestFit = routesFit
    # 将得到的适应度最佳路径bestRoute合并为一个染色体
    child = []
    for eachRoute in bestRoute:
        child += eachRoute[:-1]
    return child+[0]

def crossover(ind1, ind2):
    '''交叉操作'''
    ind1[:], ind2[:] = genChild(ind1, ind2), genChild(ind2, ind1)
    return ind1, ind2

#-----------------------------------
## 突变操作
def opt(route,dataDict=dataDict, k=2, c1=1.0, c2=500.0):
    # 用2-opt算法优化路径
    # 输入：
    # route -- sequence，记录路径
    # k -- k-opt，这里用2opt
    # c1, c2 -- 寻求最短路径长度和满足时间窗口的相对重要程度
    # 输出： 优化后的路径optimizedRoute及其路径长度
    nCities = len(route) # 城市数
    optimizedRoute = route # 最优路径
    desiredTime = [dataDict['Timewindow'][route[i]][1] for i in range(len(route))]
    serviceTime = calcRouteServiceTime(route)
    timewindowCost = [max(serviceTime[i]-desiredTime[1:-1][i],0) for i in range(len(serviceTime))]
    timewindowCost = np.sum(timewindowCost)/len(timewindowCost)
    minCost = c1*calRouteLen([route]) +  c2*timewindowCost # 最优路径代价
    for i in range(1,nCities-2):
        for j in range(i+k, nCities):
            if j-i == 1:
                continue
            reversedRoute = route[:i]+route[i:j][::-1]+route[j:]# 翻转后的路径
            # 代价函数中需要同时兼顾到达时间和路径长度
            desiredTime = [dataDict['Timewindow'][reversedRoute[i]][1] for i in range(len(reversedRoute))]
            serviceTime = calcRouteServiceTime(reversedRoute)
            timewindowCost = [max(serviceTime[i]-desiredTime[1:-1][i],0) for i in range(len(serviceTime))]
            timewindowCost = np.sum(timewindowCost)/len(timewindowCost)
            reversedRouteCost = c1*calRouteLen([reversedRoute]) + c2*timewindowCost
            # 如果翻转后路径更优，则更新最优解
            if  reversedRouteCost < minCost:
                minCost = reversedRouteCost
                optimizedRoute = reversedRoute
    return optimizedRoute

def mutate(ind):
    '''用2-opt算法对各条子路径进行局部优化'''
    routes = decodeInd(ind)
    optimizedAssembly = []
    for eachRoute in routes:
        optimizedRoute = opt(eachRoute)
        optimizedAssembly.append(optimizedRoute)
    # 将路径重新组装为染色体
    child = []
    for eachRoute in optimizedAssembly:
        child += eachRoute[:-1]
    ind[:] = child+[0]
    return ind,
#-----------------------------------
## 注册遗传算法操作
toolbox = base.Toolbox()
toolbox.register('individual', tools.initIterate, creator.Individual, genInd)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)
toolbox.register('evaluate', evaluate)
toolbox.register('select', tools.selTournament, tournsize=2) #锦标赛
toolbox.register('mate', crossover)
toolbox.register('mutate', mutate)

## 生成初始族群
toolbox.popSize = 100
pop = toolbox.population(toolbox.popSize)

## 记录迭代数据
stats=tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register('min', np.min)
stats.register('avg', np.mean)
stats.register('std', np.std)
hallOfFame = tools.HallOfFame(maxsize=1)

## 遗传算法参数
toolbox.ngen = 30
toolbox.cxpb = 0.8
toolbox.mutpb = 0.1

## 遗传算法主程序
start_tiem=time()
pop,logbook=algorithms.eaMuPlusLambda(pop, toolbox, mu=toolbox.popSize, 
                                      lambda_=toolbox.popSize,cxpb=toolbox.cxpb, mutpb=toolbox.mutpb,
                   ngen=toolbox.ngen ,stats=stats, halloffame=hallOfFame, verbose=True)

from pprint import pprint

def calLoad(routes):
    loads = []
    for eachRoute in routes:
        routeLoad = np.sum([dataDict['Demand'][i] for i in eachRoute])
        loads.append(routeLoad)
    return loads

bestInd = hallOfFame.items[0]
distributionPlan = decodeInd(bestInd)
bestFit = bestInd.fitness.values
print('Best distribution plan：')
pprint(distributionPlan)
print('Total distance：')
print(evaluate(bestInd,c0=0,c1=1,c2=0, c3=0))  #c0,c1,c1, c2分别为为单位派车成本、单位行驶成本、单位运载惩罚、单位时间惩罚成本
print('The load of each vehicle：')
print(calLoad(distributionPlan))

timeArrangement = timeTable(distributionPlan) # 对给定路线，计算到达每个客户的时间
# 索引给定的最迟到达时间
desiredTime = [dataDict['Timewindow'][bestInd[i]][1] for i in range(len(bestInd))]
# 如果最迟到达时间大于实际到达客户的时间，则延迟为0，否则延迟设为实际到达时间与最迟到达时间之差
timeDelay = [max(timeArrangement[i]-desiredTime[i],0) for i in range(len(bestInd))]
print('Delay time：')
print(timeDelay)
print("CPU time used:",time()-start_tiem)
# 画出迭代图
Draw=1# 0-不绘制；1-绘制
if Draw==1:
    minFit = logbook.select('min')
    avgFit = logbook.select('avg')
    plt.figure()
    plt.plot(minFit, 'b-', label='Minimum Value')
    plt.plot(avgFit, 'r-', label='Average Value')
    plt.xlabel('Generation')
    plt.ylabel('Value')
    plt.legend(loc='best')
    plt.show()

#画出路径优化图
Draw_route=1
if Draw_route==1:
    plot(bestInd,url,num)

## 计算结果如下：
##最佳运输计划为：
#[[0, 1, 2, 3, 4, 6, 0],
# [0, 9, 0],
# [0, 14, 29, 17, 30, 26, 18, 23, 21, 0],
# [0, 8, 25, 15, 16, 0],
# [0, 10, 11, 24, 12, 0],
# [0, 5, 7, 0],
# [0, 28, 27, 20, 19, 22, 13, 0]]
#总运输距离为：
#(278.62210617851554,)
#各辆车上负载为：
#[200, 30, 150, 131, 120, 110, 160]
#到达各客户的延迟为：
#[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]