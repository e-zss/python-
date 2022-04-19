# -*- coding: utf-8 -*-

## 环境设定
import numpy as np
import matplotlib.pyplot as plt
from deap import base, tools, creator, algorithms
import random
import pandas as pd
import Levenshtein
import math
import time
from Parameter import srtime,num,url,v_trans,Q,c_od_d,c_d,c_op,c_od,f,h1,h2,rawdata,D,T,time_windows,location,demands,t_convey,t_pack,lm
params = {
    'font.family': 'serif',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 12,
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
url="D://Onedrive/SEU/Paper_chen/Paper_3/cus_data_origin.csv" #数据存放文件路径
num=28
rawdata=pd.read_csv(url,nrows =num+1, header=None)
# 坐标
X = list(rawdata.iloc[:, 1])
Y = list(rawdata.iloc[:, 2])
# 最早到达时间
eh = list(rawdata.iloc[:, 4])
# 最晚到达时间
lh = list(rawdata.iloc[:, 5])
demands=list(rawdata.iloc[:, 3]) #q[i]
srtime=list(rawdata.iloc[:, 3])
location=list(zip(X,Y))
time_windows =[(eh[i], lh[i]) for i in  range(len(rawdata))]
dataDict = {}
# 节点坐标，节点0是配送中心的坐标
dataDict['NodeCoor'] = [(l[0], l[1]) for l in location]
# 将配送中心的需求设置为0
dataDict['Demand'] = list(rawdata.iloc[:, 3])
# 将配送中心的服务时间设置为0
dataDict['Timewindow'] = [(eh[i], lh[i]) for i in  range(len(rawdata))]
dataDict['MaxLoad'] = 12
dataDict['ServiceTime'] = 1
dataDict['Velocity'] = 500 # 车辆的平均行驶速度

dp=27 #干扰位置坐标
lamda_time=0.8
lamda_damage=0.6
lamda_diff=0.5
lamda_cost=0.7

t=10
r=4

if t==10 and r==2:
    #instance:t10-r2
    t_disrupt=10 #干扰发生时刻
    N_disrupt=[9,12,22,14,4] 
    ind_benchmark=[3,19,8,7,0,13,11,21,25,0,16,18,20,0,24,17,23,10,0,1,15,5,6,0,26,27,9,12,22,14,4,0]
    R=[[3,19,8,7],[13,11,21,25],[16,18,20],[24,17,23,10],[1,15,5,6],[26]] #干扰发生后的原调度计划
    add_time=[3.39,3.87,1.75,0.21,0.93,0]
    
elif t==10 and r==4:
    #instance:t10-r4
    t_disrupt=10 #干扰发生时刻
    N_disrupt=[16,18,20] 
    ind_benchmark=[3,19,8,7,0,9,12,22,14,4,0,13,11,21,25,0,24,17,23,10,0,1,15,5,6,0,26,27,16,18,20,0]
    R=[[3,19,8,7],[9,12,22,14,4],[13,11,21,25],[24,17,23,10],[1,15,5,6],[26]] #干扰发生后的原调度计划
    add_time=[3.39,5.54,3.87,0.21,0.93,0]
    
elif t==10 and r==6:
    #instance:t10-r6
    t_disrupt=10 #干扰发生时刻
    N_disrupt=[1,15,5,6] 
    ind_benchmark=[3,19,8,7,0,9,12,22,14,4,0,13,11,21,25,0,16,18,20,0,24,17,23,10,0,26,27,1,15,5,6,0]
    R=[[3,19,8,7],[9,12,22,14,4],[13,11,21,25],[16,18,20],[24,17,23,10],[26]] #干扰发生后的原调度计划
    add_time=[3.39,5.54,3.87,1.75,0.21,0]
elif t==12 and r==2:
    #instance:t12-r2
    t_disrupt=12 #干扰发生时刻
    N_disrupt=[9,12,22,14,4] #t12
    ind_benchmark=[3,19,8,7,0,13,11,21,25,0,18,20,0,23,10,0,15,5,6,0,26,27,9,12,22,14,4,0]
    R=[[3,19,8,7],[13,11,21,25],[18,20],[23,10],[15,5,6],[26]] #干扰发生后的原调度计划
    add_time=[1.39,1.87,3.16,4.04,1.05,0]

elif t==12 and r==4:
    #instance:t12-r4
    t_disrupt=12 #干扰发生时刻
    N_disrupt=[18,20] #t12
    R=[[3,19,8,7],[9,12,22,14,4],[13,11,21,25],[23,10],[15,5,6],[26]] #干扰发生后的原调度计划
    ind_benchmark=[3,19,8,7,0,9,12,22,14,4,0,13,11,21,25,0,23,10,0,15,5,6,0,26,27,18,20,0]
    add_time=[1.39,3.54,1.87,4.04,1.05,0]

elif t==12 and r==6:
    #instance:t12-r6
    t_disrupt=12 #干扰发生时刻
    N_disrupt=[15,5,6] #t12
    R=[[3,19,8,7],[9,12,22,14,4],[13,11,21,25],[18,20],[23,10],[26]] #干扰发生后的原调度计划
    ind_benchmark=[3,19,8,7,0,9,12,22,14,4,0,13,11,21,25,0,18,20,0,23,10,0,26,27,15,5,6,0]
    add_time=[1.39,3.54,1.87,3.16,4.04,0]
elif t==14 and r==2:
    #instance:t14-r2
    t_disrupt=14 #干扰发生时刻
    N_disrupt=[9,12,22,14,4] #t12
    ind_benchmark=[19,8,7,0,11,21,25,0,18,20,0,23,10,0,5,6,0,26,27,9,12,22,14,4,0]
    R=[[19,8,7],[11,21,25],[18,20],[23,10],[5,6],[26]] #干扰发生后的原调度计划
    add_time=[1.6,1.28,1.16,2.04,6.16,0]

elif t==14 and r==4:
    # #instance:t14-r4
    t_disrupt=14 #干扰发生时刻
    N_disrupt=[18,20] #t12
    R=[[19,8,7],[9,12,22,14,4],[11,21,25],[23,10],[5,6],[26]] #干扰发生后的原调度计划
    ind_benchmark=[19,8,7,0,9,12,22,14,4,0,11,21,25,0,23,10,0,5,6,0,26,27,18,20,0]
    add_time=[1.6,1.54,1.28,2.04,6.16,0]
elif t==14 and r==6:
    #instance:t14-r6
    t_disrupt=14 #干扰发生时刻
    N_disrupt=[5,6] #t12
    R=[[19,8,7],[9,12,22,14,4],[11,21,25],[18,20],[23,10],[26]] #干扰发生后的原调度计划
    ind_benchmark=[19,8,7,0,9,12,22,14,4,0,11,21,25,0,18,20,0,23,10,0,26,27,5,6,0]
    add_time=[1.6,1.54,1.28,1.16,2.04,0]
elif t==16 and r==2:
    #instance:t16-r2
    t_disrupt=16 #干扰发生时刻
    N_disrupt=[12,22,14,4] #t12
    R=[[8,7],[21,25],[20],[23,10],[5,6],[26]] #干扰发生后的原调度计划
    ind_benchmark=[8,7,0,21,25,0,20,0,23,10,0,5,6,0,26,27,12,22,14,4,0]
    add_time=[3.01,3.99,3.57,0.04,4.16,0]
elif t==16 and r==4:
    # #instance:t16-r4
    t_disrupt=16 #干扰发生时刻
    N_disrupt=[20] #t12
    R=[[8,7],[12,22,14,4],[21,25],[23,10],[5,6],[26]] #干扰发生后的原调度计划
    ind_benchmark=[8,7,0,12,22,14,4,0,21,25,0,23,10,0,5,6,0,26,27,20,0]
    add_time=[3.02,1.66,3.99,0.04,4.16,0]
elif t==16 and r==6:
    #instance:t16-r6
    t_disrupt=16 #干扰发生时刻
    N_disrupt=[5,6] 
    R=[[8,7],[12,22,14,4],[21,25],[20],[23,10],[26]] #干扰发生后的原调度计划
    ind_benchmark=[8,7,0,12,22,14,4,0,21,25,0,20,0,23,10,0,26,27,5,6,0]
    add_time=[3.02,1.66,3.99,3.57,0.04,0]

elif t==17 and r==6:
    #instance:t16-r6
    t_disrupt=17 #干扰发生时刻
    N_disrupt=[5,6] 
    R=[[8,7],[12,22,14,4],[21,25],[20],[10],[26]] #干扰发生后的原调度计划
    ind_benchmark=[8,7,0,12,22,14,4,0,21,25,0,20,0,10,0,26,27,5,6,0]
    add_time=[2.02,0.66,2.99,2.57,6.16,0]
    
elif t==18 and r==2:
    #instance:t18-r2
    t_disrupt=18 #干扰发生时刻
    N_disrupt=[22,14,4] #t12
    R=[[8,7],[21,25],[20],[10],[5,6],[26]] #干扰发生后的原调度计划
    ind_benchmark=[8,7,0,21,25,0,20,0,10,0,5,6,0,26,27,22,14,4,0]
    add_time=[1.01,1.99,1.57,5.16,2.16,0]
elif t==18 and r==4:
    # #instance:t18-r4
    t_disrupt=18 #干扰发生时刻
    N_disrupt=[20] #t12
    R=[[8,7],[22,14,4],[21,25],[10],[5,6],[26]] #干扰发生后的原调度计划
    ind_benchmark=[8,7,0,22,14,4,0,21,25,0,10,0,5,6,0,26,27,20,0]
    add_time=[1.02,2.78,1.99,5.16,2.16,0]
elif t==18 and r==6:
    #instance:t18-r6
    t_disrupt=18 #干扰发生时刻
    N_disrupt=[5,6] 
    R=[[8,7],[22,14,4],[21,25],[20],[10],[26]] #干扰发生后的原调度计划
    ind_benchmark=[8,7,0,22,14,4,0,21,25,0,20,0,10,0,26,27,5,6,0]
    add_time=[1.02,2.78,1.99,1.57,5.16,0]


S=[(3, 13.49), (19, 15.6), (8, 19.02), (7, 24.54), (9, 15.54), (12, 17.66), (22, 20.78), (14, 24.48), (4, 29.19), (13, 13.87), (11, 15.28), (21, 19.99), (25, 25.11), (2, 9.63), (16, 11.75), (18, 15.16), (20, 19.57), (24, 10.21), (17, 11.63), (23, 16.04), (10, 23.16), (1, 10.93), (15, 13.05), (5, 20.16), (6, 25.99),(26,t_disrupt),(27,t_disrupt)] #各节点原计划送达时间
def getInd(dp=dp,N_disrupt=N_disrupt):
    #ind_benchmark=[3,19,8,7,0,13,11,21,25,0,18,20,0,10,0,15,5,6,0,26,27,9,22,12,14,4,0]
    if random.random()<=0.01:
        return ind_benchmark
    else:
        # C=R.copy()
        # R=C
        R=[[3,19,8,7],[9,12,22,14,4],[13,11,21,25],[24,17,23,10],[1,15,5,6],[26]] #干扰发生后的原调度计划
        dv=[(dp,i) for i in N_disrupt]
        for i,j in dv:
            n=random.randint(0,len(N_disrupt))
            hv=R[n]
            hv.insert(random.randint(1, len(hv)+1),i)
            hv.insert(random.randint(hv.index(i)+1, len(hv)+1),j)
    
        for i in range(len(N_disrupt)+1):
            R[i]=sorted(set(R[i]),key = R[i].index)
        ind = []
        for eachRoute in R:
            ind = ind + eachRoute + [0]
        # print(ind)
        # print("_____________________")
        return ind

# ind_test=getInd(dp=dp)
# print(ind_test)
#-----------------------------------
## 评价函数
# 染色体解码
def decodeInd(ind):
    '''从染色体解码回路线片段，每条路径都是以0为结尾'''
    indCopy = np.array(deepcopy(ind)) # 复制ind，防止直接对染色体进行改动
    idx=[i for i,x in enumerate(indCopy) if x==0]
    idx=[-1]+idx
    routes = []
    for i,j in zip(idx[0:-1], idx[1:]):
        routes.append((ind[i+1:j+1]))
    return routes

# routes_test=decodeInd(ind_test)
# print(routes_test)
def calDist(pos1, pos2):
    #计算两点之间的距离
    return (abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]))*300

def loadPenalty(routes):
    '''辅助函数，因为在交叉和突变中可能会产生不符合负载约束的个体，需要对不合要求的个体进行惩罚'''
    #计算染色体所有路线的超载数量
    overload = 0
    # 计算每条路径的负载，取max(0, routeLoad - maxLoad)计入惩罚项
    for eachRoute in routes:
        routeLoad = np.sum([dataDict['Demand'][i] for i in eachRoute])
        overload += max(0, routeLoad - dataDict['MaxLoad'])
    return overload



def CalVisitTime(route, dataDict = dataDict,S=S,t_disrupt=t_disrupt):
    #计算实际访问时间
    #S=[(3,13.39),(13,13.87),(18,15.16),(10,14.66),(15,13.05),(26,0)] #各节点原计划送达时间
    
    #线路初始送达时间
    arrivalTime=t_disrupt
    for i in S:
        if route[0]==i[0]:
            arrivalTime=i[1]
            break
    serviceTime = [arrivalTime] * len(route)
    #确定第二个节点的时间（取决于第一个节点是否有服务时间）
    if route[0]==26:
        travelTime = calDist(dataDict['NodeCoor'][route[0]], dataDict['NodeCoor'][route[1]])/dataDict['Velocity']
        serviceTime[1]=serviceTime[0]+travelTime
    else:
        travelTime =dataDict['ServiceTime']
        travelTime+= calDist(dataDict['NodeCoor'][route[0]], dataDict['NodeCoor'][route[1]])/dataDict['Velocity']
        serviceTime[1] =serviceTime[0]+travelTime
        
    if len(route)>2:        
        for i in range(1,len(route)-1):
            travelTime =dataDict['ServiceTime']
            travelTime+= calDist(dataDict['NodeCoor'][route[i]], dataDict['NodeCoor'][route[i+1]])/dataDict['Velocity']
            serviceTime[i+1] =serviceTime[i]+travelTime
    return serviceTime
  
    
def calDissatisfaction_time(a_i,a_i_0,T_max,lamda_time): #计算单点时间不满意度
    if a_i>=T_max:
        return 1
    elif 0<=a_i<=a_i_0:
        return 0
    else:
        return 1-pow((T_max-a_i)/(T_max-a_i_0),lamda_time)
    
def calDissatisfaction_damage(a_i,alpha=0.02,gamma=0.06,sigma=0.03,epsilon=0.00015,u=0.02,h=0.05,lamda_damage=lamda_damage):
    #计算单点货损不满意度
    kesi=(1-math.exp(-alpha*pow((a_i-gamma),sigma)))
    if kesi>=h:
        return 1
    elif 0<=kesi<=u:
        return 0
    else:
        return 1-pow((h-kesi)/(h-u),lamda_damage)

def calTimedissatisfaction(routes,S=S):
    
    timeTable=[]
    for i,j in zip(routes,add_time):
        tem=[]
        for m in i:
            tem.append(m+j)
        timeTable.append(tem)
    
    # timeTable=[CalVisitTime(i,dataDict = dataDict) for i in routes]
    routes=[i[0:-1] for i in routes] #去除前置仓节点
   
    routes_ind=[] #ind格式
    for i in routes:
        for j in i:
            routes_ind.append(j)
    
    timeTable=[i[0:-1] for i in timeTable] #去除返回前置仓的时间，各节点实际到达时间
    timeTable_ind=[] #ind格式
    for i in timeTable:
        for j in i:
            timeTable_ind.append(j)
    
    desiredtimeTable=[]
    for route in routes:
        tem=[]
        for m in route:
            for i in S:
                if i[0]==m:
                    tem.append(i[1])
                    break
        desiredtimeTable.append(tem) 

    desiredtimeTable_ind=[] #ind格式
    for i in desiredtimeTable:
        for j in i:
            desiredtimeTable_ind.append(j)
            
    Dissatisfaction_time=0
    for i in range(len(routes_ind)):
        Dissatisfaction_time+=calDissatisfaction_time(a_i=timeTable_ind[i],a_i_0=desiredtimeTable_ind[i],T_max=lm,lamda_time=lamda_time)
    return   Dissatisfaction_time  
            
def calDamagedissatisfaction(routes):
    timeTable=[CalVisitTime(i,dataDict = dataDict) for i in routes]
    routes=[i[0:-1] for i in routes] #去除前置仓节点
   
    routes_ind=[] #ind格式
    for i in routes:
        for j in i:
            routes_ind.append(j)
    
    timeTable=[i[0:-1] for i in timeTable] #去除返回前置仓的时间，各节点实际到达时间
    timeTable_ind=[] #ind格式
    for i in timeTable:
        for j in i:
            timeTable_ind.append(j)
           
    Dissatisfaction_damage=0
    for i in range(len(routes_ind)):
        Dissatisfaction_damage+=calDissatisfaction_damage(a_i=timeTable_ind[i])
    return   Dissatisfaction_damage     
    

def timePenalty(routes):
    '''辅助函数，Dissatisfaction 对不能按服务时间到达顾客的情况进行惩罚'''
    #计算延迟的时间总和
    timeTable=[CalVisitTime(i,dataDict = dataDict) for i in routes]
    timeDelay=0
    for i in timeTable:
        tep=[]
        for j in i:
            tep.append(max(j-30,0))
        timeDelay+=np.sum(tep)-max(i[-1]-30,0)
    return timeDelay

def calDissatisfaction_diff(route,r,h=0.6,u=0.2): #计算单条路线偏差
    #计算路线偏差
    L1=str(route)
    L2=str(r)
    diff =1-Levenshtein.jaro(L1, L2)
    if diff>=h:
        return 1
    elif 0<=diff<=u:
        return 0
    else:
        return 1-pow((h-diff)/(h-u),lamda_diff)
    return diff 
    
def calDiffdissatisfaction(routes):    
    Dissatisfaction_diff=0
    for i,j in zip(routes,R):
        Dissatisfaction_diff+=calDissatisfaction_diff(i,j)
    return   Dissatisfaction_diff
    

def calCost(routes,dataDict=dataDict):
    '''辅助函数，返回给定路径的总长度'''
    totalDistance = 0 # 记录各条路线的总长度
    for eachRoute in routes:
        # 从每条路径中抽取相邻两个节点，计算节点距离并进行累加
        for i,j in zip(eachRoute[0:-1], eachRoute[1::]):
            totalDistance += calDist(dataDict['NodeCoor'][i], dataDict['NodeCoor'][j])    
    
    if len(routes[-1])>2:
        Cost=c_d*totalDistance+f
    else:
        Cost=c_d*totalDistance
    return Cost

def calCostdissatisfaction(routes,lamda_cost=lamda_cost):
    Cost= calCost(routes,dataDict=dataDict)
    h=calCost(decodeInd(ind_benchmark),dataDict=dataDict)
    u=0.8*h
    if Cost>=h:
        return 1
    elif 0<=Cost<=u:
        return 0
    else:
        return 1-pow((h-Cost)/(h-u),lamda_cost)
    return  Dissatisfaction_cost


def evaluate(ind,c1=1,c2=1,c3=1,c4=1):
    '''评价函数'''
    routes = decodeInd(ind) # 将个体解码为路线
    return (c1*calTimedissatisfaction(routes,S=S)
            +c2*calDamagedissatisfaction(routes)
            +c3*calDiffdissatisfaction(routes)
            +c4*calCostdissatisfaction(routes,lamda_cost=lamda_cost)),

#-----------------------------------


## 交叉操作
def genChild(ind1, ind2, N_disrupt=N_disrupt):
    """Subtour Exchange Crossover """
    indx_11=[]
    for i in ind1:
        if i in N_disrupt:
            indx_11.append(i)    
    indx_21=[]
    for i in ind2:
        if i in N_disrupt:
            indx_21.append(i)
    pattern1 = dict(zip(indx_11, indx_21))
    pattern2 = dict(zip(indx_21, indx_11))
    ind_1=[pattern1[x] if x in pattern1 else x for x in ind1] #互换顺序
    ind_2=[pattern2[x] if x in pattern2 else x for x in ind2] #互换顺序
    return ind_1,ind_2

def crossover(ind1, ind2):
    '''交叉操作'''
    ind1[:], ind2[:] = genChild(ind1, ind2)
    return ind1, ind2

#-----------------------------------
## 突变操作
def opt(route,dataDict=dataDict, k=2, c1=1.0, c2=500.0):
    if 27 in route and len(route[route.index(27)::])>=4:
        route[route.index(27)+1],route[-2]=route[-2],route[route.index(27)+1]

    return route

def mutate(ind):
    '''两点交换变异'''
    routes = decodeInd(ind)
    optimizedAssembly = []
    n=random.randint(0,5)
    lsn=np.random.randint(n,size=n)
    for eachRoute in routes:
        if routes.index(eachRoute) in lsn:
            optimizedRoute = opt(eachRoute)
        else:
            optimizedRoute = eachRoute
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
toolbox.register('individual', tools.initIterate, creator.Individual, getInd)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)
toolbox.register('evaluate', evaluate)
toolbox.register('select', tools.selTournament, tournsize=3)
toolbox.register('mate', crossover)
toolbox.register('mutate', mutate)

## 生成初始族群
toolbox.popSize = 500
pop = toolbox.population(toolbox.popSize)

## 记录迭代数据
stats=tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register('min', np.min)
stats.register('avg', np.mean)
stats.register('std', np.std)
hallOfFame = tools.HallOfFame(maxsize=1)

## 遗传算法参数
toolbox.ngen = 50
toolbox.cxpb = 0.6
toolbox.mutpb = 0.4

## 遗传算法主程序
sttime=time.time()
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
print('最佳配送计划：')
pprint(distributionPlan )
# print(bestInd)
print('总不满意度：')
print(evaluate(bestInd))
print('时间不满意度：')
print(evaluate(bestInd,c2=0,c3=0,c4=0))
print('货损不满意度：')
print(evaluate(bestInd,c1=0,c3=0,c4=0))
print('路线偏差不满意度：')
print(evaluate(bestInd,c1=0,c2=0,c4=0))
print('平台配送成本不满意度：')
print(evaluate(bestInd,c1=0,c2=0,c3=0))
# print('各辆车上负载为：')
# print(calLoad(distributionPlan))
#画出路径优化图

def plot(ind,url=url,num=num):
  """
  :data 配送批次顺序, eg：data=[0,10,4,14,12,0,7,8,19,3,0,9,16,0,15,5,6,0,13,11,18,20,2,0,17,1,0]
  :url：客户数据文件路径, eg:url = "D://Onedrive/SEU/Paper_chen/Paper_2/data_figure_table/cus_data_origin.csv"
  :num 客户数
  """
  routes=decodeInd(ind)

  rawdata = pd.read_csv(url,nrows =num+1, header=None)
  # 坐标
  ID =list(rawdata.iloc[:, 0])
  X = list(rawdata.iloc[:, 1])
  Y = list(rawdata.iloc[:, 2])
  ID_disrup=27
  x_disrup=X[27]
  y_disrup=Y[27]
  Xorder_routes=[]
  for i in routes:
      x1=[]
      for j in i:
          x1.append(X[j])
      Xorder_routes.append(x1) 
  Yorder_routes=[]
  for i in routes:
      y1=[]
      for j in i:
          y1.append(Y[j])
      Yorder_routes.append(y1)   
  
  X_incomp = [X[i]  for i in ID if i in ind and i !=ID_disrup]
  Y_incomp = [Y[i]  for i in ID if i in ind and i !=ID_disrup]
  for i,j in zip(Xorder_routes,Yorder_routes):
      plt.plot(i, j, c='black', lw=1,zorder=1)
 
  plt.scatter([X[0]], [Y[0]], c='black',marker='o', zorder=3,label='前置仓')
  plt.scatter(x_disrup,  y_disrup, c='red',marker='^', zorder=3,label='受扰节点') #受扰节点
  X_comp = [X[i]  for i in ID if i not in ind and i !=ID_disrup]
  Y_comp = [Y[i]  for i in ID if i not in ind and i !=ID_disrup]
  plt.scatter(X_comp, Y_comp, c='green',marker='s',zorder=2,label='已配送节点')
  plt.scatter(X_incomp, Y_incomp, c='black',marker='*',zorder=2,label='未配送节点')
  # plt.scatter(X[,m:], Y[,m:], marker='^', zorder=3)
  plt.xticks(range(11))
  plt.yticks(range(11))
  # plt.title(self.name)
  plt.rcParams['font.sans-serif']=['Simhei'] #用来正常显示中文标签
  font1={'family':'SimHei', 
       'weight':'light', 
       'size':8}
  plt.legend(loc=2,bbox_to_anchor=(1.0,1.0),prop=font1)
  plt.show()
 
Draw_route=1
if Draw_route==1:
    plot(bestInd,url,num)

Draw_ga=1
if Draw_ga==1:
    # 画出迭代图
    minFit = logbook.select('min')
    avgFit = logbook.select('avg')
    # plt.plot(minFit, 'b-', label='Minimum Fitness')
    plt.plot(avgFit, 'r-', label='Average Fitness')
    plt.ylim(1.0,2.0)
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend(loc='best')

print("CPU-time:")
print(time.time()-sttime)
print("benchmark:")
print(evaluate(ind=ind_benchmark))
print('benchmark时间不满意度：')
print(evaluate(ind_benchmark,c2=0,c3=0,c4=0))
print('benchmark货损不满意度：')
print(evaluate(ind_benchmark,c1=0,c3=0,c4=0))
print('benchmark路线偏差不满意度：')
print(evaluate(ind_benchmark,c1=0,c2=0,c4=0))
print('benchmark平台配送成本不满意度：')
print(evaluate(ind_benchmark,c1=0,c2=0,c3=0))
    