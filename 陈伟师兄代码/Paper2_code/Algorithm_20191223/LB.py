# -*- coding: utf-8 -*-
from numpy import *
from VRPTW_ORTools import batch_delivery_time,batch_route_delivery_time,batch_route_dis
from pick_time_S_shape import blockA_picktime,blockB_picktime,blockC_picktime,blockD_picktime
from Order_batching import batch_Quantity

transfer_time=8
packing_time_perItem=0.05*10
op_cost_perMin=1.5
batch_packingtime=[math.floor(i*packing_time_perItem*10)/10 for i in batch_Quantity]
#订单分区处理时间=拣货时间+准备时间
data1=array([blockA_picktime,blockB_picktime,blockC_picktime,blockD_picktime])
set_up_time=0.15
data=[]
for i in data1:
    tem = []
    for j in i:
        if j>0:
            tem.append(int((j+set_up_time)*10))
        else:
            tem.append(0)
    data.append(tem)
data=insert(data,0,array(list(range(1,len(blockA_picktime)+1))),axis=0)
#data: m行n列，第1行工序编号，值:订单分区处理时间=拣货时间+准备时间
delivery_cost_perMeter=0.005
delivery_dis=batch_route_dis
#批次配送成本
batch_delivery_cost=array([delivery_cost_perMeter*i for i in delivery_dis])
"""目标函数下界值计算"""
M_max=4
B_max=data.shape[1]
B=range(B_max) #批次订单b集合
M=range(M_max) #分区m集合
Pick_time=data[1:,:]

# lb1=Pick_time.sum()/M_max

lb1=0
for b in B:
    tem=0
    for m in range(M_max):
        tem+=Pick_time[m][b]
    tem*=1/B_max
    lb1+=tem
lb1*=1/M_max

lb2=0
for b in B:
    tem=0
    for m in range(M_max-1):
        tem+=Pick_time[m][1]
    tem*=1/B_max
    lb2+=tem
lb2*=1/M_max

        
lb3=0
for b in B:
    tem=0
    for m in range(1,M_max):
        tem+=(m-1)*Pick_time[m][b]
    tem*=1/B_max
    lb3+=tem
lb3*=1/M_max

   

lb4=batch_delivery_cost.sum()

LB=(lb1+lb2+lb3+B_max*M_max*transfer_time+sum(batch_packingtime))*op_cost_perMin/10+lb4
print("B_max:",B_max)
print("LB：",LB)
print("LB in op：",LB-lb4)

