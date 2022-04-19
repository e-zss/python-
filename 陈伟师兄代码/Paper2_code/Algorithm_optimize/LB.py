# -*- coding: utf-8 -*-
from numpy import *
from Parameter import data,transfer_time,op_cost_perMin,op_duetime,over_duetime_cost_perMin,batch_packingtime,delivery_time
from Parameter import delivery_cost_perMeter,route_delivery_time,delivery_dis,delivery_cost_perMin,batch_delivery_cost

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
print("LB in op：",LB-lb4)
print("LB：",LB)


