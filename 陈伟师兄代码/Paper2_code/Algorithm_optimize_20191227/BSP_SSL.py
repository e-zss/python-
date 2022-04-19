# -*- coding: utf-8 -*-


from time import time
from numpy import *
from Order_batching import getBatch
from pick_time_S_shape import getBatchPicktime
from Parameter import t_convey,c_op,c_od,c_d,delivery_cost_perMin,lm,t_pack
from tool import makespan


ind_from=1  #0--from ortools;1--from GA ；2--gurobi
""" ind ：例如[0,1,2,5,6,0,4,7,8,0,9,11,12,0]"""

if ind_from==2:
    from CVRPDV_Gurobi import route,var_a,C_d
    ind=route
    batch,batch_item=getBatch(ind)
    last_order_perBatch=[i[-2] for i in batch]
    ODtime=[var_a[i-1] for i in last_order_perBatch]
    C_d_Bat=0 #待确认

elif ind_from==0:
    from CVRPDV_ortools import vehicle_routing,batch_delivery_time,batch_route_dis
    ind=vehicle_routing
    batch,batch_item=getBatch(ind)
    ODtime=batch_delivery_time #每个批次订单最后一个订单的交付时间
    C_d=sum(batch_route_dis)*c_d
    C_d_Bat=array(batch_route_dis)*c_d
else:
    from CVRPDV_GA import bestInd,ODtime,C_d,C_d_Bat
    ind=bestInd
    batch,batch_item=getBatch(ind)

    
batch_Quantity=[len(i) for i in batch_item]
data=getBatchPicktime(batch_item)
batch_packingtime=[t_pack*i for i in batch_Quantity]
op_duetime=[lm-i for i in ODtime]


"""bsp"""
N_max=data.shape[1]
index=list(range(1,N_max+1))


data1=data[1:,:].sum(axis=0)
tup=list(zip(index,data1))
tup.sort(key=lambda x:x[1])
Seq_SPT=array([i[0] for i in tup])
data_best_all=data[:,Seq_SPT-1]
TPicktime_SPT=makespan(data_best_all)[-1,:].sum()

tup=list(zip(index,C_d_Bat))
tup.sort(key=lambda x:x[1],reverse=True)
Seq_LDT=array([i[0] for i in tup])
data_best_all=data[:,Seq_LDT-1]
TPicktime_LDT=makespan(data_best_all)[-1,:].sum()

if TPicktime_SPT<=TPicktime_LDT:
    Seq=Seq_SPT
else:
    Seq=Seq_LDT


