import time
from numpy import *
from tool import makespan
from tool import makespan_left
from Parameter import data,transfer_time,op_cost_perMin,op_duetime,over_duetime_cost_perMin,batch_packingtime,delivery_time
from Parameter import delivery_cost_perMeter,route_delivery_time,delivery_dis,delivery_cost_perMin,batch_delivery_cost

HGA2_start_time = time.time()
N_max=data.shape[1]
index=list(range(1,N_max+1))
data1=data[1:,:].sum(axis=0)
tup=list(zip(index,data1))
tup.sort(key=lambda x:x[1])
seq_WSPT=[i[0] for i in tup]
data_best_all_WSPT=data[:,array(seq_WSPT)-1]
order=[i-1 for i in seq_WSPT]
"""时间类"""
#拣货开始时间
op_start_time=makespan_left(data_best_all_WSPT,transfer_time)
#拣货完成时间
makespan_time=makespan(data_best_all_WSPT,transfer_time)
#各分区订单抵达后台时间
batch_arrive_time=array([i+transfer_time for i in (makespan(data_best_all_WSPT,transfer_time)[-1, :])])
# 批次订单开始配送时间=抵达后台时间+打包分类时间
batch_delivery_start_time = batch_arrive_time + array([batch_packingtime[i] for i in order])
#批次订单等待时间=抵达后台时间-传送时间-拣货时间
batch_wait_time=array(batch_arrive_time)-4*transfer_time-array(data[1:].sum(axis=0))

#批次订单违约时间
batch_over_duetime = []
for j in list(array(batch_delivery_start_time) - array([op_duetime[i] for i in order])):
    if j < 0:
        batch_over_duetime.append(0)
    else:
        batch_over_duetime.append(j)
#订单完成时间
batch_of_time=array(batch_delivery_start_time)+array([delivery_time[i] for i in order]) 


"""成本类：将时间/10"""
#各批次订单拣选成本
batch_op_cost=array(batch_delivery_start_time)/10*op_cost_perMin
#各批次订单违约成本
batch_overduetime_cost=array(batch_over_duetime)/10*over_duetime_cost_perMin
#批次配送成本
batch_delivery_cost=array([delivery_cost_perMeter*i for i in delivery_dis])+array([delivery_cost_perMin*i/10 for i in route_delivery_time])
#订单履行成本
batch_of_cost=batch_op_cost+batch_overduetime_cost+batch_delivery_cost
#订单总履行成本
TC=round(batch_op_cost.sum()+batch_overduetime_cost.sum()+batch_delivery_cost.sum(),2)

print(data_best_all_WSPT)
print("Object:",TC)





