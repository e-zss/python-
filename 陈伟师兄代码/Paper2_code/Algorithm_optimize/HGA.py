import time
from numpy import *
from NEH import neh
import GA
from tool import makespan
from tool import makespan_left
from Parameter import data,transfer_time,op_cost_perMin,op_duetime,over_duetime_cost_perMin,batch_packingtime,delivery_time
from Parameter import delivery_cost_perMeter,route_delivery_time,delivery_dis,delivery_cost_perMin,batch_delivery_cost

HGA2_start_time = time.time()

Seq_neh,Value_neh=neh(data,transfer_time,draw=0)
data_best_all = data[:, Seq_neh - 1]
Seq_ga=GA.ga_fsp_new(data,op_cost_perMin=op_cost_perMin,
                     op_duetime=op_duetime,
                     over_duetime_cost_perMin=over_duetime_cost_perMin,
                     batch_packingtime=batch_packingtime,
                     transfer_time=transfer_time,
                     draw=222) #draw分别甘特图、适应度图、动态适应度图，1表示绘制，2表示不绘制

data_best_all_GA=data[:,Seq_ga-1]
order=[i-1 for i in data_best_all_GA[0,:]]
"""时间类"""
#拣货开始时间
op_start_time=makespan_left(data_best_all_GA,transfer_time)
#拣货完成时间
makespan_time=makespan(data_best_all_GA,transfer_time)
#各分区订单抵达后台时间
batch_arrive_time=array([i+transfer_time for i in (makespan(data_best_all_GA,transfer_time)[-1, :])])
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
#订单履行成本
batch_of_cost=batch_op_cost+batch_overduetime_cost+batch_delivery_cost
#订单总履行成本
TC=round(batch_op_cost.sum()+batch_overduetime_cost.sum()+batch_delivery_cost.sum(),2)

print("data(picking+setup):",data)
print("data_best_all_GA:",data_best_all_GA)
print("makespan_time:",makespan_time)
print("data(total picking+setup):",data[1:].sum(axis=0))
# print("batch_wait_time:",batch_wait_time)
print("op_start_time:",op_start_time)

print("Total makespan_time:",makespan_time[-1,:].sum())
print("batch_arrive_time:",batch_arrive_time)
print("batch_delivery_start_time:",batch_delivery_start_time)
print("batch_over_duetime:",batch_over_duetime)
print("batch_of_time:",batch_of_time)
print("———————————————————————————————————— ")

# print("batch_op_cost:",batch_op_cost,sum(batch_op_cost))
# print("batch_overduetime_cost:",batch_overduetime_cost,sum(batch_overduetime_cost))
# print("batch_delivery_cost:",batch_delivery_cost,sum(batch_delivery_cost))
# print("batch_of_cost:",batch_of_cost)
# print("object value in OP:",sum(batch_op_cost)+sum(batch_overduetime_cost))
# print("Object:",TC)
# print("——————————————————————————————————————————————————————————————————————————")
print(sum(batch_op_cost),sum(batch_delivery_cost),sum(batch_overduetime_cost))
print("Object:",TC)
print("HGA-2 time used:",time.time()-HGA2_start_time)





