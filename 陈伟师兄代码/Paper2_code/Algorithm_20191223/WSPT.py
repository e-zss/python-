import time
from numpy import *
from tool import makespan
from tool import makespan_left
from Order_batching import batch_Quantity
from VRPTW_ORTools import batch_delivery_time,batch_route_delivery_time,batch_route_dis
from pick_time_S_shape import blockA_picktime,blockB_picktime,blockC_picktime,blockD_picktime

HGA2_start_time = time.time()
"""输入参数:由于即时调度时间太短，现将参数放大十倍"""
#订单履行时间期限
of_duetime=30*10 #30min
#单位配送成本
delivery_cost_perMin=0
delivery_cost_perMeter=0.005 #5元/公里（工资、油耗、折旧）
#线路配送里程
delivery_dis=batch_route_dis
#配送时间
# delivery_time=[i*10 for i in [17,20,16,17]]  #订单配送时间
delivery_time=[i*10 for i in batch_delivery_time]  #订单配送时间
# route_delivery_time=[i*10 for i in [22,27,20,23]] #线路配送时间
route_delivery_time=[i*10 for i in batch_route_delivery_time] #线路配送时间
#订单拣选截止时间
op_duetime=[of_duetime-j for j in delivery_time]
#每个品项打包分类时间
packing_time_perItem=0.05*10 #每个品项需要0.05分钟打包时间(单个订单打包时间控制在1分钟以内)
#单位时间订单拣选成本
op_cost_perMin=1.5
#单位时间违约成本
over_duetime_cost_perMin=2

#批次订单打包分类时间
batch_packingtime=[math.floor(i*packing_time_perItem*10)/10 for i in batch_Quantity]
#传送时间
transfer_time=round(3/4,1)*10

# data=array([[1,2,3,4],[0.6,1.2,0.6,1.1]*10,[1.1, 0.7, 0, 1.1]*10,[0.6, 0.7, 1.1, 0.9]*10,[0.6, 0.9, 0.9, 0.8]*10]) #n行m列，第一行工件编号，其他是加工时间

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





