import random
import pandas as pd
from Routing_plot import plot
from Order_generate import Item
from CVRPTWVS_ortools import vehicle_routing,num,url

"""输入配送批次数据"""
# data = [0,3,19,8,7,0,10,6,4,14,12,9,16,0,13,11,18,20,2,0,17,1,15,5,0]
data=vehicle_routing
# url = "D://Onedrive/SEU/Paper_chen/Paper_2/data_figure_table/cus_data_origin.csv"
# num=20
"""绘制VRP配送路径图"""
# plot(data,url,num)

"""生成批次订单"""
def generator_batch(data):
    index_0 = [i for i, x in enumerate(data) if x == 0]
    batch = []
    for j in range(len(index_0) - 1):
        batch.append(data[index_0[j]:index_0[j + 1]])
    return batch


"""订单批次品项合并"""
def generator_batch_item(batch):
    batch_item = []
    for i in batch:
        temlist = []
        for j in range(len(i)):
            temlist += Item[i[j]]
            temlist.sort()
        batch_item.append(temlist)
    return batch_item

batch=generator_batch(data)
batch_item=generator_batch_item(batch)
batch_Quantity=[]
for i in batch_item:
    batch_Quantity.append(len(i))