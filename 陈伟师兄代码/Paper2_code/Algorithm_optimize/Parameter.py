import math
from numpy import *

"""输入参数:由于即时调度时间太短，现将参数放大十倍"""
url="D://Onedrive/SEU/Paper_chen/Paper_2/data_figure_table/cus_data_origin.csv"
num=25
#订单履行时间期限
of_duetime=30*10 #30min
#单位配送成本
delivery_cost_perMin=0
delivery_cost_perMeter=0.005 #5元/公里（工资、油耗、折旧）
delivery_cost_fix=3 #每辆车派车成本
#每个品项打包分类时间
packing_time_perItem=0.05*10 #每个品项需要0.05分钟打包时间(单个订单打包时间控制在1分钟以内)
#单位时间订单拣选成本
op_cost_perMin=1.5
#单位时间违约成本
over_duetime_cost_perMin=2
#传送时间
transfer_time=round(3/4,1)*10
