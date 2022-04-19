
from time import time
from gurobipy import *
from numpy import *
from Parameter import data,transfer_time,op_cost_perMin,op_duetime,over_duetime_cost_perMin,batch_packingtime,delivery_time
from Parameter import delivery_cost_perMeter,route_delivery_time,delivery_dis,delivery_cost_perMin,batch_delivery_cost


st_time=time()
# 创建模型
M_max=4
B_max=data.shape[1]
B=range(B_max) #批次订单b集合
M=range(M_max) #分区m集合
G=99999
Pick_time=data[1:,:]

MODEL = Model("Order picking Schedule")

# 创建变量
y= MODEL.addVars(range(B_max),range(B_max),vtype=GRB.BINARY,name="y") #y_bk
c= MODEL.addVars(range(B_max),range(M_max),vtype=GRB.CONTINUOUS,name="c") #c_km
Z=MODEL.addVar(vtype=GRB.CONTINUOUS,name="Z")
z=MODEL.addVars(range(B_max),vtype=GRB.CONTINUOUS,name="z") #各批次的违约时间z_k

# 更新变量环境
MODEL.update()

# 创建目标函数CONTINUOUS
MODEL.setObjective(
    op_cost_perMin/10*(quicksum(c[k,M_max-1] for k in B)+B_max*transfer_time+sum(batch_packingtime))
    +Z, sense=GRB.MINIMIZE)


# 创建约束条件
MODEL.addConstrs((quicksum(y[b,k] for k in B) == 1 for b in B),name="s.t.4")
MODEL.addConstrs((quicksum(y[b,k] for b in B) == 1 for k in B),name="s.t.5")                 
MODEL.addConstr((c[0,0]>=quicksum(y[b,0]*Pick_time[0][b] for b in B)),name="s.t.6")                 
MODEL.addConstrs((c[0,m]>=quicksum(y[b,0]*Pick_time[m][b] for b in B)+c[0,m-1]+transfer_time for m in range(1,M_max)))                 
MODEL.addConstrs((c[k,0]>=quicksum(y[b,k]*Pick_time[0][b] for b in B)+c[k-1,0] for k in range(1,B_max)))                 

MODEL.addConstrs((c[k,m]>=c[k-1,m]+quicksum(y[b,k]*Pick_time[m][b] for b in B) for k in range(1,B_max) for m in range(1,M_max)))                 
MODEL.addConstrs((c[k,m]>=c[k,m-1]+transfer_time+quicksum(y[b,k]*Pick_time[m][b] for b in B) for k in range(1,B_max) for m in range(1,M_max)))                 
MODEL.addConstrs((c[k,m]>=0 for k in B for m in M),name="s.t.22") 
                
MODEL.addConstrs((z[k]>=0 for k in B),name="s.t.24")                 
MODEL.addConstrs((z[k]>=over_duetime_cost_perMin/10 *quicksum(y[b,k]*(c[k,M_max-1]+transfer_time+batch_packingtime[b]-op_duetime[b]) for b in B) for k in B),name="s.t.25")                 
MODEL.addConstr(Z==quicksum(z[k] for k in B)*over_duetime_cost_perMin/10,name="s.t.26")   

# 执行最优化
MODEL.optimize()
#输出模型
MODEL.write("fsp.lp")

# 输出约束
# print(MODEL.getConstrs())

# 查看变量取值
for var in MODEL.getVars():
   print(f"{var.varName}: {round(var.X, 3)}")
#查看最优目标函数值
print("Optimal Objective Value", round(MODEL.objVal,2))
print("Gurobi used time:",time()-st_time)