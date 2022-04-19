from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
X = [round(i*0.1,1) for i in range(11)]
Y = [round(i*0.1,1) for i in range(10,-1,-1)]
Z_H2=[1,0.994,0.988,0.982,0.976,0.97,0.964,0.958,0.952,0.946,0.94]
Z_TS=[1,1.004,1.007,1.011,1.014,1.018,1.021,1.024,1.028,1.031,1.035]
m=[0,0.96,1.93,2.91,3.9,4.9,5.92,6.95,7.99,9.05,10.12]
gap1=[round(i*0.1,2) for i in m]
gap=m

ax1 = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
ax1.plot(X, Y, Z_H2, c='r',marker='o')
ax1.plot(X, Y, Z_TS, c='g',marker='^')
# ax1.plot(X, Y, gap1, c='b',marker='*')
ax1.set_zlabel('目标函数值')  # 坐标轴
ax1.set_ylabel('Tc权重系数')
ax1.set_xlabel('Tmax权重系数')
plt.legend(["H-2","TS"])

# ax4 = plt.subplot(122, projection='3d')  # 创建一个三维的绘图工程
# # ax4.plot(X, Y, Z_H2, c='r',marker='o')
# # ax4.plot(X, Y, Z_TS, c='g',marker='^')
# ax4.plot(X, Y, gap, c='b',marker='*')
# ax4.set_zlabel('Objective value')  # 坐标轴
# ax4.set_ylabel('TC_weight')
# ax4.set_xlabel('Tmax_weight')



# ax2 = plt.subplot(121)
# ax2.plot(X,Z_H2,c='r',marker='o')
# ax2.plot(X,Z_TS,c='g',marker='^')
# ax2.set_xlabel('Tmax_weight')
# ax2.set_ylabel('Objective value')

# ax3 = plt.subplot(122)
# ax3.plot(Y,Z_H2,c='r',marker='o')
# ax3.plot(Y,Z_TS,c='g',marker='^')
# ax3.set_xlabel('TC_weight')
# ax3.set_ylabel('Objective value')

plt.show()
