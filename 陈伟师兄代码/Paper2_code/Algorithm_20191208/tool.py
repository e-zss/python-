
# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from random import choice
from matplotlib import cm

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.family']='SimHei'
plt.rcParams['axes.unicode_minus'] = True
# plt.style.use("fivethirtyeight")
sns.set_style({'font.sans-serif':['simhei','Arial']})

sns.set()
sns.set_style("whitegrid")


# ax = plt.gca()
# [ax.spines[i].set_visible(False) for i in ["top", "right"]]


def crtd_fs(n, m, low, high):
	"""
	随机生成流水车间作业数据
	:param n: 工序数目
	:param m: 机器数目
	:param low: 加工时间最小值
	:param high: 加工时间最大值
	:return:data_fs
	"""
	data_fs = np.zeros([m + 1, n], dtype=int)
	data_fs[0] = np.random.permutation(np.arange(1, n + 1))
	data_fs[1:] = np.random.randint(low, high + 1, [m, n])
	data_fs = data_fs[:, np.argsort(data_fs[0])]
	return data_fs


def crtd_hfs(n, s, low, high):
	"""
	随机生成混合流水车间作业数据
	:param n: 工件数目
	:param s: 2行，第1行工序，第2行对应的并行机数目
	:param low: 加工时间最小值
	:param high: 加工时间最大值
	:return:data_hfs
	"""
	data_hfs = np.zeros([n, np.sum(s[1]) + 1], dtype=int)
	data_hfs[:, 0] = np.random.permutation(np.arange(1, n + 1))
	data_hfs[:, 1:] = np.random.randint(low, high + 1, [n, np.sum(s[1])])
	data_hfs = data_hfs[np.argsort(data_hfs[:, 0]), :]
	return data_hfs


def crtd_moore(n, low, high):
	"""
	随机生成单机作业数据
	:param n: 工序数目
	:param low: 加工时间最小值
	:param high: 加工时间最大值
	:return:data_moore
	"""
	data_moore = np.zeros([3, n], dtype=int)
	data_moore[0] = np.random.permutation(np.arange(1, n + 1))
	data_moore[2] = np.random.randint(low, high + 1, n)
	data_moore[1] = data_moore[2] + np.random.randint(n * low, n*low + high + 1, n)
	data_moore = data_moore[:, np.argsort(data_moore[0])]
	return data_moore



def makespan(data,transfer_time):
	"""
	工件某工序的完成时间
	:param data: m行n列，第1行工序编号，值加工时间
	设备-i
	工件-j
	transfer_time：相邻分区的传送时间
	set_up_time：开始拣选前的准备时间（订单确认等）
	:return:makespan
	"""
	makespan = np.zeros_like(data)
	for i in range(makespan.shape[0]):
		for j in range(makespan.shape[1]):
			if i == 0:
				makespan[i, j] = data[i, j]
			if i == 1:
				makespan[i, j] = np.sum(data[1, :j + 1])
			if j == 0 and i != 0:
				makespan[i, j] = np.sum(data[1:i + 1, j])+(i-1)*transfer_time
	for i in range(2, makespan.shape[0]):
		for j in range(1, makespan.shape[1]):
			makespan[i, j] = data[i, j] + max(makespan[i, j - 1], makespan[i - 1, j]+transfer_time)
	return makespan


def makespan_value(data,transfer_time):
	"""
	最大生产流程时间
	:param data: m行n列，第1行工序编号，值加工时间
	:return:makespan_value
	"""
	makespan_value = makespan(data,transfer_time)[-1, -1]
	return makespan_value


def makespan_left(data,transfer_time):
	"""
	工件某工序的开始时间
	:param data: m行n列，第1行工序编号，值加工时间
	:return:left
	"""
	left = makespan(data,transfer_time) - data
	left[0] = data[0]
	return left

def gatt(data,transfer_time,):
	"""
	甘特图
	:param data: m行n列，第1行工序编号，值加工时间
	:return:
	"""
	makespan = makespan_value(data,transfer_time)
	left = makespan_left(data,transfer_time)
	font_size = 8
	if data.shape[1] <= 30:
		font_size = 10
	if data.shape[0] <= 4:
		font_size = 10
	for i in range(1, data.shape[0]):
		for j in range(data.shape[1]):
			if data[i, j] != 0:
				# plt.barh(i, data[i, j], left=left[i, j],color=Set2_8.mpl_colormap)
				if j==0:
					plt.barh(i, data[i, j], left=left[i, j],color="lightcoral")  # color=["lightcoral","darkkhaki","seagreen","gray"]
					plt.text(left[i, j] + data[i, j] / 8, i, r"$B_{%s}$" % int(data[0, j]), color="k", size=font_size)
				elif j==1:
					plt.barh(i, data[i, j], left=left[i, j],color="darkkhaki")  # color=["lightcoral","darkkhaki","seagreen","gray"]
					plt.text(left[i, j] + data[i, j] / 8, i, r"$B_{%s}$" % int(data[0, j]), color="k", size=font_size)
				elif j==2:
					plt.barh(i, data[i, j], left=left[i, j],color="seagreen")  # color=["lightcoral","darkkhaki","seagreen","gray"]
					plt.text(left[i, j] + data[i, j] / 8, i, r"$B_{%s}$" % int(data[0, j]), color="k", size=font_size)
				elif j == 3:
					plt.barh(i, data[i, j], left=left[i, j],color="gray")  # color=["lightcoral","darkkhaki","seagreen","gray"]
					plt.text(left[i, j] + data[i, j] / 8, i, r"$B_{%s}$" % int(data[0, j]), color="k", size=font_size)
				elif j == 4:
					plt.barh(i, data[i, j], left=left[i, j],color="steelblue")
					plt.text(left[i, j] + data[i, j] / 8, i, r"$B_{%s}$" % int(data[0, j]), color="k", size=font_size)
				elif j == 5:
					plt.barh(i, data[i, j], left=left[i, j],color="teal")
					plt.text(left[i, j] + data[i, j] / 8, i, r"$B_{%s}$" % int(data[0, j]), color="k", size=font_size)
				elif j == 6:
					plt.barh(i, data[i, j], left=left[i, j],color="tomato")
					plt.text(left[i, j] + data[i, j] / 8, i, r"$B_{%s}$" % int(data[0, j]), color="k", size=font_size)
				elif j == 7:
					plt.barh(i, data[i, j], left=left[i, j],color="lightseagreen")
					plt.text(left[i, j] + data[i, j] / 8, i, r"$B_{%s}$" % int(data[0, j]), color="k", size=font_size)
				elif j == 8:
					plt.barh(i, data[i, j], left=left[i, j],color="lightcoral")
					plt.text(left[i, j] + data[i, j] / 8, i, r"$B_{%s}$" % int(data[0, j]), color="k", size=font_size)
				elif j == 9:
					plt.barh(i, data[i, j], left=left[i, j],color="navy")
					plt.text(left[i, j] + data[i, j] / 8, i, r"$B_{%s}$" % int(data[0, j]), color="k", size=font_size)
				else:
					plt.barh(i, data[i, j], left=left[i, j],color="peru")  # color=["lightcoral","darkkhaki","seagreen","gray"]
					plt.text(left[i, j] + data[i, j] / 8, i, r"$B_{%s}$" % int(data[0, j]), color="k", size=font_size)

	plt.plot([makespan] * (left.shape[0] + 1), range(left.shape[0] + 1), 'r--', alpha=0.8,color="firebrick")
	plt.text(makespan - 0.5, 0, r"$Makespan={%s}$" % np.round(makespan, 2), ha="right", fontdict={"size": 8})
	plt.yticks(range(1, left.shape[0]), range(1, left.shape[0]))
	plt.tick_params(labelsize=8)
	plt.ylabel(r"Picker", fontsize=11)
	plt.xlabel(r"Time*10 (min)", fontsize=11)
