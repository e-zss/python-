import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set()#切换到seaborn的默认运行配置
sns.set_style("whitegrid")

# plt.style.use('seaborn-pastel') #选择美化样式 ['bmh', 'classic', 'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 'grayscale', 'seaborn-bright', 'seaborn-colorblind', 'seaborn-dark-palette', 'seaborn-dark', 'seaborn-darkgrid', 'seaborn-deep', 'seaborn-muted', 'seaborn-notebook', 'seaborn-paper', 'seaborn-pastel', 'seaborn-poster', 'seaborn-talk', 'seaborn-ticks', 'seaborn-white', 'seaborn-whitegrid', 'seaborn', 'Solarize_Light2', 'tableau-colorblind10', '_classic_test']

url = "D://Onedrive/SEU/Paper_chen/Paper_1/data_figure_table/cus_data_origin.csv"
rawdata = pd.read_csv(url,nrows =19, header=None)
# 坐标
X = list(rawdata.iloc[:, 1])
Y = list(rawdata.iloc[:, 2])
def plot(data):
  Xorder = [X[i] for i in data]
  Yorder = [Y[i] for i in data]
  plt.plot(Xorder, Yorder, c='black', lw=1,zorder=1)
  plt.scatter(X, Y, c='black',marker='*',zorder=2)
  plt.scatter([X[0]], [Y[0]], c='black',marker='o', zorder=3)
  # plt.scatter(X[-m:], Y[-m:], marker='^', zorder=3)
  plt.xticks(range(11))
  plt.yticks(range(11))
  # plt.title(self.name)
  plt.show()


data=[0,3,8,0,12,14,4,0,9,16,0,2,18,0,15,5,0,10,6,0,13,11,0,17,1,7,0]
plot(data)
