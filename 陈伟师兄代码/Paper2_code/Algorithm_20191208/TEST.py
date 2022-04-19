import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data=pd.DataFrame({"Layout":["三室一厅","三房间2厅","三房间一厅"]})
data["Layout"]=data["Layout"].str.replace("房间","室")
# data.applymap(lambda x:str(x).replace("房间","室"))

print(data)