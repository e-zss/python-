import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tool import makespan
transfer_time=8
# data_best_all_GA: [[ 3  1  2  4]
#  [ 7  7 13 12]
#  [ 0 12  8 12]
#  [12  7  8 10]
#  [10  7 10  9]]
# makespan_time: [[ 3  1  2  4]
#  [ 7 14 27 39]
#  [15 34 43 59]
#  [35 49 59 77]
#  [53 64 77 94]]

data=np.array([[1,2,3,4],[7,7,12,13],[0,12,12,8],[12,7,9,10],[10,7,9,10],[10,7,9,10]])
data1=data[1:,:].sum(axis=1)
index=list(range(1,len(data1)+1))
tup=list(zip(index,data1))
tup.sort(key=lambda x:x[1])
seq_WSPT=[i[0] for i in tup]
print(data1)