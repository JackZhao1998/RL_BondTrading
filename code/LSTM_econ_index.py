import numpy as np
import math
import pandas as pd
from pandas import DataFrame
import numpy as np
import statistics as stats
import matplotlib.pyplot as plt
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#import datas
df1=pd.read_excel('/Users/mac/Desktop/Reinforcement Learning Bond/datas/US_econ_index/非农就业人数.xls')
df2=pd.read_excel('/Users/mac/Desktop/Reinforcement Learning Bond/datas/US_econ_index/个人消费环比.xls')
df3=pd.read_excel('/Users/mac/Desktop/Reinforcement Learning Bond/datas/US_econ_index/耐用品订单.xls')
df4=pd.read_excel('/Users/mac/Desktop/Reinforcement Learning Bond/datas/US_econ_index/失业率.xls')
df5=pd.read_excel('/Users/mac/Desktop/Reinforcement Learning Bond/datas/US_econ_index/制造业PMI.xls')
df6=pd.read_excel('/Users/mac/Desktop/Reinforcement Learning Bond/datas/US_econ_index/美国劳动力人数.xls')
df7=pd.read_csv('/Users/mac/Desktop/Reinforcement Learning Bond/datas/TNX.csv')


data=pd.merge(df1,df6,how='left',on=None,left_on='日期',right_on='日期')
data['非农人数占比']=data['美国_就业人数_非农']/data['美国_劳动力人口_总计_季调']
data=pd.merge(data,df2,how='left',on=None,left_on='日期',right_on='日期')
data=data.dropna(subset=['美国_就业人数_非农','美国_劳动力人口_总计_季调','美国_个人消费支出_环比_季调'])

date_index=pd.date_range(str(data['日期'][len(data['日期'])]),str(data['日期'][1]),freq='D')
dates=[]
df=dict()
for date in date_index:
    dates.append((str(date)[0:10]))
df['date']=dates
df=DataFrame(df)
df=pd.merge(df,df7,how='left',on=None,left_on='date',right_on='date')
df=df.fillna(method='ffill')
data=pd.merge(data,df,how='left',on=None,left_on='日期',right_on='date')
data=data.drop(columns=['美国_就业人数_非农','美国_劳动力人口_总计_季调','date'])
data=data.dropna()

#reverse the order of dataframe
data=data.iloc[::-1]
re_order={}
order=[]
for item in data['日期']:
    order.append(item)
re_order['日期']=order
re_order=pd.DataFrame(re_order)
data=pd.merge(re_order,data,how='left',on=None,left_on='日期',right_on='日期')

#calculate yield difference and signal
yield_difference=[]
i=0
while i<len(data['yield'])-1:
    yd=(data['yield'][i+1]-data['yield'][i])/data['yield'][i]*100
    yield_difference.append(yd)
    i += 1
yield_difference.append(0)

signal=[]
for change in yield_difference:
    if change > 0:
        signal.append(1)
    elif change <0:
        signal.append(-1)
    else:
        signal.append(0)

data['yield_change']=yield_difference
data['signal']=signal
data=data.rename(columns={'美国_个人消费支出_环比_季调':'个人消费支出环比'})


print(data)
'''
set up training models
'''
'''
training_set=
test_set=

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(784, 64) #define input and output
        self.fc2=nn.Linear(64, 64)
        self.fc3=nn.Linear(64, 64)
        self.fc4=nn.Linear(64, 10)

    def forward(self,x): #define x route, 'F.relu' as the activation function
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        x=self.fc4(x)
        return F.log_softmax(x,dim=1)
net=Net()

optimizer=optim.Adam(net.parameters(),lr=0.001)
'''
