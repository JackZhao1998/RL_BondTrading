import numpy as np
import math
import pandas as pd
import numpy as np
import statistics as stats
import matplotlib.pyplot as plt

#import data
df=pd.read_csv('/Users/mac/Desktop/Reinforcement Learning Bond/datas/bond_backtest.csv')
df.fillna(method='ffill')
dates=df['date']
rates=df['yield']
closes=df['close']

#put data into structured dictionaries and an index list of years
years_list=list()
rate_datas=dict()
closes_datas=dict()
for date in dates:
    date=str(date)
    if date[0:4] not in rate_datas:
        rate_datas[date[0:4]]=dict()
        closes_datas[date[0:4]]=dict()
        years_list.append(date[0:4])
i=0;
while i < len(dates):
    match=dates[i][0:4]
    rate_datas[match][dates[i]]=float(rates[i])
    closes_datas[match][dates[i]]=float(closes[i])
    i=i+1

rate_lists=dict()
closes_lists=dict()
for year in years_list:
    rate=list()
    close=list()
    for date in rate_datas[year]:
        rate.append(rate_datas[year][date])
        close.append(closes_datas[year][date])
    rate_lists[year]=rate
    closes_lists[year]=close

# years_list:a list of all years
# rate_dates:a dictionary of dictionaries of rates and dates
# closes_datas:a dictionary of dictionaries of closes and dates
# rate_lists: a dictionary of lists of rates
# closes_lists: a dictionary of lists of closes

#define Q matrix
dimension=15
q_values=np.zeros((dimension,dimension,2))
q_TD=q_values

#set up training sets
'''
['2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019','2020','2021']
'''
training_years=['2017','2018','2019','2020','2021']

#set up lists
training_set_rates=list()
training_set_Deltas=list()
training_set_obs=list()
training_set_closes=list()

#set up rate&closes
for year in training_years:
    i=0
    while i<len(rate_lists[year]):
        training_set_rates.append(rate_lists[year][i])
        training_set_closes.append(closes_lists[year][i])
        i=i+1

#calculate required training sets
delta_itv=5# 1 week
global max_ob,min_ob,itv_ob
max_ob=max(training_set_rates)
min_ob=min(training_set_rates)
itv_ob=(max_ob-min_ob)/dimension
Deltas=list()
i=delta_itv #for 20 trading days
while i<len(training_set_rates):
    delta=(training_set_rates[i]-training_set_rates[i-delta_itv])
    Deltas.append(delta)
    i=i+1
global max_delta,min_delta,itv_delta
max_delta=max(Deltas)
min_delta=min(Deltas)
itv_delta=(max_delta-min_delta)/dimension

#modify rates&closes
training_set_rates=training_set_rates[delta_itv:]
training_set_closes=training_set_closes[delta_itv:]
training_set_obs=training_set_rates
training_set_Deltas=Deltas

#define a data input function
def get_location(delta,obs):
    global max_delta,min_delta,itv_delta,max_ob,min_ob,itv_ob

    x_cord=math.floor((delta-min_delta)/itv_delta)
    if x_cord>=14:
        x_cord=14

    y_cord=math.floor((obs-min_ob)/itv_ob)
    if y_cord>=14:
        y_cord=14

    return[x_cord,y_cord]



#import and get an action matrix from the learning results
dimension=15
df=pd.read_csv('/Users/mac/Desktop/Reinforcement Learning Bond/action_matrices/absobs+delta_action_matrix.csv')
action_matrix=np.zeros((15,15))
i=0
while i<15:
    j=0
    while j<15:
        action_matrix[i,j]=df[str(i)][j]
        j=j+1
    i=i+1


#set up accounts
cash=1000
share=0
value=0

#get an action list
actions_taken=list()
i=0
while i<len(training_set_rates):
    x_cord, y_cord=get_location(training_set_Deltas[i],training_set_obs[i])
    action=action_matrix[x_cord,y_cord]
    i=i+1
    actions_taken.append(action)

value_record=list()

#start to trade
if actions_taken[0]==1:
    share=cash/training_set_closes[0]
    cash=0
    value=cash+share*training_set_closes[0]
elif actions_taken[0]==-1:
    share=-cash/training_set_closes[0]
    cash=cash-share*training_set_closes[0]
    value=cash+share*training_set_closes[0]
value_record.append(value)

i=1
while i<len(training_set_closes):
    #1 no position
    if actions_taken[i-1]==0:
        if actions_taken[i]==1:#long position
            share=cash/training_set_closes[i]
            cash=0
            value=cash+share*training_set_closes[i]
        elif actions_taken[i]==-1:#short position
            share=-cash/training_set_closes[i]
            cash=cash-share*training_set_closes[i]
            value=cash+share*training_set_closes[i]
    #2  long position
    elif actions_taken[i-1]==1:
        if actions_taken[i]==1:#long position
            value=cash+share*training_set_closes[i]
        elif actions_taken[i]==0:#no position
            cash=share*training_set_closes[i]
            share=0
            value=cash+share*training_set_closes[i]
        else:#short position
            cash=share*training_set_closes[i]
            share=0
            share=-cash/training_set_closes[i]
            cash=cash-share*training_set_closes[i]
            value=cash+share*training_set_closes[i]
    #3 short position
    else:
        if actions_taken[i]==0:#no position
            cash=cash+share*training_set_closes[i]
            share=0
            value=cash+share*training_set_closes[i]
        elif actions_taken[i]==-1:#short position
            value=cash+share*training_set_closes[i]
        else:#long position
            cash=cash+share*training_set_closes[i]
            share=cash/training_set_closes[i]
            cash=0
            value=share*training_set_closes[i]+cash
    value_record.append(value)
    i=i+1

#process result datas to plot
i=0
initial_rate=training_set_closes[0]
while i<len(training_set_closes):
    training_set_closes[i]=training_set_closes[i]/initial_rate-1
    value_record[i]=value_record[i]/1000-1
    i=i+1

plt.plot(range(len(value_record)),value_record,range(len(value_record)),training_set_closes)
plt.show()
