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


#set up training sets
'''
['2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019','2020','2021']
'''
training_years=['2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019','2020','2021']
#set up lists
training_set_rates=list()
training_set_BBWs=list()
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
training_set_BBWs=list()
i=20 #for 20 trading days
while i<len(training_set_rates):
    std=stats.stdev(training_set_rates[i-20:i])
    training_set_BBWs.append(std*2)
    i=i+1

#modify rates&closes
training_set_rates=training_set_rates[20:]
training_set_closes=training_set_closes[20:]
training_set_obs=training_set_rates

#rank the data lists
global rank_obs_list, rank_BBWs_list
rank_obs=np.sort(training_set_obs)
rank_BBWs=np.sort(training_set_BBWs)
rank_obs_list=list()
rank_BBWs_list=list()
i=0
while i<len(rank_obs):
    rank_obs_list.append(rank_obs[i])
    rank_BBWs_list.append(rank_BBWs[i])
    i=i+1

#define a data input function
itv=1/15

def get_location(BBW,obs):
    global rank_obs_list, rank_BBWs_list
    BBW_pct=rank_BBWs_list.index(BBW)
    BBW_pct=(BBW_pct+1)/len(rank_BBWs)
    x_cord=math.floor((BBW_pct)/itv)
    if x_cord>=14:
        x_cord=14

    obs_pct=rank_obs_list.index(obs)
    obs_pct=(obs_pct+1)/len(rank_obs)
    y_cord=math.floor((obs_pct)/itv)
    if y_cord>=14:
        y_cord=14

    return[x_cord,y_cord]


#set up test sets
'''
'2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019','2020','2021'
'''
test_years=['2014','2015','2016','2017','2018','2019','2020','2021']
test_set_rates=list()
test_set_BBWs=list()
test_set_obs=list()
test_set_closes=list()


#set up test rate&closes
for year in test_years:
    i=0
    while i<len(rate_lists[year]):
        test_set_rates.append(rate_lists[year][i])
        test_set_closes.append(closes_lists[year][i])
        i=i+1

test_BBWs=list()

i=20 #for 20 trading days
while i<len(test_set_rates):
    std=stats.stdev(test_set_rates[i-20:i])
    test_BBWs.append(std*2)
    i=i+1

test_set_rates=test_set_rates[20:]
test_set_closes=test_set_closes[20:]
test_set_obs=test_set_rates
test_set_BBWs=test_BBWs

#import and get an action matrix from the learning results
dimension=15
df=pd.read_csv('/Users/mac/Desktop/Reinforcement Learning Bond/action_matrices/absobs+BBW_action_matrix.csv')
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
value=1000

#get an action list
actions_taken=list()
i=0
while i<len(test_set_rates):
    x_cord, y_cord=get_location(test_set_BBWs[i],test_set_obs[i])
    action=action_matrix[x_cord,y_cord]
    i=i+1
    actions_taken.append(action)

value_record=list()

#start to trade
if actions_taken[0]==1:
    share=cash/test_set_closes[0]
    cash=0
    value=cash+share*test_set_closes[0]
elif actions_taken[0]==-1:
    share=-cash/test_set_closes[0]
    cash=cash-share*test_set_closes[0]
    value=cash+share*test_set_closes[0]


value_record.append(value)

i=1
while i<len(test_set_closes):
    #1 no position
    if actions_taken[i-1]==0:
        if actions_taken[i]==1:#long position
            share=cash/test_set_closes[i]
            cash=0
            value=cash+share*test_set_closes[i]
        elif actions_taken[i]==-1:#short position
            share=-cash/test_set_closes[i]
            cash=cash-share*test_set_closes[i]
            value=cash+share*test_set_closes[i]
    #2  long position
    elif actions_taken[i-1]==1:
        if actions_taken[i]==1:#long position
            value=cash+share*test_set_closes[i]
        elif actions_taken[i]==0:#no position
            cash=share*test_set_closes[i]
            share=0
            value=cash+share*test_set_closes[i]
        else:#short position
            cash=share*test_set_closes[i]
            share=0
            share=-cash/test_set_closes[i]
            cash=cash-share*test_set_closes[i]
            value=cash+share*test_set_closes[i]
    #3 short position
    else:
        if actions_taken[i]==0:#no position
            cash=cash+share*test_set_closes[i]
            share=0
            value=cash+share*test_set_closes[i]
        elif actions_taken[i]==-1:#short position
            value=cash+share*test_set_closes[i]
        else:#long position
            cash=cash+share*test_set_closes[i]
            share=cash/test_set_closes[i]
            cash=0
            value=share*test_set_closes[i]+cash
    value_record.append(value)
    i=i+1

#process result datas to plot
i=0
initial_rate=test_set_closes[0]

while i<len(test_set_closes):
    test_set_closes[i]=test_set_closes[i]/initial_rate-1
    value_record[i]=value_record[i]/1000-1
    i=i+1

plt.plot(range(len(value_record)),value_record,range(len(value_record)),test_set_closes)
plt.legend(labels=['Test Results','Bond Closes'])
plt.show()
