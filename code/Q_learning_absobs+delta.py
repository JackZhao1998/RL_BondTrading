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
training_years=['2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017']

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



#define action
actions=[0,1]#0=short 1=long

#define reward function
def calculate_reward(action_taken,yield_difference):#yield_difference=T+1 - T
    if action_taken==1:#take long position
        if yield_difference>0:#yield increases
            return -1
        else:#yield decreases
            return 1
    else:#take short position
        if yield_difference>0:#yield increases
            return 1
        else:#yield decreases
            return -1

#define an action selection function
def get_action(x_cord,y_cord,epsilon):
    if np.random.random()<epsilon:
        return np.argmax(q_values[x_cord,y_cord])
    else:
        return np.random.choice([0,1])

#set up Q-learning Process
epsilon=0.9999
discount_factor=0.9
learning_rate=0.9
reward_trace=list()
max_reward=0

for episode in range(100):
    i=0
    total_reward=0
    while i<len(training_set_obs)-1:
        x_cord,y_cord=get_location(training_set_Deltas[i],training_set_obs[i])
        action_taken=get_action(x_cord,y_cord,epsilon)
        yield_difference=training_set_rates[i+1]-training_set_rates[i]
        reward=calculate_reward(action_taken,yield_difference)
        total_reward=total_reward+reward
        old_x_cord,old_y_cord=x_cord,y_cord
        old_q_value=q_values[old_x_cord,old_y_cord,action_taken]
        x_cord,y_cord=get_location(training_set_Deltas[i+1],training_set_obs[i+1])
        TD=reward+discount_factor*np.max(q_values[x_cord,y_cord])-old_q_value
        new_q_value=old_q_value+(learning_rate*TD)
        q_TD[old_x_cord,old_y_cord,action_taken]=new_q_value
        i=i+1
    if total_reward>max_reward:
        max_reward=total_reward
        q_values=q_TD

    reward_trace.append(max_reward)

print(len(training_set_obs))
plt.plot(reward_trace)
plt.show()

print(q_values)
action_matrix=np.zeros((dimension,dimension))
i=0
while i<dimension:
    j=0
    while j<dimension:
        if q_values[i,j,0]>q_values[i,j,1]:
            action_matrix[i,j]=-1
        elif q_values[i,j,0]<q_values[i,j,1]:
            action_matrix[i,j]=1
        else:
            action_matrix[i,j]=0
        j=j+1
    i=i+1
print(action_matrix)
actions=dict()
i=0
while i<dimension:
    actions[str(i)]=action_matrix[i]
    i=i+1
df=pd.DataFrame(actions,columns=['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14'])
df.to_csv('/Users/mac/Desktop/Reinforcement Learning Bond/action_matrices/absobs+delta_action_matrix1.csv')
