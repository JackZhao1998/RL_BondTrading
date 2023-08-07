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
whole_set_years=['2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019','2020','2021']
training_years=['2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015']
validation_years=['2016','2017','2018']


#set up lists
whole_set_rates=list()
whole_set_BBWs=list()
whole_set_obs=list()
whole_set_closes=list()

training_set_rates=list()
training_set_BBWs=list()
training_set_obs=list()
training_set_closes=list()


validation_set_rates=list()
validation_set_BBWs=list()
validation_set_obs=list()
validation_set_closes=list()

#set up whole set rate&closes
for year in whole_set_years:
    i=0
    while i<len(rate_lists[year]):
        whole_set_rates.append(rate_lists[year][i])
        whole_set_closes.append(closes_lists[year][i])
        i=i+1

#set up rate&closes
for year in training_years:
    i=0
    while i<len(rate_lists[year]):
        training_set_rates.append(rate_lists[year][i])
        training_set_closes.append(closes_lists[year][i])
        i=i+1

#set up validation rate&closes
for year in validation_years:
    i=0
    while i<len(rate_lists[year]):
        validation_set_rates.append(rate_lists[year][i])
        validation_set_closes.append(closes_lists[year][i])
        i=i+1


#calculate required whole sets
whole_set_BBWs=list()
i=20 #for 20 trading days
while i<len(whole_set_rates):
    std=stats.stdev(whole_set_rates[i-20:i])
    whole_set_BBWs.append(std*2)
    i=i+1

#calculate required training sets
training_set_BBWs=list()
i=20 #for 20 trading days
while i<len(training_set_rates):
    std=stats.stdev(training_set_rates[i-20:i])
    training_set_BBWs.append(std*2)
    i=i+1

#calculate required validation sets
validation_set_BBWs=list()
i=20 #for 20 trading days
while i<len(validation_set_rates):
    std=stats.stdev(validation_set_rates[i-20:i])
    validation_set_BBWs.append(std*2)
    i=i+1

#modify rates&closes
whole_set_rates=whole_set_rates[20:]
whole_set_closes=whole_set_closes[20:]
whole_set_obs=whole_set_rates

training_set_rates=training_set_rates[20:]
training_set_closes=training_set_closes[20:]
training_set_obs=training_set_rates

validation_set_rates=validation_set_rates[20:]
validation_set_closes=validation_set_closes[20:]
validation_set_obs=validation_set_rates

#rank the data lists
global rank_obs_list, rank_BBWs_list
rank_obs=np.sort(whole_set_obs)
rank_BBWs=np.sort(whole_set_BBWs)
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

#get position lists
x_cord_list=list()
y_cord_list=list()
validation_x_cord_list=list()
validation_y_cord_list=list()

i=0
while i<len(training_set_obs):
    x_cord,y_cord=get_location(training_set_BBWs[i],training_set_obs[i])
    x_cord_list.append(x_cord)
    y_cord_list.append(y_cord)
    i=i+1

i=0
while i<len(validation_set_obs):
    validation_x_cord,validation_y_cord=get_location(validation_set_BBWs[i],validation_set_obs[i])
    validation_x_cord_list.append(validation_x_cord)
    validation_y_cord_list.append(validation_y_cord)
    i=i+1


#define action
actions=[0,1]#0=short 1=long

#define reward function
def calculate_reward(action_taken,yield_difference,pct_change):#yield_difference=T+1 - T
    if action_taken==1:#take long position
        if yield_difference>0:#yield increases
            reward= -1
        else:#yield decreases
            reward= 1
    else:#take short position
        if yield_difference>0:#yield increases
            reward= 1
        else:#yield decreases
            reward= -1

    reward=reward*pct_change

    return reward

#define an action selection function
def get_action(x_cord,y_cord,epsilon):
    if np.random.random()<epsilon:
        return np.argmax(q_values[x_cord,y_cord])
    else:
        return np.random.choice([0,1])

#set up Q-learning Process
epsilon=1
discount_factor=1
learning_rate=0.9
reward_trace=list()
validation_reward_trace=list()
max_reward=0

for episode in range(1000000):
    i=0
    total_reward=0
    count=14
    while i<len(training_set_obs)-1:
        if i<len(training_set_obs)-15:
            if count==14:
                lx_cord,ly_cord=x_cord_list[i],y_cord_list[i]
                action_taken=get_action(lx_cord,ly_cord,epsilon)
                yield_difference=training_set_rates[i+15]-training_set_rates[i]
                pct_change=abs(training_set_closes[i+1]/training_set_closes[i]-1)
                lt_reward=calculate_reward(action_taken,yield_difference,pct_change)
                count=count-1
            else:
                lt_reward=0
                count=count-1
            if count==0:
                count=14
        x_cord,y_cord=x_cord_list[i],y_cord_list[i]
        action_taken=get_action(x_cord,y_cord,epsilon)
        yield_difference=training_set_rates[i+1]-training_set_rates[i]
        pct_change=abs(training_set_closes[i+1]/training_set_closes[i]-1)
        reward=calculate_reward(action_taken,yield_difference,pct_change)
        total_reward=total_reward+reward+lt_reward*100
        old_x_cord,old_y_cord=x_cord,y_cord
        old_q_value=q_values[old_x_cord,old_y_cord,action_taken]
        TD=reward+(discount_factor*np.max(q_values[x_cord,y_cord]))-old_q_value
        new_q_value=old_q_value+(learning_rate*TD)
        q_TD[old_x_cord,old_y_cord,action_taken]=new_q_value
        i=i+1
    if total_reward>max_reward:
        q_values=q_TD
        max_reward=total_reward
    reward_trace.append(max_reward)

    i=0
    validation_total_reward=0
    while i<len(validation_set_obs)-1:
        x_cord,y_cord=validation_x_cord_list[i],validation_y_cord_list[i]
        action_taken=get_action(x_cord,y_cord,epsilon)
        yield_difference=validation_set_rates[i+1]-validation_set_rates[i]
        pct_change=abs(validation_set_closes[i+1]/validation_set_closes[i]-1)
        reward=calculate_reward(action_taken,yield_difference,pct_change)
        validation_total_reward=validation_total_reward+reward
        i=i+1
    validation_reward_trace.append(validation_total_reward)

plot1=plt.figure(1)
plt.plot(reward_trace)
plt.show()
plt.plot(validation_reward_trace)
plt.show()


#output the action matrix
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

actions=dict()

i=0
while i<dimension:
    actions[str(i)]=action_matrix[i]
    i=i+1
df=pd.DataFrame(actions,columns=['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14'])
df.to_csv('/Users/mac/Desktop/Reinforcement Learning Bond/action_matrices/absobs+BBW_action_matrix.csv')
