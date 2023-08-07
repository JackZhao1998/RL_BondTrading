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
training_set_rates=list()
training_set_obs=list()
training_set_BB14W=list()
training_set_closes=list()
for year in training_years:
    i=0
    while i<len(rate_lists[year]):
        training_set_rates.append(rate_lists[year][i])
        training_set_closes.append(closes_lists[year][i])
        i=i+1

#calculate required training sets
widths=list()
means=list()
stds=list()
i=70 #for 14 weeks as stated
while i<len(training_set_rates):
    mean=stats.mean(training_set_rates[i-70:i])
    std=stats.stdev(training_set_rates[i-70:i])
    widths.append(std*6)
    stds.append(std)
    means.append(mean)
    i=i+1
training_set_rates=training_set_rates[70:]
i=0
while i<len(means):
    ob=(training_set_rates[i]-means[i])/(3*stds[i])
    training_set_obs.append(ob)
    i=i+1
#calculate BB14Ws
BB14Ws=list()
widths_means=list()
widths_stds=list()
i=70#past 14 weeks
while i<len(widths):
    widths_mean=stats.mean(widths[i-70:i])
    widths_means.append(widths_mean)
    widths_std=stats.stdev(widths[i-70:i])
    widths_stds.append(widths_std)
    i=i+1
i=0
widths=widths[70:]
while i<len(widths):
    BB14W=(widths[i]-widths_means[i])/(3*widths_stds[i])
    training_set_BB14W.append(BB14W)
    i=i+1
training_set_obs=training_set_obs[70:]
training_set_rates=training_set_rates[70:]
training_set_closes=training_set_closes[140:]

#define reward matrix
dimension=15
q_values=np.zeros((dimension,dimension,2))
q_TD=q_values
#define action
actions=[0,1]#0=short 1=long

#define reward function
def calculate_reward(action_taken,yield_difference,pct_change):#yield_difference=T+1 - T
    if action_taken==1:#take long position
        if yield_difference>0:#yield increases
            reward=-1
        else:#yield decreases
            reward=1
    else:#take short position
        if yield_difference>0:#yield increases
            reward=1
        else:#yield decreases
            reward=-1

    reward=reward*pct_change*10000
    return reward

#define a data input function
def get_location(BB14W,obs):
    if obs>1:
        y_cord=15
    elif obs<-1:
        y_cord=1
    else:
        y_cord=math.ceil((abs(obs+1)/(2/dimension)))

    if BB14W>1:
        x_cord=15
    elif BB14W<-1:
        x_cord=1
    else:
        x_cord=math.ceil((abs(1+BB14W)/(2/dimension)))
    return[x_cord-1,y_cord-1]

#define an action selection function
def get_action(x_cord,y_cord,epsilon):
    if np.random.random()<epsilon:
        return np.argmax(q_values[x_cord,y_cord])
    else:
        return np.random.choice([0,1])

#set up Q-learning Process
epsilon=1
discount_factor=0.9
learning_rate=0.9
reward_trace=list()
max_reward=0
for episode in range(100):
    i=0
    total_reward=0
    while i<len(training_set_obs)-1:
        x_cord,y_cord=get_location(training_set_BB14W[i],training_set_obs[i])
        action_taken=get_action(x_cord,y_cord,epsilon)
        yield_difference=training_set_rates[i+1]-training_set_rates[i]
        pct_change=abs(training_set_closes[i+1]/training_set_closes[i]-1)
        reward=calculate_reward(action_taken,yield_difference,pct_change)
        total_reward=total_reward+reward
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

print(len(training_set_obs))
plt.plot(reward_trace)
plt.show()

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
df.to_csv('/Users/mac/Desktop/Reinforcement Learning Bond/action_matrices/action_matrix3.csv')
plt.plot(training_set_BB14W,training_set_obs)
plt.show()
