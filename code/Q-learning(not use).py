import numpy as np
import math
import pandas as pd
import numpy as np
import statistics as stats
import matplotlib.pyplot as plt


#import data
df=pd.read_csv('/Users/mac/Desktop/Reinforcement Learning Bond/bond_backtest.csv')
df.fillna(method='ffill')
dates=df['date']
rates=df['yield']

#put data into structured dictionaries and an index list of years
years=dict()
years_list=list()
for date in dates:
    date=str(date)
    if date[0:4] not in years:
        years[date[0:4]]=dict()
        years_list.append(date[0:4])
i=0;
while i < len(dates):
    match=dates[i][0:4]
    years[match][dates[i]]=float(rates[i])
    i=i+1

#define the funciton relative_observation
def relative_observation(year):
    #for each year's data,calculate the observation datas required
    values=list()
    for value in years[year]:
        values.append(years[year][value])

#create lists of observation values
    global obs
    widths=list()
    means=list()
    stds=list()
    obs=list()
    i=70 #for 14 weeks as stated
    while i<len(values):
        mean=stats.mean(values[i-70:i])
        std=stats.stdev(values[i-70:i])
        widths.append(std*6)
        stds.append(std)
        means.append(mean)
        i=i+1
    values=values[70:]
#calculate observation values
    i=0
    while i<len(values):
        ob=(values[i]-means[i])/(3*stds[i])
        obs.append(ob)
        i=i+1

#calculate list of BB14Ws values
    global BB14Ws
    BB14Ws=list()
    widths_means=list()
    widths_stds=list()
#calclute BB14Ws
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
        BB14Ws.append(BB14W)
        i=i+1
#modify obs with normalized BB14Ws
    obs=obs[70:]

#calculate and store values in dictionaries
global obs_list
global BB14Ws_list
obs_list=dict()
BB14Ws_list=dict()
for year in years_list:
    relative_observation(year)
    obs_list[year]=obs
    BB14Ws_list[year]=BB14Ws

#set up training set
training_years=['2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019','2020']
training_set_obs=list()
training_set_BB14W=list()
for year in training_years:
    for x in obs_list[year]:
        training_set_obs.append(x)
    for y in BB14Ws_list[year]:
        training_set_BB14W.append(y)

#define reward matrix
dimension=15
q_values=np.zeros((dimension,dimension,2))

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
    return[x_cord,y_cord]

#define an action selection function
def get_action(x_cord,y_cord,epsilon):
    if np.random.random()<epsilon:
        return np.argmax(q_values[x_cord-1,y_cord-1])
    else:
        return np.random.choice([-1,1])

#set up Q-learning Process
epsilon=0.9
discount_factor=0.9
learning_rate=0.9

for episode in range(1000000):
    i=0
    while i<len(training_set_obs)-1:
        x_cord,y_cord=get_location(training_set_BB14W[i],training_set_obs[i])
        action_taken=get_action(x_cord,y_cord,epsilon)
        yield_difference=training_set_obs[i+1]-training_set_obs[i]
        reward=calculate_reward(action_taken,yield_difference)
        old_x_cord,old_y_cord=x_cord,y_cord
        old_q_value=q_values[old_x_cord-1,old_y_cord-1,action_taken]
        TD=reward+(discount_factor*np.max(q_values[x_cord-1,y_cord-1]))-old_q_value
        new_q_value=old_q_value+(learning_rate*TD)
        q_values[old_x_cord-1,old_y_cord-1,action_taken]=new_q_value
        i=i+1

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
