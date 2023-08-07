import numpy as np
import math
import pandas as pd
from pandas import DataFrame
import numpy as np
import statistics as stats
import matplotlib.pyplot as plt
import random

#import data
def initiate_data(file_name):
    global dates, closes, years_list
    df=pd.read_csv(file_name)
    df.fillna(method='ffill')
    df=df.drop(columns='Unnamed: 0')
    dates=df['trade_date']
    closes=df['close']

    #put data into structured dictionaries and an index list of years
    years_list=list()
    closes_datas=dict()
    for date in dates:
        date=str(date)
        if date[0:4] not in closes_datas:
            closes_datas[date[0:4]]=dict()
            years_list.append(date[0:4])
    i=0;
    while i < len(dates):
        match=str(dates[i])[0:4]
        closes_datas[match][dates[i]]=float(closes[i])
        i=i+1

    global  closes_lists
    closes_lists=dict()
    for year in years_list:
        close=list()
        for date in closes_datas[year]:
            close.append(closes_datas[year][date])
        closes_lists[year]=close
    years_list.reverse()
    print(years_list)
    # years_list:a list of all years
    # closes_datas:a dictionary of dictionaries of closes and dates
    # closes_lists: a dictionary of lists of closes
    print("Data initialized")
    return

'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

def set_up_data(training_years,validation_years):
    global training_set_BB14Ws, training_set_obs, training_set_closes, validation_set_BB14Ws, validation_set_obs, validation_set_closes
    global training_set_x_cord_list, training_set_y_cord_list, validation_x_cord_list, validation_y_cord_list

    #set up lists
    training_set_BB14Ws=list()
    training_set_obs=list()
    training_set_closes=list()

    validation_set_BB14Ws=list()
    validation_set_obs=list()
    validation_set_closes=list()

    #set up training rate&closes
    for year in training_years:
        i=0
        while i<len(closes_lists[year]):
            training_set_closes.append(closes_lists[year][i])
            i=i+1

    #set up validation rate&closes
    for year in validation_years:
        i=0
        while i<len(closes_lists[year]):
            validation_set_closes.append(closes_lists[year][i])
            i=i+1

    #calculate required training sets
    training_set_widths=list()
    training_set_means=list()
    training_set_stds=list()
    i=time_delta #for 14 weeks as stated
    while i<len(training_set_closes):
        mean=stats.mean(training_set_closes[i-time_delta:i])
        std=stats.stdev(training_set_closes[i-time_delta:i])
        training_set_widths.append(std*6)
        training_set_stds.append(std)
        training_set_means.append(mean)
        i=i+1
    training_set_closes=training_set_closes[time_delta:]
    i=0
    while i<len(training_set_means):
        ob=(training_set_closes[i]-training_set_means[i])/(3*training_set_stds[i])
        training_set_obs.append(ob)
        i=i+1
    #calculate BB14Ws
    training_set_BB14Ws=list()
    training_set_widths_means=list()
    training_set_widths_stds=list()
    i=time_delta#past 14 weeks
    while i<len(training_set_widths):
        widths_mean=stats.mean(training_set_widths[i-time_delta:i])
        training_set_widths_means.append(widths_mean)
        widths_std=stats.stdev(training_set_widths[i-time_delta:i])
        training_set_widths_stds.append(widths_std)
        i=i+1
    i=0
    training_set_widths=training_set_widths[time_delta:]
    while i<len(training_set_widths):
        BB14W=(training_set_widths[i]-training_set_widths_means[i])/(3*training_set_widths_stds[i])
        training_set_BB14Ws.append(BB14W)
        i=i+1

    training_set_obs=training_set_obs[time_delta:]
    training_set_closes=training_set_closes[time_delta:]

    #calculate required validation sets
    validation_set_widths=list()
    validation_set_means=list()
    validation_set_stds=list()
    i=time_delta #for 14 weeks as stated
    while i<len(validation_set_closes):
        mean=stats.mean(validation_set_closes[i-time_delta:i])
        std=stats.stdev(validation_set_closes[i-time_delta:i])
        validation_set_widths.append(std*6)
        validation_set_stds.append(std)
        validation_set_means.append(mean)
        i=i+1
    validation_set_closes=validation_set_closes[time_delta:]
    i=0
    while i<len(validation_set_means):
        ob=(validation_set_closes[i]-validation_set_means[i])/(3*validation_set_stds[i])
        validation_set_obs.append(ob)
        i=i+1
    #calculate BB14Ws
    validation_set_BB14Ws=list()
    validation_set_widths_means=list()
    validation_set_widths_stds=list()
    i=time_delta#past 14 weeks
    while i<len(validation_set_widths):
        widths_mean=stats.mean(validation_set_widths[i-time_delta:i])
        validation_set_widths_means.append(widths_mean)
        widths_std=stats.stdev(validation_set_widths[i-time_delta:i])
        validation_set_widths_stds.append(widths_std)
        i=i+1
    i=0
    validation_set_widths=validation_set_widths[time_delta:]
    while i<len(validation_set_widths):
        BB14W=(validation_set_widths[i]-validation_set_widths_means[i])/(3*validation_set_widths_stds[i])
        validation_set_BB14Ws.append(BB14W)
        i=i+1

    validation_set_obs=validation_set_obs[time_delta:]
    validation_set_closes=validation_set_closes[time_delta:]

    #get position lists
    global x_cord_list, y_cord_list
    x_cord_list=list()
    y_cord_list=list()

    validation_x_cord_list=list()
    validation_y_cord_list=list()

    i=0
    while i<len(training_set_obs):
        x_cord,y_cord=get_location(training_set_BB14Ws[i],training_set_obs[i])
        x_cord_list.append(x_cord)
        y_cord_list.append(y_cord)
        i=i+1

    i=0
    while i<len(validation_set_obs):
        validation_x_cord,validation_y_cord=get_location(validation_set_BB14Ws[i],validation_set_obs[i])
        validation_x_cord_list.append(validation_x_cord)
        validation_y_cord_list.append(validation_y_cord)
        i=i+1

    return

'''
---------------------------------------------------------------------------------------------------------------
'''

def get_location(BB14W,obs):
    if obs>1:
        y_cord=dimension-1
    elif obs<-1:
        y_cord=0
    else:
        y_cord=math.floor((abs(obs+1)/(2/dimension)))

    if BB14W>1:
        x_cord=dimension-1
    elif BB14W<-1:
        x_cord=0
    else:
        x_cord=math.floor((abs(1+BB14W)/(2/dimension)))
    return[x_cord,y_cord]

#define reward function
def calculate_reward(action_taken,difference,pct_change):#difference=T+1 - T
    if action_taken==1:#take long position
        if difference>0:#yield increases
            reward= 1
        else:#yield decreases
            reward= -1
    else:#take short position
        if difference>0:# increases
            reward= -1
        else:# decreases
            reward= 1

    reward=reward*pct_change*100
    if np.abs(pct_change)>0.05:
        reward += 10

    return reward


'''
--------------------------------------------------------------------------------------
'''
def online_ps():
    global training_set_rates, training_set_BB14Ws, training_set_obs, training_set_closes, validation_set_rates, validation_set_BB14Ws, validation_set_obs, validation_set_closes
    global dimension, itv , long_position, file_name
    global training_years,validation_years,whole_set_years, training_select_years
    global x_cord_list, y_cord_list

    #initialize
    set_up_data(training_years,validation_years)

    #training
    long_position=np.zeros((dimension,dimension))

    for i in range(len(training_set_obs)-1):
        x_cord,y_cord=x_cord_list[i],y_cord_list[i]
        action_taken=1
        difference=training_set_closes[i+1]-training_set_closes[i]
        pct_change=abs(training_set_closes[i+1]/training_set_closes[i]-1)
        reward=calculate_reward(action_taken,difference,pct_change)
        long_position[x_cord,y_cord] += reward

    return

'''
------------------------------------------------------------------------------------------------------------------------
'''

def backtest(test_years,action_matrix):
    global whole_set_rates, whole_set_BB14Ws, whole_set_obs, whole_set_closes, training_set_rates, training_set_BB14Ws, training_set_obs, training_set_closes, validation_set_rates, validation_set_BB14Ws, validation_set_obs, validation_set_closes
    global dimension, itv, long_position

    #set up test rate&closes
    test_set_widths=[]
    test_set_means=[]
    test_set_stds=[]
    test_set_closes=[]
    test_set_obs=[]
    test_set_BB14Ws=[]

    #set up test rate&closes
    for year in test_years:
        i=0
        while i<len(closes_lists[year]):
            test_set_closes.append(closes_lists[year][i])
            i=i+1

    i=time_delta #for 14 weeks as stated
    while i<len(test_set_closes):
        mean=stats.mean(test_set_closes[i-time_delta:i])
        std=stats.stdev(test_set_closes[i-time_delta:i])
        test_set_widths.append(std*6)
        test_set_stds.append(std)
        test_set_means.append(mean)
        i=i+1
    test_set_closes=test_set_closes[time_delta:]
    i=0
    while i<len(test_set_means):
        ob=(test_set_closes[i]-test_set_means[i])/(3*test_set_stds[i])
        test_set_obs.append(ob)
        i=i+1
    #calculate BB14Ws
    test_set_BB14Ws=list()
    test_set_widths_means=list()
    test_set_widths_stds=list()
    i=time_delta#past 14 weeks
    while i<len(test_set_widths):
        widths_mean=stats.mean(test_set_widths[i-time_delta:i])
        test_set_widths_means.append(widths_mean)
        widths_std=stats.stdev(test_set_widths[i-time_delta:i])
        test_set_widths_stds.append(widths_std)
        i=i+1
    i=0
    test_set_widths=test_set_widths[time_delta:]
    while i<len(test_set_widths):
        BB14W=(test_set_widths[i]-test_set_widths_means[i])/(3*test_set_widths_stds[i])
        test_set_BB14Ws.append(BB14W)
        i=i+1

    test_set_obs=test_set_obs[time_delta:]
    test_set_closes=test_set_closes[time_delta:]


    #set up accounts
    cash=1000
    share=0
    value=1000

    #get an action list
    actions_taken=list()
    i=0
    while i<len(test_set_closes):
        x_cord, y_cord=get_location(test_set_BB14Ws[i],test_set_obs[i])
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
    #print(test_set_closes)

    plt.plot(range(len(value_record)),value_record,range(len(value_record)),test_set_closes)
    plt.legend(labels=['Test Results','stock Closes'])
    plt.show()

    return


'''
-------------------------------------------------------------------------------------------------------------------------
'''

#set up training sets

'''
['2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019','2020','2021']
'''

global whole_set_years, total_training_years, validation_years, test_years, training_years, training_select_years
global dimension, itv
dimension=10
itv=1/dimension
print('The dimension is '+str(dimension))

global time_delta
time_delta=20


#define action
global actions
actions=[0,1]#0=short 1=long

#import data
file_name='/Users/jackritian/Desktop/barra Model/data base/stock data/000001.SZ_daily.csv'

initiate_data(file_name)

#set up training matrix
sum_matrix=np.zeros((dimension,dimension))

#set up training years
global years_list
whole_set_years=years_list
training_select_years=years_list[0:math.ceil(len(years_list)*0.6)]
training_years=training_select_years
validation_years=years_list[math.ceil(len(years_list)*0.6):math.ceil(len(years_list)*0.8)]
test_years=years_list[math.ceil(len(years_list)*0.8):]

print(training_select_years)
print(validation_years)
print(test_years)

#set up loop
position_count=0
for year in training_select_years:
    if position_count == len(training_select_years)-2:
        break
    else:
        position_count += 1
        training_years=training_select_years[position_count:position_count+2]
        test_years=training_years
        online_ps()
        print("Rewards take the long_postion are")
        print(long_position)

        #output the action matrix
        epsilon=2
        action_matrix=np.zeros((dimension,dimension))

        for i in range(dimension):
            for j in range(dimension):
                if long_position[i,j] > epsilon:
                    action_matrix[i,j]=1
                elif long_position[i,j] < -epsilon:
                    action_matrix[i,j]=-1
                else:
                    action_matrix[i,j]=0
        sum_matrix = sum_matrix + action_matrix
        test_years=training_years
        #backtest(test_years,action_matrix)
        print(action_matrix)

action_matrix=np.zeros((dimension,dimension))

print(sum_matrix)

for i in range(dimension):
    for j in range(dimension):
        if sum_matrix[i,j] > 0:
            action_matrix[i,j]=1
        elif sum_matrix[i,j] < 0:
            action_matrix[i,j]=-1
        else:
            action_matrix[i,j]=0

print('The action Matrix is')
print(action_matrix)
backtest(training_select_years,action_matrix)
backtest(validation_years,action_matrix)
test_years=years_list[math.ceil(len(years_list)*0.8):]
backtest(test_years,action_matrix)

H=np.array(action_matrix)

fig = plt.figure(figsize=(6, 3.2))

ax = fig.add_subplot(111)
ax.set_title('colorMap')
plt.imshow(H)
ax.set_aspect('equal')
cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
cax.get_xaxis().set_visible(False)
cax.get_yaxis().set_visible(False)
cax.patch.set_alpha(0)
cax.set_frame_on(False)
plt.colorbar(orientation='vertical')
plt.show()
