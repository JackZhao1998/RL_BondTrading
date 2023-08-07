import numpy as np
import math
import pandas as pd
from pandas import DataFrame
import numpy as np
import statistics as stats
import matplotlib.pyplot as plt
import random

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

global rate_lists, closes_lists
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

'''
-------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

#set up training sets

'''
['2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019','2020','2021']
'''

global whole_set_years, total_training_years, validation_years, test_years, training_years, training_select_years

whole_set_years=['2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019','2020','2021']
training_years=['2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013']
training_select_years=['2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013']
validation_years=['2014','2015','2016','2017']
test_years=['2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013']

#define a data input function
global itv
itv=1/15

global dimension
dimension=15
print('The dimension is '+str(dimension))

#define action
global actions
actions=[0,1]#0=short 1=long

'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

def set_up_data(whole_set_years,training_years,validation_years):
    global whole_set_rates, whole_set_BB14Ws, whole_set_obs, whole_set_closes, training_set_rates, training_set_BB14Ws, training_set_obs, training_set_closes, validation_set_rates, validation_set_BB14Ws, validation_set_obs, validation_set_closes
    global training_set_x_cord_list, training_set_y_cord_list, validation_x_cord_list, validation_y_cord_list

    #set up lists
    whole_set_rates=list()
    whole_set_closes=list()

    training_set_rates=list()
    training_set_BB14Ws=list()
    training_set_obs=list()
    training_set_closes=list()

    validation_set_rates=list()
    validation_set_BB14Ws=list()
    validation_set_obs=list()
    validation_set_closes=list()

    #set up whole set rate&closes
    for year in whole_set_years:
        i=0
        while i<len(rate_lists[year]):
            whole_set_rates.append(rate_lists[year][i])
            whole_set_closes.append(closes_lists[year][i])
            i=i+1

    #set up training rate&closes
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

    #calculate required training sets
    training_set_widths=list()
    training_set_means=list()
    training_set_stds=list()
    i=70 #for 14 weeks as stated
    while i<len(training_set_rates):
        mean=stats.mean(training_set_rates[i-70:i])
        std=stats.stdev(training_set_rates[i-70:i])
        training_set_widths.append(std*6)
        training_set_stds.append(std)
        training_set_means.append(mean)
        i=i+1
    training_set_rates=training_set_rates[70:]
    i=0
    while i<len(training_set_means):
        ob=(training_set_rates[i]-training_set_means[i])/(3*training_set_stds[i])
        training_set_obs.append(ob)
        i=i+1
    #calculate BB14Ws
    training_set_BB14Ws=list()
    training_set_widths_means=list()
    training_set_widths_stds=list()
    i=70#past 14 weeks
    while i<len(training_set_widths):
        widths_mean=stats.mean(training_set_widths[i-70:i])
        training_set_widths_means.append(widths_mean)
        widths_std=stats.stdev(training_set_widths[i-70:i])
        training_set_widths_stds.append(widths_std)
        i=i+1
    i=0
    training_set_widths=training_set_widths[70:]
    while i<len(training_set_widths):
        BB14W=(training_set_widths[i]-training_set_widths_means[i])/(3*training_set_widths_stds[i])
        training_set_BB14Ws.append(BB14W)
        i=i+1

    training_set_obs=training_set_obs[70:]
    training_set_rates=training_set_rates[70:]
    training_set_closes=training_set_closes[140:]

    #calculate required validation sets
    validation_set_widths=list()
    validation_set_means=list()
    validation_set_stds=list()
    i=70 #for 14 weeks as stated
    while i<len(validation_set_rates):
        mean=stats.mean(validation_set_rates[i-70:i])
        std=stats.stdev(validation_set_rates[i-70:i])
        validation_set_widths.append(std*6)
        validation_set_stds.append(std)
        validation_set_means.append(mean)
        i=i+1
    validation_set_rates=validation_set_rates[70:]
    i=0
    while i<len(validation_set_means):
        ob=(validation_set_rates[i]-validation_set_means[i])/(3*validation_set_stds[i])
        validation_set_obs.append(ob)
        i=i+1
    #calculate BB14Ws
    validation_set_BB14Ws=list()
    validation_set_widths_means=list()
    validation_set_widths_stds=list()
    i=70#past 14 weeks
    while i<len(validation_set_widths):
        widths_mean=stats.mean(validation_set_widths[i-70:i])
        validation_set_widths_means.append(widths_mean)
        widths_std=stats.stdev(validation_set_widths[i-70:i])
        validation_set_widths_stds.append(widths_std)
        i=i+1
    i=0
    validation_set_widths=validation_set_widths[70:]
    while i<len(validation_set_widths):
        BB14W=(validation_set_widths[i]-validation_set_widths_means[i])/(3*validation_set_widths_stds[i])
        validation_set_BB14Ws.append(BB14W)
        i=i+1

    validation_set_obs=validation_set_obs[70:]
    validation_set_rates=validation_set_rates[70:]
    validation_set_closes=validation_set_closes[140:]

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
        y_cord=14
    elif obs<-1:
        y_cord=0
    else:
        y_cord=math.floor((abs(obs+1)/(2/dimension)))

    if BB14W>1:
        x_cord=14
    elif BB14W<-1:
        x_cord=0
    else:
        x_cord=math.floor((abs(1+BB14W)/(2/dimension)))
    return[x_cord,y_cord]

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

    reward=reward*pct_change*100

    return reward


#define an action selection function
def get_action(x_cord,y_cord,epsilon,action_matrix):
    if np.random.random()<epsilon:
        return np.argmax(action_matrix[x_cord,y_cord])
    else:
        return np.random.choice([0,1])


def get_error(action_taken,yield_difference):
    if action_taken==1:#take long position
        if yield_difference>0:#yield increases
            error= 1
        else:#yield decreases
            error= 0
    else:#take short position
        if yield_difference>0:#yield increases
            error= 0
        else:#yield decreases
            error= 1

    return error


def weight_roulette(x_cord,y_cord,as_matrix):

    action=random.choices([0,1],k=1)

    prob = as_matrix[x_cord, y_cord]
    neg_weight = prob[0]
    pos_weight = prob[1]
    sum=np.abs(neg_weight)+np.abs(pos_weight)
    neg_prob = np.abs(neg_weight)/sum
    pos_prob = np.abs(pos_weight)/sum
    #print(neg_weight,pos_weight)

    if neg_weight == 0:
        if pos_weight ==0:
            action = random.choices([0,1],k=1)
        elif pos_weight>0:
            action = 1
        else:
            action = 0

    elif neg_weight <0:
        if pos_weight==0:
            action = 1
        elif pos_weight >0:
            action = 1
        else:
            action = random.choices([0,1],weights=[pos_prob,neg_prob],k=1)[0]

    else:
        if pos_weight<0:
            action = 0
        elif pos_weight == 0:
            action = 0
        else:
            action = random.choices([0,1],weights=[neg_prob,pos_prob],k=1)[0]


    return(action)


'''
--------------------------------------------------------------------------------------
'''

def online_ps(epoch,gamma):
    global training_set_rates, training_set_BB14Ws, training_set_obs, training_set_closes, validation_set_rates, validation_set_BB14Ws, validation_set_obs, validation_set_closes
    global dimension, itv , long_position
    global training_years,validation_years,whole_set_years, training_select_years
    global x_cord_list, y_cord_list

    #initialize
    for episode in range(epoch):
        set_up_data(whole_set_years,training_years,validation_years)

        #training
        long_position=np.zeros((dimension,dimension))

        for i in range(len(training_set_obs)-1):
            x_cord,y_cord=x_cord_list[i],y_cord_list[i]
            action_taken=1
            yield_difference=training_set_rates[i+1]-training_set_rates[i]
            pct_change=abs(training_set_closes[i+1]/training_set_closes[i]-1)
            reward=calculate_reward(action_taken,yield_difference,pct_change)
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
    test_set_rates=[]
    test_set_closes=[]
    test_set_obs=[]
    test_set_BB14Ws=[]

    #set up test rate&closes
    for year in test_years:
        i=0
        while i<len(rate_lists[year]):
            test_set_rates.append(rate_lists[year][i])
            test_set_closes.append(closes_lists[year][i])
            i=i+1

    i=70 #for 14 weeks as stated
    while i<len(test_set_rates):
        mean=stats.mean(test_set_rates[i-70:i])
        std=stats.stdev(test_set_rates[i-70:i])
        test_set_widths.append(std*6)
        test_set_stds.append(std)
        test_set_means.append(mean)
        i=i+1
    test_set_rates=test_set_rates[70:]
    i=0
    while i<len(test_set_means):
        ob=(test_set_rates[i]-test_set_means[i])/(3*test_set_stds[i])
        test_set_obs.append(ob)
        i=i+1
    #calculate BB14Ws
    test_set_BB14Ws=list()
    test_set_widths_means=list()
    test_set_widths_stds=list()
    i=70#past 14 weeks
    while i<len(test_set_widths):
        widths_mean=stats.mean(test_set_widths[i-70:i])
        test_set_widths_means.append(widths_mean)
        widths_std=stats.stdev(test_set_widths[i-70:i])
        test_set_widths_stds.append(widths_std)
        i=i+1
    i=0
    test_set_widths=test_set_widths[70:]
    while i<len(test_set_widths):
        BB14W=(test_set_widths[i]-test_set_widths_means[i])/(3*test_set_widths_stds[i])
        test_set_BB14Ws.append(BB14W)
        i=i+1

    test_set_obs=test_set_obs[70:]
    test_set_rates=test_set_rates[70:]
    test_set_closes=test_set_closes[140:]


    #set up accounts
    cash=1000
    share=0
    value=1000

    #get an action list
    actions_taken=list()
    i=0
    while i<len(test_set_rates):
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

    plt.plot(range(len(value_record)),value_record,range(len(value_record)),test_set_closes)
    plt.legend(labels=['Test Results','Bond Closes'])
    plt.show()

    return


'''
-------------------------------------------------------------------------------------------------------------------------
'''


#set up training matrices

epoch=1
gamma=0.5
sum_matrix=np.zeros((dimension,dimension))

for year in training_select_years:
    if year == '2012':
        break
    else:
        training_years=[year,str(int(year)+1),str(int(year)+2)]
        test_years=training_years
        online_ps(epoch,gamma)
        print(long_position)

        #output the action matrix
        epsilon=0.1
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
        #print(action_matrix)
print(sum_matrix)

for i in range(dimension):
    for j in range(dimension):
        if sum_matrix[i,j] > 3:
            sum_matrix[i,j]=1
        elif sum_matrix[i,j] < -3:
            sum_matrix[i,j]=-1
        else:
            sum_matrix[i,j]=0
print(sum_matrix)
test_years=['2014','2015','2016','2017']
backtest(test_years,sum_matrix)
