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
closes=df['close_1']

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


global dimension
dimension=15
print('The dimension is '+str(dimension))

#set up training sets
'''
['2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019','2020','2021']
'''
global whole_set_years, total_training_years, validation_years, test_years
whole_set_years=['2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019','2020','2021']
total_training_years=['2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013']
total_training_selection_years=['2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012']
validation_years=['2014','2015','2016','2017']
test_years=['2018','2019','2020','2021']

#define a data input function
global itv
itv=1/15

#define action
global actions
actions=[0,1]#0=short 1=long



'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

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


def set_up_data(whole_set_years,training_years,validation_years,test_years):
    global whole_set_rates, whole_set_BBWs, whole_set_obs, whole_set_closes, training_set_rates, training_set_BBWs, training_set_obs, training_set_closes, validation_set_rates, validation_set_BBWs, validation_set_obs, validation_set_closes
    global rank_obs_list, rank_BBWs_list, rank_obs, rank_BBWs
    global x_cord_list, y_cord_list, validation_x_cord_list, validation_y_cord_list

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

    print(training_years)

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
    global rank_obs_list, rank_BBWs_list, rank_obs, rank_BBWs
    rank_obs=np.sort(whole_set_obs)
    rank_BBWs=np.sort(whole_set_BBWs)
    rank_obs_list=list()
    rank_BBWs_list=list()
    i=0
    while i<len(rank_obs):
        rank_obs_list.append(rank_obs[i])
        rank_BBWs_list.append(rank_BBWs[i])
        i=i+1


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

    return


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

    #reward=reward*pct_change

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
    global whole_set_rates, whole_set_BBWs, whole_set_obs, whole_set_closes, training_set_rates, training_set_BBWs, training_set_obs, training_set_closes, validation_set_rates, validation_set_BBWs, validation_set_obs, validation_set_closes
    global rank_obs_list, rank_BBWs_list, rank_obs, rank_BBWs
    global as_matrix, ct_matrix
    global dimension, itv

    #initialize as_matrix
    as_matrix=np.random.rand(dimension,dimension,2)

    training_error_rate=[]
    validation_error_rate=[]
    average_error=[]
    max_reward=0

    error_by_year={}

    #initialize
    for episode in range(epoch):
        print('now is the '+str(episode)+' episode')
        #set up training sets
        '''
        start_year=random.choices(total_training_selection_years,k=1)[0]
        training_years=[start_year,str(int(start_year)+1)]
        '''
        training_years=['2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013']
        start_year='2002'
        set_up_data(whole_set_years,training_years,validation_years,test_years)


        #set up ct_matrix
        ct_matrix=np.zeros((dimension,dimension,2))

        #training
        training_error=0
        for i in range(len(training_set_obs)-1):
            x_cord,y_cord=x_cord_list[i],y_cord_list[i]
            action_taken=weight_roulette(x_cord,y_cord,as_matrix)
            #print(action_taken)
            yield_difference=training_set_rates[i+1]-training_set_rates[i]
            pct_change=abs(training_set_closes[i+1]/training_set_closes[i]-1)
            reward=calculate_reward(action_taken,yield_difference,pct_change)
            #print('reward is '+str(reward))
            error=get_error(action_taken,yield_difference)
            training_error += error
            ct_matrix[x_cord,y_cord,action_taken] = ct_matrix[x_cord,y_cord,action_taken]+1
            for x in range(dimension):
                for y in range(dimension):
                    for z in range(2):
                        as_matrix[x,y,z]=as_matrix[x,y,z]+reward*ct_matrix[x,y,z]
                        ct_matrix[x,y,z]=gamma*ct_matrix[x,y,z]


        training_error = training_error/len(training_set_obs)
        training_error_rate.append(training_error)

        #validation
        validation_error=0
        for j in range(len(validation_set_obs)-1):
            x_cord,y_cord=validation_x_cord_list[j],validation_y_cord_list[j]
            action_taken=get_action(x_cord,y_cord,1,as_matrix)
            yield_difference=validation_set_rates[j+1]-validation_set_rates[j]
            pct_change=abs(validation_set_closes[j+1]/validation_set_closes[j]-1)
            reward=calculate_reward(action_taken,yield_difference,pct_change)
            error=get_error(action_taken,yield_difference)
            validation_error += error

        validation_error = validation_error/len(validation_set_obs)
        validation_error_rate.append(validation_error)

        print(training_error, validation_error)

        if start_year not in error_by_year:
            error_by_year[start_year]=[]
            error_by_year[start_year].append(training_error)
        else:
            error_by_year[start_year].append(training_error)


    print(training_error_rate)
    print(validation_error_rate)
    print(error_by_year)

    x=np.linspace(1,epoch,epoch)
    plot1=plt.figure(1)
    plt.plot(x,training_error_rate,x,validation_error_rate)
    plt.legend(['training error','validation_error','average_error'],loc=2,frameon=False)
    plt.show()

    return

#set up online_ps learning Process
#set up training matrices
global as_matrix, ct_matrix
as_matrix=np.zeros((dimension,dimension,2))
ct_matrix=np.zeros((dimension,dimension,2))
epoch=50
gamma=0.5
online_ps(epoch,gamma)





'''
-----------------------------------------------------------------------------------
'''

'''


def Q_learning(epoch,epsilon,discount_factor,learning_rate):
    global q_values, old_q_value, q_TD

    reward_trace=[]
    validation_reward_trace=[]
    training_error_rate=[]
    validation_error_rate=[]
    average_error=[]
    max_reward=0
    opt_epsilon=0
    min_error=1

    for episode in range(epoch):
        i=0
        total_reward=0
        training_error=0
        while i<len(training_set_obs)-1:
            x_cord,y_cord=x_cord_list[i],y_cord_list[i]
            action_taken=get_action(x_cord,y_cord,epsilon)
            yield_difference=training_set_rates[i+1]-training_set_rates[i]
            pct_change=abs(training_set_closes[i+1]/training_set_closes[i]-1)
            reward=calculate_reward(action_taken,yield_difference,pct_change)
            error=get_error(action_taken,yield_difference)
            training_error += error
            total_reward=total_reward+reward
            old_x_cord,old_y_cord=x_cord,y_cord
            old_q_value=q_values[old_x_cord,old_y_cord,action_taken]
            TD=reward+(discount_factor*np.max(q_values[x_cord,y_cord]))-old_q_value
            new_q_value=old_q_value+(learning_rate*TD)
            q_TD[old_x_cord,old_y_cord,action_taken]=new_q_value
            i=i+1
        training_error = training_error/len(training_set_obs)
        training_error_rate.append(training_error)

        old_q_values=q_values
        q_values=q_TD

        j=0
        validation_total_reward=0
        validation_error=0
        while j<len(validation_set_obs)-1:
            x_cord,y_cord=validation_x_cord_list[j],validation_y_cord_list[j]
            action_taken=get_action(x_cord,y_cord,epsilon)
            yield_difference=validation_set_rates[j+1]-validation_set_rates[j]
            pct_change=abs(validation_set_closes[j+1]/validation_set_closes[j]-1)
            reward=calculate_reward(action_taken,yield_difference,pct_change)
            validation_total_reward=validation_total_reward+reward
            error=get_error(action_taken,yield_difference)
            validation_error += error
            j=j+1
        validation_error = validation_error/len(validation_set_obs)
        validation_error_rate.append(validation_error)
        mean_error=(training_error+validation_error)/2
        average_error.append(mean_error)

        if validation_total_reward>=max_reward:
            max_reward=validation_total_reward
        else:
            q_values=old_q_values

        validation_reward_trace.append(validation_total_reward*1000)
        reward_trace.append(total_reward)

    if min_error >= mean_error:
        min_error=mean_error
        opt_epsilon=epsilon

    plot1=plt.figure(1)
    plt.plot(range(epoch),training_error_rate,range(epoch),validation_error_rate,range(epoch),average_error)
    plt.legend(['training error','validation_error','average_error'],loc=2,frameon=False)
    plt.show()
    return([min_error,opt_epsilon])

'''


'''
#set up Q-learning Process
#set up Q_learning training matrices
global q_values, old_q_values, q_TD
q_values=np.zeros((dimension,dimension,2))
old_q_values=q_values
q_TD=q_values
ct_matrix=np.zeros((dimension,dimension,1))

#set up Q-learning peremeter
peremeters=np.linspace(990,1000,11)/1000
discount_factor=0.9
learning_rate=0.9
epoch=10000

epsilon_result=[]
error=[]
for epsilon in peremeters:
    result=rl(epoch,epsilon,discount_factor,learning_rate)
    epsilon_result.append(result[1])
    error.append(result[0])

output_result={}
output_result['epsilon_result']=epsilon_result
output_result['error']=error
output_result=DataFrame(output_result)
output_result.to_csv('/Users/mac/Desktop/Reinforcement Learning Bond/peremeter/epsilon3.csv')
'''

#output the action matrix
action_matrix=np.zeros((dimension,dimension))
i=0
while i<dimension:
    j=0
    while j<dimension:
        if as_matrix[i,j,0]>as_matrix[i,j,1]:
            action_matrix[i,j]=-1
        elif as_matrix[i,j,0]<as_matrix[i,j,1]:
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
