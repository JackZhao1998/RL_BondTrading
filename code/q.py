import numpy as np
import random
import pandas as pd
from pandas import DataFrame

df = pd.read_csv('/Users/mac/Desktop/Barra Model/data base/stock_list.csv')#store the stock list
print(df)
stock_list=df['ts_code']
for stock in stock_list:
    print(stock)
    break
