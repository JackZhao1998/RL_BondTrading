from selenium import webdriver             #导入selenium驱动程序
from selenium.webdriver.common.keys import Keys
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import time

url='https://sis.jhu.edu/sswf/SSS/SearchForClasses/SSS_SearchForClasses.aspx?MyIndex=91497'
account='yzhao136@jh.edu'
password='Zy@031298'

#input drive path
driver_path='/Users/mac/Desktop/webparsing/chromedriver'
browser = webdriver.Chrome(driver_path)#set up webdriver
browser.get(url) #打开初始页面
