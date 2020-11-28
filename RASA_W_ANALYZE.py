# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 12:00:52 2020

@author: RASA-
"""

def RASA_W_ANALYZE(wDate):
    import pandas as pd
    import numpy as np
    dates = pd.date_range(start='20171231', end='20210501',freq='M')
    mean_data = []
    Mean_data = []
    Mean_value=[]
    Date_value =[]
    for i in range(0,dates.size-1):
            if np.any((dates[i+1]>= wDate['date_time']) & (dates[i]< wDate['date_time'])):
                	t2= wDate[(dates[i+1]>= wDate['date_time']) & (dates[i]< wDate['date_time'])]
                	mean_data.append(t2)
                	Mean_data.append(t2.loc[:,['date_time', 'maxtempC', 'mintempC']])
                	mean_v=int (((t2.loc[:,[ 'maxtempC', 'mintempC']]).astype('int').mean()).mean())
                	Date_value.append(str(t2.iloc[0,0].date().year) + '-' +str(t2.iloc[0,0].date().month))
                	Mean_value.append(mean_v) 
    MEAN = np.array(Mean_value)
    
    return [MEAN, Date_value]