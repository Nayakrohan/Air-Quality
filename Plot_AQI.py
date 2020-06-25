# -*- coding: utf-8 -*-
"""
Created on Sun May 31 12:20:53 2020

@author: admin
"""

import pandas as pd
import matplotlib.pyplot as plt
import time

def avg(i):
    temp_i = 0
    average = []
    for rows in pd.read_csv('AQI/aqi{}.csv'.format(i),chunksize = 24):
        add_var = 0
        avg = 0.0
        data = []
        df = pd.DataFrame(data = rows)
        for index, row in df.iterrows():
            data.append(row['PM2.5'])
        for j in data:
            if type(j) is float or type(j) is int:
                add_var = add_var + j
            elif type(j) is str:
                if j!='NoData' and j!='PwrFail' and j!='---' and j!='InVld':
                    temp = float(j)
                    add_var = add_var + temp
        avg = add_var/24
        #print(add_var)
        #print(avg)
        temp_i = temp_i +1
            
        average.append(avg)
    return average                
              
if __name__ == '__main__':
    start_time = time.time()
    list_years = []
    for i in range(2013, 2019):
        list_years.append(avg(i))
    stop_time = time.time()
      
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
        


