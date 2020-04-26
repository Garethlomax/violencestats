#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 23:14:26 2020

@author: garethlomax
"""


import pandas as pd 
import time
import datetime
df = pd.read_csv("../Datasets/ged191.csv")

#inplace = true modifies by referenc (not exactxly but behaves like it)

#datetine strptime

a = time.mktime(time.strptime(df.date_start[0], "%Y-%m-%d"))
b = time.mktime(time.strptime(df.date_end[0], "%Y-%m-%d"))

c = datetime.timedelta(seconds=b-a)

# def time_delay_days()
# summarise by group 

# find number of people


# 152616

def time_delay_days(df):
    a = df['date_start']
    b = df['date_end']
    
    a = time.mktime(time.strptime(a, "%Y-%m-%d"))
    b = time.mktime(time.strptime(b, "%Y-%m-%d"))
    c = datetime.timedelta(seconds=b-a)
    return c.days

df['duration_days'] = df[['date_start', 'date_end']].apply(time_delay_days, axis = 1)


# df.assign(c=df['a']-df['b']

# make origional date 

def origin_time(strtime):
    return time.mktime(time.strptime(strtime, "%Y-%m-%d"))
    
df['rel_time'] = df['date_start'].apply(origin_time)

# summ = df.groupby('conflict_new_id').agg({'rel_time': 'min'})

df['first_event_time'] = df.groupby('conflict_new_id').rel_time.transform('min')

# TODO: sort by reltime first
df['time_diff'] = df.groupby('conflict_new_id').rel_time.diff()
    
    
    