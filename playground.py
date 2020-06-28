#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 23:14:26 2020

@author: garethlomax
"""


import pandas as pd 
import time
import datetime
import numpy as np
from data_inf import VarLenDataloader, VarLenDataset
df = pd.read_csv("../Datasets/ged191.csv")

#inplace = true modifies by referenc (not exactxly but behaves like it)

#datetine strptime

sorting_variable = 'priogrid_gid'

a = time.mktime(time.strptime(df.date_start[0], "%Y-%m-%d"))
b = time.mktime(time.strptime(df.date_end[0], "%Y-%m-%d"))

c = datetime.timedelta(seconds=b-a)

# def time_delay_days()
# summarise by group 

# find number of people

# filter out gids with only one group
# do we filter again? 
df = df.groupby('priogrid_gid').filter(lambda x: len(x) > 4) # we want temporal behaviour
df = df.groupby('priogrid_gid').filter(lambda x: len(x) < 100) # we want temporal behaviour


# 152616
def offset_log(x):
    return np.log(x + 1)

df['log_deaths_a'] = offset_log(df.deaths_a)
df['log_deaths_b'] = offset_log(df.deaths_b)
df['log_best'] = offset_log(df.best)


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

df['first_event_time'] = df.groupby(sorting_variable).rel_time.transform('min')
# df['event_count'] = df.groupby(sorting_variable).transform('count')
# df['event_count'] = df.groupby(sorting_variable).size()
df['event_count'] = df.groupby(sorting_variable).priogrid_gid.transform('count')

un, counts = np.unique(df.event_count, return_counts = True)

number_of_seq_length = counts/un

df['violence_category'] = pd.Categorical(df.type_of_violence)
df = pd.concat([df, pd.get_dummies(df['violence_category'], prefix = 'viol')], axis=1)
# TODO: sort by reltime first
# This still gives us nans
# TODO: sort out what happens in first sequence
# not giving proper time difference - lagged by one 
df['time_diff'] = df.sort_values(["rel_time", 'id']).groupby(sorting_variable).rel_time.diff()
df['time_diff'] = df.time_diff.fillna(0)
def time_delay_zero(df):
    # origin = time.mktime((0.0,))
    return datetime.timedelta(seconds=df).days
    
df['time_diff_days'] = df.time_diff.apply(time_delay_zero)


df[df.conflict_new_id == 644][['date_start', 'rel_time', 'time_diff_days']]



#hdf5
# group by event then prio with sort by time then id - save this as hdf5 then 
#load custom way.
        
# finds lenght for each grid cell 
df = df[['viol_1','viol_2',sorting_variable, 'event_count', 'id','side_a']]

a = df.groupby(sorting_variable).agg({'event_count' : 'mean'})

# b = VarLenDataset(df)

# c = VarLenDataloader(b, 100)

# j = 0
# for d in c:
#     if j ==2:
#         break
#     j += 1
    
    
def array_dict_map(dictionary, keys):
    """ for mappin embeddings from neural nets"""
    out = np.zeros((len(keys), 50))
    for i,key in enumerate(keys):
        out[i] = dictionary[key]
    return out



def embed_dict(keys, vals):
    dictionary = {key: val for key, val in zip(keys, vals)}
    return dictionary
    
    
def embedded_side_dict():
    sides_names = np.load("sides_names.npy")
    sides_generated = np.load("sides_generated.npy")
    return embed_dict(sides_names, sides_generated)

dictionary = embedded_side_dict()

def embed_df_col(df, column, dictionary):
    """takes one column"""
    new_df = array_dict_map(dictionary, df[column])
    
    return pd.DataFrame(new_df)

new_df = embed_df_col(df, 'side_a', dictionary)













