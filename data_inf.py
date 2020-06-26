#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 14:53:47 2020

@author: garethlomax
"""
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import RandomSampler
import numpy as np 
import pandas as pd

class CustomLoader(object):

    def __init__(self, dataset, my_bsz, drop_last=True):
        self.ds = dataset
        self.my_bsz = my_bsz
        self.drop_last = drop_last
        self.sampler = RandomSampler(dataset)

    def __iter__(self):
        batch = torch.Tensor()
        for idx in self.sampler:
            batch = torch.cat([batch, self.ds[idx]])
            while batch.size(0) >= self.my_bsz:
                if batch.size(0) == self.my_bsz:
                    yield batch
                    batch = torch.Tensor()
                else:
                    return_batch, batch = batch.split([self.my_bsz,batch.size(0)-self.my_bsz])
                    yield return_batch
        if batch.size(0) > 0 and not self.drop_last:
            yield batch

class VarLenDataset(Dataset):
    
    def __init__(self, df):
        self.df = df
        
        un, counts = np.unique(df.event_count, return_counts = True)
        self.length_list = un
        self.length_list_frequency = counts/un
        
    def get_len(self, length):
        """ doesnt act like normal get item, 
        returns slice of dataframe with event count matching this
        """
        sample = self.df[self.df.event_count == length]
        return sample
        #TODO: change this to sorting variable
        # sample.group_by('priogrid_gid') #TODO: get rid of the group
        
        
        
    
                
        
class VarLenDataloader(object):
    """ Dataloader for lstm entries grouped by length"""
    def __init__(self, dataset, max_batch_size, shuffle = True):
        # dataset should have summary of sizes of each length ect
        self.feature_len = len(dataset.df.keys()) # TODO: fix this
        self.dset = dataset
        self.length_list = dataset.length_list
        self.length_freq = dataset.length_list_frequency
        self.index_list = np.arange(len(self.length_list))
        
    def shuffle_lengths(self):
        """shuffles index of length_list"""
        np.random.shuffle(self.index_list)
        
    def to_numpy(self, gdf, length):
        """df should be grouped"""
        no_groups = len(gdf)
        output_array = np.zeros(shape = (no_groups,length,self.feature_len))
        for i, group_iter in enumerate(gdf):
            output_array[i] = group_iter[1] #grouped iteration
            # returns group name and dataframe as group_iter
            # i is just a number.
            return output_array
        
        
            
            
             
            
            
            
            
        
    def __iter__(self):
        self.shuffle_lengths()
        for i in self.index_list:
            ul = self.dset.get_len(self.length_list[i]) # get dataset of uniform length
            ul = ul.sort_values('id').groupby('priogrid_gid') # we have now grouped into sorted grids
            # we now operate over the different dataframes
            self.output_array = self.to_numpy(ul, self.length_list[i])
            yield self.output_array
    
            
            
            
                
            
            
            
    
            
        
        
        
    # def __iter__(self):
        
        
        