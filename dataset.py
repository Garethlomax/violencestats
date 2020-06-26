#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 21:44:30 2020

@author: garethlomax
"""

import torch
from torch.utils.data import Dataset
class Embed_dataset(Dataset):
    def __init__(self, trace):
        
        #convert to tensor
        self.trace = torch.tensor(self.trace).double()
    def __len__(self):
        
        return len(self.trace)
    
    def __getitem__(self, i):
        return self.trace[i]
    



