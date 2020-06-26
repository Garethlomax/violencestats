#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 15:50:06 2020

@author: garethlomax
"""
import torch
import torch.nn as nn

class Embedder_Actors(nn.Module):
    def __init__(self):
        super.__init__(self,Embedder_Actors)
        
        self.l1 = nn.Linear(784, 300)
        self.l2 = nn.Linear(300, 50)
        self.l3 = nn.Linear(50, 300)
        self.l4 = nn.Linear(300, 784)
        
        self.activation = nn.ReLU()
        
    def encoder(self, x):
        x = self.activation(self.l1(x))
        x = self.activation(self.l2(x))
        return x
    
    def decoder(self, x):
        x = self.activation(self.l3(x))
        x = self.activation(self.l4(x))
        return x
    
    def forward(self, x):
        z = self.encoder(x)
        reconstruction = self.decoder(z)
        return reconstruction, z
        