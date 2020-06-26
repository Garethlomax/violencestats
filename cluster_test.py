#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 15:15:03 2020

@author: garethlomax
"""


from sklearn.cluster import KMeans
import numpy as np
X = np.random.randint(0, 100, size=(1000,82))
kmeans = KMeans(n_clusters=20, random_state=0).fit(X)
