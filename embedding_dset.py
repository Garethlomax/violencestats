#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 16:00:36 2020

@author: garethlomax
"""

# one hot encoding for actors. 

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

df = pd.read_csv("../Datasets/ged191.csv")

print(len(df.side_a.unique()))

print(len(df.side_b.unique()))

print(len(df.conflict_name.unique()))

a = set(df.side_a.unique())

b = set(df.side_b.unique())
c = a.union(b)
print(len(c)) # total sides involved. 

#
c = list(c)
c.append("dummy_to_make_non_prime")

sides = np.array(c)
# sides = df.conflict_name.unique()

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(sides)

onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)


np.save("sides_names", sides)
np.save("sides_integer_mapping", integer_encoded)
np.save("sides_onehot.npy", onehot_encoded)

# # define example
# data = ['cold', 'cold', 'warm', 'cold', 'hot', 'hot', 'warm', 'cold', 'warm', 'hot']
# values = np.array(data)
# print(values)
# # integer encode

# print(integer_encoded)
# # binary encode
# onehot_encoder = OneHotEncoder(sparse=False)
# integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
# onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
# print(onehot_encoded)
# # invert first example
# inverted = label_encoder.inverse_transform([np.argmax(onehot_encoded[0, :])])
# print(inverted)