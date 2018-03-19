#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 17:08:04 2017

@author: janny
"""
import pandas as pd

input_df = pd.read_csv("/mnt/share/Titanic/data/train.csv")
submit_df = pd.read_csv("/mnt/share/Titanic/data/test.csv")

df=pd.concat([input_df,submit_df])
df.reset_index(inplace=True)
df.drop('index',axis=1,inplace=True)

df.info() 

def isChild(age):
    if age<16:
        return 1
    else:
        return 0
        
df['Child']=df['Age'].apply(isChild)
df.Child[df.Surpe]