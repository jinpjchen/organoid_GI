import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy

def nzflat(M, k=1):
    t = M.where(np.triu(np.ones(M.shape), k=k).astype(np.bool)).stack()
    return t[t != 0]

def upper_triangle(M, k=1):
    t = M.where(np.triu(np.ones(M.shape), k=k).astype(np.bool)).stack()
    return t

def _strip_cat_cols(df):
    cat_cols = df.select_dtypes(include=['category']).columns
    if len(cat_cols) > 0:
        print('! Converting categorical columns to string...')
        out = df.copy()
        for col in cat_cols:
            out[col] = out[col].astype('str')
    else:
        out = df
    return out
    
def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    # From: https://github.com/oliviaguest/gini
    # based on bottom eq: http://www.statsdirect.com/help/content/image/stat0206_wmf.gif
    # from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    array = array.flatten() #all values are treated equally, arrays must be 1d
    if np.amin(array) < 0:
        array -= np.amin(array) #values cannot be negative
    array += 0.0000001 #values cannot be 0
    array = np.sort(array) #values must be sorted
    index = np.arange(1,array.shape[0]+1) #index per array element
    n = array.shape[0]#number of array elements
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array))) #Gini coefficient
