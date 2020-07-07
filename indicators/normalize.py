import pandas as pd
import numpy as np
from typing import Union
# from sklearn.preprocessing import KBinsDiscretizer

def discretize(a: np.array, 
               bins: Union[list, tuple], 
               labels: Union[None, list, tuple]=None,
               right=False) -> np.array:
    """
    discretize a numpy array into bins
    if labels is given then apply those lables
    """
    if labels == None:
        labels = bins

    assert len(bins)== len(labels)

    bins_by_labels = dict(zip(range(0,len(bins)), labels))
    digitized = np.digitize(np.nan_to_num(a, nan=.0), bins=bins, right=right)
    res = np.empty((0))

    for v in digitized:
        for b, l in bins_by_labels.items():
            if v == b:
                res = np.append(res, [l])
                    
    return res

def min_max(a: np.array) -> np.array:
    """
    get the mean of this window len(v) and compare it to current price
    """
    # x = ((s - s.min()) / (s.max() - s.min())).iat[-1]
    # return x * 2 - 1
    return (a - a.min()) / (a.max() - a.min())


def min_max_v(a: np.array) -> np.float:
    """
    get the mean of this window len(v) and compare it to current price
    """
    return min_max(a)[-1]


def zscore(a: np.array) -> np.array:
    """
    get an array of zscores for a given np.array
    """
    return (a - a.mean()) / a.std(ddof=0)


def abs_mean(a: np.array) -> np.array:
    """
    get an array of zscores for a given np.array
    """
    return (a - a.mean()) / a.std(ddof=0)


def zscore_v(a: np.array) -> np.float:
    """
    get the last zscore value for an np.array
    """
    cmp = (a - a.mean()) / a.std(divmod)
    return cmp.iat[-1]


def log_v(v: np.array):
    return np.log(v)[-1]


