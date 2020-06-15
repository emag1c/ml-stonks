import pandas as pd
# from sklearn.preprocessing import KBinsDiscretizer


def discretize_series(input, bins=None, labels=None):
    """

    :type input: pd.Series
    :type n_bins: int
    :return: pd.Series
    """

    if bins is None:
        bins = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]

    return pd.cut(input, bins, labels=labels)


def discretize_series_sklearn(input, bins=None, labels=None):
    """

    :type input: pd.Series
    :type n_bins: int
    :return: pd.Series
    """

    if bins is None:
        bins = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]

    return pd.cut(input, bins, labels=labels)


def min_max(s):
    """
    get the mean of this window len(v) and compare it to current price
    :type v: pd.Series
    :rtype: np.float64
    """
    # x = ((s - s.min()) / (s.max() - s.min())).iat[-1]
    # return x * 2 - 1
    return ((s - s.min()) / (s.max() - s.min())).iat[-1]


def min_max_unsigned(s):
    """
    get the mean of this window len(v) and compare it to current price
    :type v: pd.Series
    :rtype: np.float64
    """
    # factor in the trend
    return ((s - s.min()) / (s.max() - s.min())).iat[-1]


def zscore(v):
    cmp = (v - v.mean()) / v.std(ddof=0)
    return cmp.iat[-1]



