import pandas as pd
import numpy as np

from talib import abstract
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.metrics import r2_score
from . import normalize
import math
from typing import Tuple, List, Callable, Union


def mid(a: np.array, b: np.array) -> np.array:
    """
    mid value between 2 arrays
    """
    return (a + b) / 2


def ad(high: Union[np.array, pd.Series],
        low: Union[np.array, pd.Series],
        close: Union[np.array, pd.Series],
        volume: Union[np.array, pd.Series]) -> np.array:
    """
    ADOSC - Chaikin A/D Oscillator
    """
    return np.array(abstract.AD(high, low, close, volume))


def obv(close: Union[np.array, pd.Series], 
        volume: Union[np.array, pd.Series]) -> np.array:
    """
    OBV - On Balance Volume
    """
    return np.array(abstract.OBV(close, volume))


def adosc(high: Union[np.array, pd.Series],
          low: Union[np.array, pd.Series],
          close: Union[np.array, pd.Series],
          volume: Union[np.array, pd.Series],
          fast=3,
          slow=10) -> np.array:
    """
    ADOSC - Chaikin A/D Oscillator
    """
    return np.array(abstract.ADOSC(high, low, close, volume, fast, slow))


def atr(high: Union[np.array, pd.Series],
        low: Union[np.array, pd.Series],
        close: Union[np.array, pd.Series],
        period: int) -> np.array:
    """ average true range """
    return np.array(abstract.ATR(high, low, close, period))


def sma(a: Union[np.array, pd.Series], period: int) -> np.array:
    """ simple moving average """
    return np.array(abstract.SMA(a, period))


def ema(a: Union[np.array, pd.Series], period: int) -> np.array:
    """ exponential moving average """
    return np.array(abstract.EMA(a, period))


def emacd(s, fast=6, slow=12) -> np.array:
    """
    exponential moving average convergence/divergance
    """
    ema(s, fast) - ema(s, slow)


def macd(s, fast=6, slow=12) -> pd.Series:
    """
    moving average convergence/divergance
    """
    sma(s, fast) - sma(s, slow)


def sar(high: Union[np.array, pd.Series],
        low: Union[np.array, pd.Series],
        acceleration=0,
        maximum=0) -> np.array:
    """
    Parabolic SAR
    """
    return abstract.SAR(high, low, acceleration, maximum)


def bbands(close: Union[np.array, pd.Series], period=2, dev=2) -> Tuple[np.array]:
    """ 
    Bollinger bands
    """
    return abstract.BBANDS(close, period, dev, dev)


def adx(high: Union[np.array, pd.Series],
        low: Union[np.array, pd.Series],
        close: Union[np.array, pd.Series],
        period=14) -> np.array:
    """ 
    Average Direction Index
    """
    return abstract.ADX(high, low, close, period)


def rolling_window(a: np.array, window: int):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides, writeable=False)


def rolling_apply(a: np.array, f, window: int, pad=False):
    wds: np.ndarray = rolling_window(a, window)
    if pad:
        pad_array = np.empty(window-1)
        pad_array[:] = np.nan
        return np.concatenate((pad_array[:], np.apply_along_axis(f, 1, wds)))
    return np.apply_along_axis(f, 1, wds)


def log_v(v: np.array) -> np.float:
    """
    last value of log
    """
    return np.log(v)[-1]


def avg_change(a: np.array,
               b: np.array,
               period=32) -> np.array:
    """
    average change between two arrays a and b
    """
    assert len(a) == len(b)

    diff: np.array = a - b
    res = np.empty(len(diff))
    res[:] = np.nan

    for i in range(period-1, len(diff)):
        if i == period - 1:
            # first avg
            res[i] = diff[0:i].mean()
        else:
            res[i] = (res[i-1] * period + res[i]) / period

    return res


def eom(volume: np.array,
        close: np.array,
        min_max_period=200,
        smoothing=6) -> np.array:
    """
    Ease Of Movement
    the greater the value the greater the "perssure" is for this bar.
    Greater pressure = smaller bar, but more volume.
    The less volume and greater the movement, then the less pressure it has
    """
    assert len(volume) == len(close)

    o = np.delete(close, -1)  # remove last value (shift values backwards)
    # q is the inverse of the min-maxed bar size
    q = rolling_min_max(np.absolute((close[1:]- o)), min_max_period, True)
    v = (rolling_min_max(volume[1:], min_max_period, True) * -1.) + 1.
    return np.concatenate(([np.nan], sma((q + v)/2, smoothing)))


def eomcd(volume, close, min_max_period=200, fast=3, slow=6) -> np.array:
    """
    Ease Of Movement convergence divergence
    """
    return eom(volume, close, min_max_period, fast) -\
        eom(volume, close, min_max_period, slow)


def pfc(a: np.array, look_ahead=24) -> np.array:
    """
    precent future change
    """
    fc = np.empty(len(a))
    fc[:] = np.nan
    for i in range(0, len(a)):
        base = a[i]
        fi = i + look_ahead + 1  # future index
        if fi in a.index:
            # loop over values in future index
            min_c = (a[i + 1:fi].min() - base) / abs(base)
            max_c = (a[i + 1:fi].max() - base) / abs(base)

            if min_c < 0 and abs(min_c) > max_c:
                a.at[i] = min_c
            else:
                a.at[i] = max_c
    return a


def best_fit_poly_fn(max_degrees=6, predict=1) -> Callable:
    """
    create a func that will get the best fit polynomial regression fit
    """
    def fn(a):
        x = np.array(range(0,len(a))).reshape(-1, 1)
        y = a.reshape(-1, 1)
        r2 = None
        poly = None  # type: PolynomialFeatures
        lr = None  # type: LinearRegression
        for i in range(1, max(1, min(max_degrees, math.floor(len(a) / 2)))):
            # test all regressions and take the best r2 value
            _poly = PolynomialFeatures(degree=i)
            x_poly = _poly.fit_transform(x)
            _lr = LinearRegression()
            _lr.fit(x_poly, y)
            y_pred = _lr.predict(_poly.fit_transform(x))
            _r2 = r2_score(y_pred, x)
            if r2 is None or _r2 > r2:
                r2 = _r2
                poly = _poly
                lr = _lr

        future_val = poly.fit_transform([[x[-1][0] + predict]])
        return lr.predict(future_val)[-1][0]

    return fn


def poly2d_v(a: np.array) -> np.float:
    """
    2 degree polynomial regression value
    """
    x = np.array(range(0,len(a))).reshape(-1, 1)
    y = a.reshape(-1, 1)
    polynomial_features = PolynomialFeatures(degree=2)
    x_poly = polynomial_features.fit_transform(x)
    lr = LinearRegression()
    lr.fit(x_poly, y)
    #  predict one into the future
    future_val = polynomial_features.fit_transform([[x[-1][0]]])
    return lr.predict(future_val)[-1][0]


def mk_poly_fn(degree=2, predict=0) -> Callable:
    """
    :type s: pd.Series
    :rtype:
    """

    def fn(a: np.array):
        x = np.array(range(0,len(a))).reshape(-1, 1)
        y = a.reshape(-1, 1)
        polynomial_features = PolynomialFeatures(degree=degree)
        x_poly = polynomial_features.fit_transform(x)
        lr = LinearRegression()
        lr.fit(x_poly, y)
        #  predict one into the future
        future_val = polynomial_features.fit_transform([[x[-1][0] + predict]])
        return lr.predict(future_val)[-1][0]

    return fn


def lin_v(a: np.array) -> np.float:
    """
    :type s: pd.Series
    :rtype:
    """
    x = np.array(range(0,len(a))).reshape(-1, 1)
    y = a.reshape(-1, 1)
    lr = LinearRegression()
    lr.fit(x, y)
    return lr.predict([[x[-1][0] + 1]])[-1][0]


def lin_slope(s: np.array) -> np.float:
    """
    linear regression slope
    """
    x = np.array(range(0,len(a))).reshape(-1, 1)
    y = s.reshape(-1, 1)
    lr = LinearRegression()
    lr.fit(x, y)
    y_pred = lr.predict(x)
    return (y_pred[-2][0] - y_pred[-1][0]) / (x[-2][0] - x[-1][0])


def lin(a: np.array):
    """
    linear regression
    """
    x = np.array(range(0,len(a))).reshape(-1, 1)
    y = a.reshape(-1, 1)
    lr = LinearRegression()
    lr.fit(x, y)
    y_pred = lr.predict(x)
    return pd.DataFrame(y_pred)[0]


def rolling_lin(a: np.array, window=32) -> np.array:
    """
    rolling linear regression over a given window size
    """
    return rolling_apply(a, lin_v, window, True)


def rolling_lin_slope(a: np.array, period=32):
    """
    rolling linear regression slope values over a given window size
    """
    return rolling_apply(a, lin_slope, period, True)


def rolling_poly2d(a: np.array, window=32, degree=2, predict=1) -> np.array:
    """
    rolling 2 dimensional polynomial regression values over a given window size
    """
    return rolling_apply(a, mk_poly_fn(degree, predict), window, True)


def rolling_best_fit_regression(a: np.array, window=32, max_degrees=6, predict=1) -> np.array:
    """
    rolling best fit polynomial regression values over a given window size
    """
    return rolling_apply(a, best_fit_poly_fn(max_degrees, predict), window, True)


def keltner_channels(high: np.array,
                     low: np.array,
                     close: np.array,
                     period=20,
                     multiplier=2,) -> Tuple[np.array]:
    """
    Keltner Channels
    """
    e = ema(close, period)
    a = atr(high, low, close, period)
    return e + (multiplier * a), e - (multiplier * a), e


def squeeze(bband_upper: np.array,
            bband_lower: np.array,
            keltner_upper: np.array,
            keltner_lower: np.array) -> np.array:
    """
    squeze returns a boolean array that is True if bbands are inside the keltner channels
    """
    assert len(bband_upper) == len(bband_lower) == len(
        keltner_lower) == len(keltner_upper)
    return (bband_upper <= keltner_upper) & (bband_lower >= keltner_lower)


def keltner_bband_cd(bband_upper: np.array,
                     bband_lower: np.array,
                     keltner_upper: np.array,
                     keltner_lower: np.array) -> Tuple[np.array]:
    """
    given uppare and lower bbands and keltner channel values,
    return the convergence/divergence of the upper bband and upper keltner channel
    and c/d of the kelnter lower and bband lower values
    """
    assert len(bband_lower) == len(bband_upper) == len(
        keltner_lower) == len(keltner_upper)
    return bband_upper - keltner_upper, keltner_lower - bband_lower


def delta_pc(upper: np.array,
             lower: np.array,
             period=32,
             offset=0) -> np.array:
    """
    delta percent change
    """
    assert len(upper) == len(lower)
    diff = upper - lower
    p = np.empty(len(diff))  # percent change
    a = np.empty(len(diff))  # delta percent change
    p[:] = np.nan
    a[:] = np.nan
    for i in range(1 + offset, len(diff)):
        p[i] = (diff[i] - diff[-1]) / diff[i]
        if i == period + offset + 1:
            a[i] = p[0:i].sum() / period
        else:
            a[i] = (a[i-1] * (period - 1) + p[i]) / period
    return a


def rsiv(volume: np.array,
         close: np.array,
         period=32,
         volume_weight=1) -> np.array:
    """
    rsi adusted with ease of movement 
    """
    assert len(volume) == len(open) == len(close)

    gains = np.empty(len(volume))
    losses = np.empty(len(volume))
    flow = np.empty(len(volume))
    gains[:] = np.nan
    losses[:] = np.nan
    flow[:] = np.nan

    e = eom(volume, close, period)

    avg_gain = 0.
    avg_loss = 0.

    for i in range(1, len(volume)):
        o = close[i-1]
        c = close[i]
        diff = abs(c - o) / o

        if c >= o:  # this is a gain
            gains[i] = diff * (e[i] * volume_weight)
        else:  # this is a loss
            losses[i] = diff * (e[i] * volume_weight)
        if i == period + 1:
            # first avgs
            avg_gain = gains[i-period:i].sum() / period
            avg_loss = losses[i-period:i].sum() / period
        else:
            avg_gain = (avg_gain * (period-1) + diff) / period
            avg_loss = (avg_loss * (period-1) + diff) / period

        flow[i] = (100 - (100/(1+(avg_gain/avg_loss))))

    return flow


def rolling_std(a, window, pad=False):
    """
    standard deviation applied to array with a rolling window
    """
    wds: np.ndarray = rolling_window(a, window)
    if pad:
        pad_array = np.empty(window-1)
        pad_array[:] = np.nan
        return np.concatenate((pad_array[:], wds.std(axis=1)))
    return wds.std(axis=1)


def rolling_var(a, window, pad=False):
    """
    variance applied to array with a rolling window
    """
    wds: np.ndarray = rolling_window(a, window)
    if pad:
        pad_array = np.empty(window-1)
        pad_array[:] = np.nan
        return np.concatenate((pad_array[:], wds.var(axis=1)))
    return wds.var(axis=1)


def rolling_min_max(a: np.array, window: int, pad=False) -> np.array:
    """
    min max normalization applied to array with a rolling window
    """
    wds: np.ndarray = rolling_window(a, window)
    if pad:
        res = np.empty(window-1)
        res[:] = np.nan
    else:
        res = np.empty(0)

    for wd in wds:
        m = normalize.min_max_v(wd)
        res = np.concatenate((res, [m]))
    return res


def ema_sma_cd(a: np.array, period:int) -> np.array:
    return ema(a) - sma(a)