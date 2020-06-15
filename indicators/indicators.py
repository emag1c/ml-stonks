import pandas as pd
import numpy as np

from talib import abstract
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.metrics import r2_score
from . import normalize
import math

SMA = abstract.SMA
BBANDS = abstract.BBANDS
EMA = abstract.EMA
ATR = abstract.ATR


def mid_price(open, close):
    return (open + close) / 2


def rolling_log(v):
    """
    :type v: pd.Series
    :return:
    """
    lg = np.log(v)  # type: np.ndarray
    return lg[-1]


def avg_price_change(open, close, period=32):
    """
    :param prices:
    :type open: pd.Series
    :type close: pd.Series
    :param period:
    :return:
    """

    diff = close - open

    s = pd.Series(index=diff.index)
    for i, v in diff.items():
        if i < period - 1:
            s.iat[i] = np.NaN
        elif i == period - 1:
            s.iat[i] = diff.iloc[0:i].mean()
        else:
            s.iat[i] = (s.iat[i-1] * period + v) / period

    return s


def rsi_volume(volume, open, close, period=32):
    """
    :type volume: pd.Series
    :type open: pd.Series
    :type close: pd.Series
    :rtype: pd.Series
    """
    # todo: finish

    diff = close - open
    gains = pd.Series(index=volume.index)
    losses = pd.Series(index=volume.index)
    flow = pd.Series(index=volume.index)

    avg_gain = 0
    avg_loss = 0

    for i in range(1, len(volume)):
        v = volume.iat[i]
        pv = volume.iat[-1]
        o = close.iat[i-1]
        c = close.iat[i]

        # this is a gain
        vchange = (v - pv) / v
        gain = 0
        loss = 0

        if c >= o:
            gain = (c - o) / o
            gains.iat[i] = gain * vchange
        else:
            loss = abs((c - o) / o)
            losses.iat[i] = loss * vchange

        if i == period + 1:
            # first avgs
            avg_gain = gains.iloc[i-period:i].sum() / period
            avg_loss = losses.iloc[i-period:i].sum() / period
        else:
            avg_gain = (avg_gain * (period-1) + gain) / period
            avg_loss = (avg_loss * (period-1) + loss) / period

        # print(f"gain = {gain}\tloss={loss}\tavg_gain={avg_gain}\tavg_loss={avg_loss}\tvchange={vchange}")
        flow.iat[i] = (100 - (100/(1+(avg_gain/avg_loss))))
    return flow


def money_flow_sum(volume, open, close, period=32, smoothing=6):
    """
    :type volume: pd.Series
    :type open: pd.Series
    :type close: pd.Series
    :rtype: pd.Series
    """
    s = pd.Series(index=volume.index, dtype=np.float64)

    for i, v in volume.items():
        s.at[i] = v if open.at[i] >= close.at[i] else -v

    s = s.rolling(period).sum()

    if smoothing > 1:
        return s.rolling(period).apply(normalize.min_max).rolling(smoothing).mean()
    return s.rolling(period).apply(normalize.min_max)


def easy_of_movement(volume, open, close, period=32):
    """
    the greater the value the greater the "perssure" is for this bar.
    Greater pressure = smaller bar, but more volume.
    The less volume and greater the movement, then the less pressure it has
    :param volume:
    :param open:
    :param close:
    :param period:
    :return:
    """
    # q is the inverse of the min-maxed bar size
    q = (close - open).abs().rolling(period).apply(normalize.min_max_unsigned) / 2
    v = ((volume.rolling(period).apply(normalize.min_max_unsigned) * -1) + 1) / 2
    return q + v


def money_flow_weighted(volume, open, close, period=32, regression_period=32):
    """
    :type volume: pd.Series
    :type open: pd.Series
    :type close: pd.Series
    :rtype: pd.Series
    """
    # weights is multiplier for the total volume
    # give more credit to wider gaps than smaller gaps
    s = (close - open).rolling(period).apply(normalize.min_max) * volume
    return s.rolling(regression_period).mean()


def percent_future_change(df, look_ahead=24):
    """
    :type df: pd.DataFrame
    :type look_ahead: int
    :rtype: pd.Series
    """
    s = pd.Series(index=df.index, dtype=np.float64)
    for i, row in df.iterrows():
        s.at[i] = 0
        base = df.iloc[i]["Close"]
        fi = i + look_ahead  # future index
        if fi in df.index:
            # loop over values in future index
            min_c = (df["Low"][i + 1:fi].min() - base) / abs(base)
            max_c = (df["High"][i + 1:fi].max() - base) / abs(base)

            if min_c < 1 and abs(min_c) > max_c:
                s.at[i] = min_c
            else:
                s.at[i] = max_c
    return s


def linear_regression_value(s):
    """
    :type s: pd.Series
    :rtype:
    """
    x = s.index.values.reshape(-1, 1)
    y = s.values.reshape(-1, 1)
    lr = LinearRegression()
    lr.fit(x, y)
    return lr.predict([[x[-1][0] + 1]])[-1][0]


def best_fit_regression_fn(max_degrees=6, predict=1):
    def fn(s):
        x = s.index.values.reshape(-1, 1)
        y = s.values.reshape(-1, 1)
        r2 = None
        poly = None  # type: PolynomialFeatures
        lr = None  # type: LinearRegression
        for i in range(1, max(1, min(max_degrees, math.floor(len(s) / 2)))):
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


def poly_regression_value(s):
    """
    :type s: pd.Series
    :rtype:
    """
    x = s.index.values.reshape(-1, 1)
    y = s.values.reshape(-1, 1)
    polynomial_features = PolynomialFeatures(degree=2)
    x_poly = polynomial_features.fit_transform(x)
    lr = LinearRegression()
    lr.fit(x_poly, y)
    #  predict one into the future
    future_val = polynomial_features.fit_transform([[x[-1][0]]])
    return lr.predict(future_val)[-1][0]


def mk_poly_fn(degree=2, predict=0):
    """
    :type s: pd.Series
    :rtype:
    """

    def fn(s):
        x = s.index.values.reshape(-1, 1)
        y = s.values.reshape(-1, 1)
        polynomial_features = PolynomialFeatures(degree=degree)
        x_poly = polynomial_features.fit_transform(x)
        lr = LinearRegression()
        lr.fit(x_poly, y)
        #  predict one into the future
        future_val = polynomial_features.fit_transform([[x[-1][0] + predict]])
        return lr.predict(future_val)[-1][0]

    return fn


def linear_regression_slope(s):
    """
    :type s: pd.Series
    :rtype: np.float64
    """
    x = s.index.values.reshape(-1, 1)
    y = s.values.reshape(-1, 1)
    lr = LinearRegression()
    lr.fit(x, y)
    y_pred = lr.predict(x)
    return (y_pred[-2][0] - y_pred[-1][0]) / (x[-2][0] - x[-1][0])


def linear_regression(s):
    """
    :type s: pd.Series
    :rtype: pd.Series
    """
    x = s.index.values.reshape(-1, 1)
    y = s.values.reshape(-1, 1)
    lr = LinearRegression()
    lr.fit(x, y)
    y_pred = lr.predict(x)
    return pd.DataFrame(y_pred)[0]


def rolling_regression_slope(s, period=32):
    """
    :type s: pd.Series
    """
    return s.rolling(period).apply(linear_regression_slope)


def regression_slope_min_max(s, period=32):
    return s.rolling(period).apply(normalize.min_max_unsigned)


def rolling_regression(s, period=32):
    """
    :type s: pd.Series
    """
    return s.rolling(period).apply(linear_regression_value)


def rolling_regression_poly(s, period=32, degree=2, predict=1):
    """
    :type s: pd.Series
    """
    return s.rolling(period).apply(mk_poly_fn(degree, predict))


def rolling_best_fit_regression(s, period=32, max_degrees=6, predict=1):
    """
    :type s: pd.Series
    """
    return s.rolling(period).apply(best_fit_regression_fn(max_degrees, predict))


# convergence divergence
def convergence_divergence_min_max(a, b, period=32, smoothing=6):
    """
    :type a: pd.Series
    :type b: pd.Series
    :type period: int
    :type smoothing: int
    :return:
    """
    cmp = a - b
    if smoothing < 2:
        return cmp.rolling(period).apply(normalize.min_max)
    return cmp.rolling(smoothing).mean().rolling(period).apply(normalize.min_max)
    # return cmp


def trend_min_max(s, period=32):
    return s.rolling(period).apply(normalize.min_max)


def keltner_channels(high, low, close, period = 20):
    ema = pd.Series(EMA(close, period))
    atr = pd.Series(ATR(high, low, close, period))
    upper = ema + (2 * atr)
    lower = ema - (2 * atr)

    return upper, lower, ema


def squeeze(bband_upper, bband_lower, keltner_upper, keltner_lower):
    return (bband_upper <= keltner_upper) & (bband_lower >= keltner_lower)


def difference_percentage_change(upper, lower, period=32, offset=0):
    """
    :type upper:  pd.Series
    :type lower: pd.Series
    :type period: int
    :rtype: pd.Series
    """
    diff = upper - lower
    p = pd.Series(index=diff.index, dtype=np.float64)
    s = pd.Series(index=diff.index, dtype=np.float64)
    for i in range(1 + offset, len(diff)):
        p.iat[i] = (diff.iat[i] - diff.iat[-1]) / diff.iat[i]
        if i == period + offset - 1:
            s.iat[i] = p.iloc[0:i].sum() / period
        else:
            s.iat[i] = (s.iat[i-1] * (period - 1) + p.iat[i]) / period
    return s


