import pandas as pd
import numpy as np

from talib import abstract
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.metrics import r2_score
from . import normalize
import math
from typing import Tuple, List, Callable, Union, Dict, Any
from dataclasses import dataclass
from collections import OrderedDict
from abc import abstractmethod


def rolling_window(a: np.array, window: int):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides, writeable=False)


def rolling_apply(a: np.array, f, window: int, pad=False):
    wds: np.ndarray = rolling_window(a, window)
    if pad:
        pad_array = np.empty(window - 1)
        pad_array[:] = np.nan
        return np.concatenate((pad_array[:], np.apply_along_axis(f, 1, wds)))
    return np.apply_along_axis(f, 1, wds)


def log_v(v: np.array) -> np.float:
    """
    last value of log
    """
    return np.log(v)[-1]


@dataclass
class Parameter:
    name: str
    value: any

    def __repr__(self):
        return f'{self.name}={self.value}'


class Indicator:
    def __init__(self, params: List[Parameter] = None, outputs: List[str] = None, name=None):
        self.__params = OrderedDict()
        self.__param_str = ""
        self.__outputs = [""]
        self.__name = name if name else self.__class__.__name__

        if params is not None:
            for p in params:
                self.__params[p.name] = p
            self.__param_str = " " + ", ".join([x.__repr__() for x in self.__params.values()])

        if outputs is not None:
            self.__outputs = outputs

    @property
    def name(self) -> str:
        """ name is the name of this indicator including the parameter values """
        return self.__name + self.__param_str

    @property
    def output_names(self) -> List[str]:
        return [self.output_name(x) for x in range(self.num_outputs)]

    @property
    def num_outputs(self) -> int:
        return len(self.__outputs)

    def output_name(self, pos: int):
        """ output name is the name of the output as position ``pos`` """
        out_name = " " + self.__outputs[pos] if self.__outputs[pos] else ""
        return self.__name + out_name + self.__param_str

    def p_val(self, name: str):
        """ the value of parameter ``name`` """
        return self.__params[name].value

    @abstractmethod
    def _exec(self, *args, **kwargs):
        """ __exec required by all subclasses as the func that actually executes the indicator logic """
        pass

    def to_frame(self, *args, **kwargs) -> pd.DataFrame:
        """ returns a pandas dataframe with columns equal to the output columns """
        outputs = self._exec(*args, **kwargs)

        df_kwargs = {
            "index": None,
            "columns": self.output_names,
            "dtype": None,
            "copy": False
        }

        for k in df_kwargs.keys():
            if k in kwargs:
                df_kwargs[k] = kwargs[k]

        df = pd.DataFrame(**df_kwargs)
        for i in range(len(outputs)):
            df[df.columns[i]] = outputs[i]
        return df

    def concat(self, frame: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        df = self.to_frame(*args, **kwargs)

        concat_kwargs = {
            "axis": 1,
            "ignore_index": False,
            "sort": False,
            "join": "outer",
        }

        for k in concat_kwargs.keys():
            if k in kwargs:
                concat_kwargs[k] = kwargs[k]

        return pd.concat((df, frame), **concat_kwargs)

    def __call__(self, *args, **kwargs):
        return self._exec(*args, **kwargs)

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name


def mid(a: np.array, b: np.array) -> np.array:
    """
    mid value between 2 arrays
    """
    return (a + b) / 2


class Mid(Indicator):
    """
    Mid value between 2 series
    """
    def __init__(self, name=None):
        super(Mid, self).__init__(name=name)

    def _exec(self, a: np.array, b: np.array) -> Tuple[np.array]:
        return mid(a, b),


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


class ADOSC(Indicator):
    """
    Chaikin A/D Oscillator
    """

    def __init__(self, fast=3, slow=10, name=None):
        assert fast > 0
        assert slow > 0
        params = [
            Parameter("f", fast),
            Parameter("s", slow),
        ]
        super(ADOSC, self).__init__(params, name=name)

    def _exec(self,
              high: Union[np.array, pd.Series],
              low: Union[np.array, pd.Series],
              close: Union[np.array, pd.Series],
              volume: Union[np.array, pd.Series]) -> Tuple[np.array]:
        return adosc(high, low, close, volume, self.p_val("f"), self.p_val("s")),


def atr(high: Union[np.array, pd.Series],
        low: Union[np.array, pd.Series],
        close: Union[np.array, pd.Series],
        period: int) -> np.array:
    """ average true range """
    return np.array(abstract.ATR(high, low, close, period))


class ATR(Indicator):
    """
    average true range
    """

    def __init__(self, period: int, name=None):
        assert period > 0
        params = [
            Parameter("p", period),
        ]
        super(ATR, self).__init__(params, name=name)

    def _exec(self,
              high: Union[np.array, pd.Series],
              low: Union[np.array, pd.Series],
              close: Union[np.array, pd.Series]) -> Tuple[np.array]:
        return atr(high, low, close, self.p_val("p")),


def sma(a: Union[np.array, pd.Series], period: int) -> np.array:
    """ simple moving average """
    return np.array(abstract.SMA(a, period))


class SMA(Indicator):
    """
    simple moving average
    """

    def __init__(self, period: int, name=None):
        assert period > 0
        params = [
            Parameter("p", period),
        ]
        super(SMA, self).__init__(params, name=name)

    def _exec(self, series: Union[np.array, pd.Series]):
        return sma(series, self.p_val("p")),


def ema(a: Union[np.array, pd.Series], period: int) -> np.array:
    """ exponential moving average """
    return np.array(abstract.EMA(a, period))


class EMA(Indicator):
    """
    exponential moving average
    """

    def __init__(self, period: int, name=None):
        assert period > 0
        params = [
            Parameter("p", period),
        ]
        super(EMA, self).__init__(params, name=name)

    def _exec(self, series: Union[np.array, pd.Series]) -> Tuple[np.array]:
        return ema(series, self.p_val("p")),


def emacd(s, fast=6, slow=12) -> np.array:
    """
    exponential moving average convergence/divergance
    """
    ema(s, fast) - ema(s, slow)


class EMACD(Indicator):
    """
     exponential moving average convergence/divergance
    """

    def __init__(self, fast=6, slow=12, name=None):
        assert fast > 0
        assert slow > 0
        params = [
            Parameter("f", fast),
            Parameter("s", slow),
        ]
        super(EMACD, self).__init__(params, name=name)

    def _exec(self, series: Union[np.array, pd.Series]) -> Tuple[np.array]:
        return emacd(series, self.p_val("f"), self.p_val("s")),


def macd(s, fast=6, slow=12) -> pd.Series:
    """
    moving average convergence/divergance
    """
    sma(s, fast) - sma(s, slow)


class MACD(Indicator):
    """
    moving average convergence/divergance
    """

    def __init__(self, fast=6, slow=12, name=None):
        assert fast > 0
        assert slow > 0
        params = [
            Parameter("f", fast),
            Parameter("s", slow),
        ]
        super(MACD, self).__init__(params, name=name)

    def _exec(self, series: Union[np.array, pd.Series]) -> Tuple[np.array]:
        return macd(series, self.p_val("f"), self.p_val("s")),


def sar(high: Union[np.array, pd.Series],
        low: Union[np.array, pd.Series],
        acceleration=0,
        maximum=0) -> np.array:
    """
    Parabolic SAR
    """
    return np.array(abstract.SAR(high, low, acceleration, maximum))


class SAR(Indicator):
    """
    Parabolic SAR
    """

    def __init__(self, acceleration=0, maximum=0, name=None):
        params = [
            Parameter("a", acceleration),
            Parameter("m", maximum),
        ]
        super(SAR, self).__init__(params, name=name)

    def _exec(self, high: Union[np.array, pd.Series], low: Union[np.array, pd.Series]) -> Tuple[np.array]:
        return sar(low, high, self.p_val("a"), self.p_val("m")),


def bbands(close: Union[np.array, pd.Series], period=2, dev=2.) -> Tuple[np.array, np.array, np.array]:
    """ 
    Bollinger bands
    """
    return abstract.BBANDS(close, period, float(dev), float(dev))


class BBANDS(Indicator):
    """
    Bollinger bands
    """

    def __init__(self, period=16, dev=2., name=None):
        params = [
            Parameter("p", period),
            Parameter("d", float(dev)),
        ]
        outputs = [
            "UPPER",
            "MID",
            "LOWER"
        ]
        super(BBANDS, self).__init__(params, outputs, name)

    def _exec(self, series: Union[np.array, pd.Series]) -> Tuple[np.array, np.array, np.array]:
        return bbands(series, self.p_val("p"), self.p_val("d"))


def adx(high: Union[np.array, pd.Series],
        low: Union[np.array, pd.Series],
        close: Union[np.array, pd.Series],
        period=14) -> Tuple[np.array, np.array, np.array]:
    """ 
    Average Direction Index
    """
    return (abstract.ADX(high, low, close, period),
            abstract.PLUS_DI(high, low, close, period),
            abstract.MINUS_DI(high, low, close, period))


class ADX(Indicator):
    """
    Average Direction Index
    """

    def __init__(self, period=16, name=None):
        assert period > 0
        params = [
            Parameter("p", period)
        ]
        outputs = [
            "ADX",
            "PDI",
            "MDI"
        ]
        super(ADX, self).__init__(params, outputs, name=name)

    def _exec(self,
              high: Union[np.array, pd.Series],
              low: Union[np.array, pd.Series],
              close: Union[np.array, pd.Series]) -> Tuple[np.array, np.array, np.array]:
        return adx(high, low, close, self.p_val("p"))


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

    for i in range(period - 1, len(diff)):
        if i == period - 1:
            # first avg
            res[i] = diff[0:i].mean()
        else:
            res[i] = (res[i - 1] * period + res[i]) / period

    return res


class AvgChange(Indicator):
    """
    average change between two arrays a and b
    """

    def __init__(self, period=32, name=None):
        assert period > 0
        params = [
            Parameter("p", period)
        ]
        super(AvgChange, self).__init__(params, name=name)

    def _exec(self,
              a: Union[np.array, pd.Series],
              b: Union[np.array, pd.Series]) -> Tuple[np.array]:
        return avg_change(a, b, self.p_val("p")),


def eom(volume: np.array,
        close: np.array,
        min_max_period=200,
        smoothing=6) -> np.array:
    """
    Ease Of Movement
    the greater the value the greater the "pressure" is for this bar.
    Greater pressure = smaller bar, but more volume.
    The less volume and greater the movement, then the less pressure it has
    """
    assert len(volume) == len(close)

    o = np.delete(close, -1)  # remove last value (shift values backwards)
    # q is the inverse of the min-maxed bar size
    q = rolling_min_max(np.absolute((close[1:] - o)), min_max_period, True)
    v = (rolling_min_max(volume[1:], min_max_period, True) * -1.) + 1.
    return np.concatenate(([np.nan], sma((q + v) / 2, smoothing)))


class EOM(Indicator):
    """
    Ease Of Movement
    the greater the value the greater the "pressure" is for this bar.
    Greater pressure = smaller bar, but more volume.
    The less volume and greater the movement, then the less pressure it has
    """

    def __init__(self, min_max_period=200, smoothing=6, name=None):
        assert min_max_period > 0
        params = [
            Parameter("p", min_max_period),
            Parameter("s", smoothing),
        ]
        super(EOM, self).__init__(params, name=name)

    def _exec(self,
              volume: np.array,
              close: np.array, ) -> Tuple[np.array]:
        return eom(volume, close, self.p_val("p"), self.p_val("s")),


def eeom(volume: np.array,
         close: np.array,
         min_max_period=200,
         smoothing=6) -> np.array:
    """
    Ease Of Movement Exponential
    same as ease of movement  but with exponetial moving average
    """
    assert len(volume) == len(close)

    o = np.delete(close, -1)  # remove last value (shift values backwards)
    # q is the inverse of the min-maxed bar size
    q = rolling_min_max(np.absolute((close[1:] - o)), min_max_period, True)
    v = (rolling_min_max(volume[1:], min_max_period, True) * -1.) + 1.
    return np.concatenate(([np.nan], ema((q + v) / 2, smoothing)))


class EEOM(Indicator):
    """
    Exponential Ease Of Movement
    the greater the value the greater the "pressure" is for this bar.
    Greater pressure = smaller bar, but more volume.
    The less volume and greater the movement, then the less pressure it has
    """

    def __init__(self, min_max_period=200, smoothing=6, name=None):
        assert min_max_period > 0
        params = [
            Parameter("p", min_max_period),
            Parameter("s", smoothing),
        ]
        super(EEOM, self).__init__(params, name=name)

    def _exec(self, volume: np.array, close: np.array, ) -> Tuple[np.array]:
        return eeom(volume, close, self.p_val("p"), self.p_val("s")),


def eomcd(volume, close, min_max_period=200, fast=3, slow=6, exponential=False) -> np.array:
    """
    Ease Of Movement convergence divergence
    """
    if exponential:
        return eeom(volume, close, min_max_period, fast) - \
               eeom(volume, close, min_max_period, slow)
    else:
        return eom(volume, close, min_max_period, fast) - \
               eom(volume, close, min_max_period, slow)


class EOMCD(Indicator):
    """
    Ease Of Movement convergence divergence
    """

    def __init__(self, min_max_period=200, fast=3, slow=6, exponential=False, name=None):
        assert min_max_period > 0
        assert fast < slow
        params = [
            Parameter("p", min_max_period),
            Parameter("s", fast),
            Parameter("s", slow),
            Parameter("e", exponential)
        ]
        super(EOMCD, self).__init__(params, name=name)

    def _exec(self, volume: np.array, close: np.array, ) -> Tuple[np.array]:
        return eomcd(volume, close, self.p_val("p"), self.p_val("f"), self.p_val("s"), self.p_val("e")),


def pfc(a: np.array, look_ahead=6) -> np.array:
    """
    percent future change
    """
    fc = np.empty(len(a))
    fc[:] = np.nan
    for i in range(0, len(a)):
        base = a[i]
        fi = i + look_ahead + 1  # future index
        if fi < len(a):
            # loop over values in future index
            min_c = (a[i + 1:fi].min() - base) / abs(base)
            max_c = (a[i + 1:fi].max() - base) / abs(base)
            if min_c < 0 and abs(min_c) > max_c:
                fc[i] = min_c
            else:
                fc[i] = max_c
    return fc


class PercentFutureChange(Indicator):
    """
    percent future change
    """

    def __init__(self, look_ahead=6, name=None):
        assert look_ahead > 0
        params = [
            Parameter("la", look_ahead),
        ]
        super(PercentFutureChange, self).__init__(params, name=name)

    def _exec(self, series: np.array) -> Tuple[np.array]:
        return pfc(series, self.p_val("la")),


def apfc(a: np.array, look_ahead=6, avg_period=3) -> np.array:
    """
    average precent future change
    """
    fc = np.empty(len(a))
    fc[:] = np.nan
    for i in range(0, len(a)):
        base = a[i]
        fi = i + look_ahead + 1  # future index
        if fi < len(a):
            # loop over values in future index
            min_c = (a[i + 1:fi].min() - base) / abs(base)
            max_c = (a[i + 1:fi].max() - base) / abs(base)
            if min_c < 0 and abs(min_c) > max_c:
                fc[i] = min_c
            else:
                fc[i] = max_c
    return sma(fc, avg_period)


class AvgPercentFutureChange(Indicator):
    """
    percent future change averaged over given period
    """

    def __init__(self, look_ahead=6, period=3, name=None):
        assert look_ahead > 0
        params = [
            Parameter("la", look_ahead),
            Parameter("p", period),
        ]
        super(AvgPercentFutureChange, self).__init__(params, name=name)

    def _exec(self, series: np.array) -> Tuple[np.array]:
        return apfc(series, self.p_val("la"), self.p_val("p")),


def best_fit_poly_fn(max_degrees=6, predict=1) -> Callable:
    """
    create a func that will get the best fit polynomial regression fit
    """

    def fn(a):
        x = np.array(range(0, len(a))).reshape(-1, 1)
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
    x = np.array(range(0, len(a))).reshape(-1, 1)
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
        x = np.array(range(0, len(a))).reshape(-1, 1)
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
    x = np.array(range(0, len(a))).reshape(-1, 1)
    y = a.reshape(-1, 1)
    lr = LinearRegression()
    lr.fit(x, y)
    return lr.predict([[x[-1][0] + 1]])[-1][0]


def lin_slope(s: np.array) -> np.float:
    """
    linear regression slope
    """
    x = np.array(range(0, len(s))).reshape(-1, 1)
    y = s.reshape(-1, 1)
    lr = LinearRegression()
    lr.fit(x, y)
    y_pred = lr.predict(x)
    return (y_pred[-2][0] - y_pred[-1][0]) / (x[-2][0] - x[-1][0])


class LinSlope(Indicator):
    """
    linear slope
    """

    def __init__(self, name=None):
        super(LinSlope, self).__init__(name=name)

    def _exec(self, series: np.array) -> Tuple[np.array]:
        return lin_slope(series),


def lin(a: np.array):
    """
    linear regression
    """
    x = np.array(range(0, len(a))).reshape(-1, 1)
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


class RollingLin(Indicator):
    """
    rolling linear regression over a given window size
    """

    def __init__(self, window=32, name=None):
        assert window > 0
        params = [
            Parameter("w", window),
        ]
        super(RollingLin, self).__init__(params, name=name)

    def _exec(self, series: np.array) -> Tuple[np.array]:
        return rolling_lin(series, self.p_val("w")),


def rolling_lin_slope(a: np.array, period=32):
    """
    rolling linear regression slope values over a given window size
    """
    return rolling_apply(a, lin_slope, period, True)


class RollingLinSlope(Indicator):
    """
    rolling linear regression slope over a given window size
    """

    def __init__(self, window=32, name=None):
        assert window > 32
        params = [
            Parameter("w", window),
        ]
        super(RollingLinSlope, self).__init__(params, name=name)

    def _exec(self, series: np.array) -> Tuple[np.array]:
        return rolling_lin_slope(series, self.p_val("w")),


def rolling_poly_regression(a: np.array, window=32, degree=2, predict=1) -> np.array:
    """
    rolling 2 dimensional polynomial regression values over a given window size
    """
    return rolling_apply(a, mk_poly_fn(degree, predict), window, True)


class RollingPREG(Indicator):
    """
    rolling linear regression slope over a given window size
    """

    def __init__(self, window=32, degree=2, predict=1, name=None):
        assert window > 0
        assert degree > 0
        params = [
            Parameter("w", window),
            Parameter("d", degree),
            Parameter("p", predict),
        ]
        super(RollingPREG, self).__init__(params, name=name)

    def _exec(self, series: np.array) -> Tuple[np.array]:
        return rolling_poly_regression(series, self.p_val("w"), self.p_val("d"), self.p_val("p")),


def rolling_best_fit_regression(a: np.array, window=32, max_degrees=6, predict=1) -> np.array:
    """
    rolling best fit polynomial regression values over a given window size
    """
    return rolling_apply(a, best_fit_poly_fn(max_degrees, predict), window, True)


def keltner_channels(high: np.array,
                     low: np.array,
                     close: np.array,
                     period=20,
                     multiplier=2, ) -> Tuple[np.array, np.array]:
    """
    Keltner Channels
    """
    e = ema(close, period)
    a = atr(high, low, close, period)
    return e + (multiplier * a), e - (multiplier * a)


class KelCh(Indicator):
    """
    Keltner Channels
    """

    def __init__(self, period=20, multiplier=2, name=None):
        assert period > 0
        assert multiplier > 0
        params = [
            Parameter("p", period),
            Parameter("m", multiplier),
        ]
        outputs = [
            "Upper",
            "Lower"
        ]
        super(KelCh, self).__init__(params, outputs, name=name)

    def _exec(self,
              high: np.array,
              low: np.array,
              close: np.array, ) -> Tuple[np.array, np.array]:
        return keltner_channels(high, low, close, self.p_val("p"), self.p_val("m"))


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


class Squeeze(Indicator):
    """
    squeze returns a boolean array that is True if bbands are inside the keltner channels
    """

    def __init__(self, period=20, bband_dev=2., kel_multiplier=2, name=None):
        assert period > 0
        assert bband_dev > 0
        assert kel_multiplier > 0
        params = [
            Parameter("p", period),
            Parameter("bd", bband_dev),
            Parameter("km", kel_multiplier),
        ]
        super(Squeeze, self).__init__(params, name=name)

    def _exec(self,
              high: np.array,
              low: np.array,
              close: np.array, ) -> Tuple[np.array]:
        bbu, _, bbl = bbands(close, self.p_val("p"), self.p_val("bd"))
        kelu, kell = keltner_channels(high, low, close, self.p_val("p"), self.p_val("km"))
        return (bbu <= kelu) & (bbl >= kell),


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


class KelBbandDif(Indicator):
    """
    keltner bband difference
    bband upper - keltner upper
    and
    keltner lower - bband upper
    """

    def __init__(self, period=20, bband_dev=2., kel_multiplier=2, name=None):
        assert period > 0
        assert bband_dev > 0
        assert kel_multiplier > 0
        params = [
            Parameter("p", period),
            Parameter("bd", bband_dev),
            Parameter("km", kel_multiplier),
        ]
        super(KelBbandDif, self).__init__(params, name=name)

    def _exec(self,
              high: np.array,
              low: np.array,
              close: np.array, ) -> Tuple[np.array, np.array]:
        bbu, _, bbl = bbands(close, self.p_val("p"), self.p_val("bd"))
        kelu, kell = keltner_channels(high, low, close, self.p_val("p"), self.p_val("km"))
        return bbu - kelu, kell - bbl


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
            a[i] = (a[i - 1] * (period - 1) + p[i]) / period
    return a


class DeltaPC(Indicator):
    """
    delta percent change
    """

    def __init__(self, period=20, offset=0, name=None):
        assert period > 0
        params = [
            Parameter("p", period),
            Parameter("o", offset),
        ]
        super(DeltaPC, self).__init__(params, name=name)

    def _exec(self, upper: np.array, lower: np.array) -> Tuple[np.array]:
        return delta_pc(upper, lower, self.p_val("p"), self.p_val("o")),


def rsi(close: np.array, period=32) -> np.array:
    return abstract.RSI(close, period)


class RSI(Indicator):
    """
    rsi adjusted with volume
    """

    def __init__(self, period=20, name=None):
        assert period > 0
        params = [
            Parameter("p", period),
        ]
        super(RSI, self).__init__(params, name=name)

    def _exec(self, close: np.array) -> Tuple[np.array]:
        return rsi(close, self.p_val("p")),


def rsiv(volume: np.array,
         close: np.array,
         period=32,
         volume_weight=1) -> np.array:
    """
    rsi adjusted with volume
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
        o = close[i - 1]
        c = close[i]
        diff = abs(c - o) / o

        if c >= o:  # this is a gain
            gains[i] = diff * (e[i] * volume_weight)
        else:  # this is a loss
            losses[i] = diff * (e[i] * volume_weight)
        if i == period + 1:
            # first avgs
            avg_gain = gains[i - period:i].sum() / period
            avg_loss = losses[i - period:i].sum() / period
        else:
            avg_gain = (avg_gain * (period - 1) + diff) / period
            avg_loss = (avg_loss * (period - 1) + diff) / period

        flow[i] = (100 - (100 / (1 + (avg_gain / avg_loss))))

    return flow


class RSIV(Indicator):
    """
    rsi adjusted with volume
    """

    def __init__(self, period=20, volume_weight=1, name=None):
        assert period > 0
        assert volume_weight > 0

        params = [
            Parameter("p", period),
            Parameter("w", volume_weight),
        ]
        super(RSIV, self).__init__(params, name=name)

    def _exec(self, volume: np.array, close: np.array) -> Tuple[np.array]:
        return rsiv(volume, close, self.p_val("p"), self.p_val("w")),


def rolling_std(a, window, pad=False):
    """
    standard deviation applied to array with a rolling window
    """
    wds: np.ndarray = rolling_window(a, window)
    if pad:
        pad_array = np.empty(window - 1)
        pad_array[:] = np.nan
        return np.concatenate((pad_array[:], wds.std(axis=1)))
    return wds.std(axis=1)


class RollingSTD(Indicator):
    """
    standard deviation applied over a rolling window
    """

    def __init__(self, window=20, name=None):
        assert window > 0
        params = [
            Parameter("w", window),
        ]
        super(RollingSTD, self).__init__(params, name=name)

    def _exec(self, series: np.array, close: np.array) -> Tuple[np.array]:
        return rolling_std(series, self.p_val("w"), True),


def rolling_var(a, window, pad=False):
    """
    variance applied to array with a rolling window
    """
    wds: np.ndarray = rolling_window(a, window)
    if pad:
        pad_array = np.empty(window - 1)
        pad_array[:] = np.nan
        return np.concatenate((pad_array[:], wds.var(axis=1)))
    return wds.var(axis=1)


class RollingVAR(Indicator):
    """
    variance applied over a rolling window
    """

    def __init__(self, window=20, name=None):
        assert window > 0
        params = [
            Parameter("w", window),
        ]
        super(RollingVAR, self).__init__(params, name=name)

    def _exec(self, series: np.array, close: np.array) -> Tuple[np.array]:
        return rolling_var(series, self.p_val("w"), True),


def rolling_min_max(a: np.array, window: int, pad=False) -> np.array:
    """
    min max normalization applied to array with a rolling window
    """
    wds: np.ndarray = rolling_window(a, window)
    if pad:
        res = np.empty(window - 1)
        res[:] = np.nan
    else:
        res = np.empty(0)

    for wd in wds:
        m = normalize.min_max_v(wd)
        res = np.concatenate((res, [m]))
    return res


class RollingMinMax(Indicator):
    """
    min max (0,1) applied over a rolling window
    """

    def __init__(self, window=20, name=None):
        assert window > 0
        params = [
            Parameter("w", window),
        ]
        super(RollingMinMax, self).__init__(params, name=name)

    def _exec(self, series: np.array, close: np.array) -> Tuple[np.array]:
        return rolling_min_max(series, self.p_val("w"), True),


def ema_sma_cd(a: np.array, period: int) -> np.array:
    return ema(a, period) - sma(a, period)


class EMAvSMA(Indicator):
    """
    ema v. sma convergence divergence
    """

    def __init__(self, period=20, name=None):
        assert period > 0
        params = [
            Parameter("p", period),
        ]
        super(EMAvSMA, self).__init__(params, name=name)

    def _exec(self, series: np.array, close: np.array) -> Tuple[np.array]:
        return ema_sma_cd(series, self.p_val("p")),


def delta(a: np.array) -> np.array:
    b = np.delete(a, 0)
    return np.concatenate(([np.nan], b - a[:-1]))


def dema(series: Union[np.array, pd.Series], period=12) -> np.array:
    """
    double exponential moving average
    """
    return abstract.DEMA(series, period)


class DEMA(Indicator):
    """
    double exponential moving average
    """

    def __init__(self, period=20, name=None):
        assert period > 0
        params = [
            Parameter("p", period),
        ]
        super(DEMA, self).__init__(params, name=name)

    def _exec(self, series: np.array) -> Tuple[np.array]:
        return dema(series, self.p_val("p")),


def tema(series: Union[np.array, pd.Series], period=12) -> np.array:
    """
    Triple exponential moving average
    """
    return abstract.TEMA(series, period)


class TEMA(Indicator):
    """
    Triple exponential moving average
    """

    def __init__(self, period=20, name=None):
        assert period > 0
        params = [
            Parameter("p", period),
        ]
        super(TEMA, self).__init__(params, name=name)

    def _exec(self, series: np.array) -> Tuple[np.array]:
        return tema(series, self.p_val("p")),


def wma(series: Union[np.array, pd.Series], period=30) -> np.array:
    """
    Weighted Moving Average
    """
    return np.array(abstract.WMA(series, period))


class WMA(Indicator):
    """
    Weighted Moving Average
    """

    def __init__(self, period=20, name=None):
        assert period > 0
        params = [
            Parameter("p", period),
        ]
        super(WMA, self).__init__(params, name=name)

    def _exec(self, series: np.array) -> Tuple[np.array]:
        return wma(series, self.p_val("p")),


def mfi(high: Union[np.array, pd.Series],
        low: Union[np.array, pd.Series],
        close: Union[np.array, pd.Series],
        volume: Union[np.array, pd.Series],
        period=30) -> np.array:
    """
    money flow index
    """
    return np.array(abstract.MFI(high, low, close, volume, period))


class MFI(Indicator):
    """
    money flow index
    https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/MFI
    """

    def __init__(self, period=20, name=None):
        assert period > 0
        params = [
            Parameter("p", period),
        ]
        super(MFI, self).__init__(params, name=name)

    def _exec(self,
              high: Union[np.array, pd.Series],
              low: Union[np.array, pd.Series],
              close: Union[np.array, pd.Series],
              volume: Union[np.array, pd.Series]) -> Tuple[np.array]:
        return mfi(high, low, close, volume, self.p_val("p")),


def stoch(high: Union[np.array, pd.Series],
          low: Union[np.array, pd.Series],
          close: Union[np.array, pd.Series],
          fastk_period=5,
          slowk_period=3,
          slowd_period=3) -> Tuple[np.array, np.array]:
    """
    Stochastic Oscillator
    slowk, slowd = STOCH(high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    https://www.investopedia.com/terms/s/stochasticoscillator.asp
    """
    k, d = abstract.STOCH(high, low, close, fastk_period, slowk_period, 0, slowd_period, 0)
    return np.array(k), np.array(d)


class STOCH(Indicator):
    """
    Stochastic Oscillator
    slowk, slowd = STOCH(high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    https://www.investopedia.com/terms/s/stochasticoscillator.asp
    """

    def __init__(self, fastk_period=5, slowk_period=3, slowd_period=3, name=None):
        assert fastk_period > 0
        assert slowk_period > 0
        assert slowd_period > 0
        params = [
            Parameter("fk", fastk_period),
            Parameter("sk", slowk_period),
            Parameter("sd", slowd_period),
        ]
        outputs = [
            "SK",
            "SD"
        ]
        super(STOCH, self).__init__(params, outputs, name=name)

    def _exec(self,
              high: Union[np.array, pd.Series],
              low: Union[np.array, pd.Series],
              close: Union[np.array, pd.Series]) -> Tuple[np.array, np.array]:
        return stoch(high, low, close, self.p_val("fk"), self.p_val("sk"), self.p_val("sd"))


def stoch_rsi(close: Union[np.array, pd.Series],
              period=14,
              fastk_period=5,
              fastd_period=3) -> Tuple[np.array, np.array]:
    """
    Stochastic Oscillator of RSI
    fastk, fastd = STOCHRSI(close, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
    https://www.tradingview.com/support/solutions/43000502333-stochastic-rsi-stoch-rsi/
    """
    fast_k, fast_d = abstract.STOCHRSI(close, period, fastk_period, fastd_period, 0)
    return np.array(fast_k), np.array(fast_d)


class STOCHRSI(Indicator):
    """
    Stochastic Oscillator of RSI
    fastk, fastd = STOCHRSI(close, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
    https://www.tradingview.com/support/solutions/43000502333-stochastic-rsi-stoch-rsi/
    """

    def __init__(self, period=14, fastk_period=5, fastd_period=3, name=None):
        assert period > 0
        assert fastk_period > 0
        assert fastd_period > 0
        params = [
            Parameter("p", period),
            Parameter("fk", fastk_period),
            Parameter("fd", fastd_period),
        ]
        outputs = [
            "FK",
            "FD"
        ]
        super(STOCHRSI, self).__init__(params, outputs, name=name)

    def _exec(self, close: Union[np.array, pd.Series]) -> Tuple[np.array, np.array]:
        return stoch_rsi(close, self.p_val("p"), self.p_val("fk"), self.p_val("fd"))


def williams_r(high: Union[np.array, pd.Series],
               low: Union[np.array, pd.Series],
               close: Union[np.array, pd.Series],
               period=14) -> np.array:
    """
    Williams' %R
    real = WILLR(high, low, close, timeperiod=14)
    https://www.investopedia.com/terms/w/williamsr.asp
    """
    return np.array(abstract.WILLR(high, low, close, period))


class WILLR(Indicator):
    """
    Williams' %R
    real = WILLR(high, low, close, timeperiod=14)
    https://www.investopedia.com/terms/w/williamsr.asp
    """

    def __init__(self, period=14, name=None):
        assert period > 0
        params = [
            Parameter("p", period),
        ]
        super(WILLR, self).__init__(params, name=name)

    def _exec(self,
              high: Union[np.array, pd.Series],
              low: Union[np.array, pd.Series],
              close: Union[np.array, pd.Series]) -> Tuple[np.array]:
        return williams_r(high, low, close, self.p_val("p")),


def ad(high: Union[np.array, pd.Series],
       low: Union[np.array, pd.Series],
       close: Union[np.array, pd.Series],
       volume: Union[np.array, pd.Series]) -> np.array:
    """
    ADOSC - Chaikin A/D Oscillator
    real = AD(high, low, close, volume)
    """
    return np.array(abstract.AD(high, low, close, volume))


class AD(Indicator):
    """
    Chaikin A/D Oscillator
    real = AD(high, low, close, volume)
    """

    def __init__(self, name=None):
        super(AD, self).__init__(name=name)

    def _exec(self,
              high: Union[np.array, pd.Series],
              low: Union[np.array, pd.Series],
              close: Union[np.array, pd.Series],
              volume: Union[np.array, pd.Series]) -> Tuple[np.array]:
        return ad(high, low, close, volume),


def adosc(high: Union[np.array, pd.Series],
          low: Union[np.array, pd.Series],
          close: Union[np.array, pd.Series],
          volume: Union[np.array, pd.Series],
          fast_period: 3,
          slow_period: 10) -> np.array:
    """
    ADOSC - Chaikin A/D Oscillator
    https://www.investopedia.com/terms/c/chaikinoscillator.asp
    real = ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)
    """
    return np.array(abstract.ADOSC(high, low, close, volume, fast_period, slow_period))


class ADOSC(Indicator):
    """
    Chaikin A/D Oscillator
    https://www.investopedia.com/terms/c/chaikinoscillator.asp
    real = AD(high, low, close, volume)
    """

    def __init__(self, fast_period: 3, slow_period: 10, name=None):
        assert slow_period > fast_period > 0
        params = [
            Parameter("f", fast_period),
            Parameter("s", slow_period),
        ]
        super(ADOSC, self).__init__(params, name=name)

    def _exec(self,
              high: Union[np.array, pd.Series],
              low: Union[np.array, pd.Series],
              close: Union[np.array, pd.Series],
              volume: Union[np.array, pd.Series]) -> Tuple[np.array]:
        return adosc(high, low, close, volume, self.p_val("f"), self.p_val("s")),


def obv(close: Union[np.array, pd.Series],
        volume: Union[np.array, pd.Series]) -> np.array:
    """
    OBV - On Balance Volume
    https://www.investopedia.com/terms/o/onbalancevolume.asp
    real = OBV(close, volume)
    """
    return np.array(abstract.OBV(close, volume))


class OBV(Indicator):
    """
    OBV - On Balance Volume
    https://www.investopedia.com/terms/o/onbalancevolume.asp
    real = OBV(close, volume)
    """

    def __init__(self, name=None):
        super(OBV, self).__init__(name=name)

    def _exec(self, close: Union[np.array, pd.Series],
              volume: Union[np.array, pd.Series]) -> Tuple[np.array]:
        return obv(close, volume),


class CandlePatterns(Indicator):
    """
    Candle Pattern Recognition
    https://mrjbq7.github.io/ta-lib/func_groups/pattern_recognition.html
    """

    def __init__(self, output=None, name=None):
        default_outputs = [
            "2CROWS",
            "3BLACKCROWS",
            "3INSIDE",
            "3LINESTRIKE",
            "3OUTSIDE",
            "3STARSINSOUTH",
            "3WHITESOLDIERS",
            "ABANDONEDBABY",
            "ADVANCEBLOCK",
            "BELTHOLD",
            "BREAKAWAY",
            "CLOSINGMARUBOZU",
            "CONCEALBABYSWALL",
            "COUNTERATTACK",
            "DARKCLOUDCOVER",
            "DOJI",
            "DOJISTAR",
            "DRAGONFLYDOJI",
            "ENGULFING",
            "EVENINGDOJISTAR",
            "EVENINGSTAR",
            "GAPSIDESIDEWHITE",
            "GRAVESTONEDOJI",
            "HAMMER",
            "HANGINGMAN",
            "HARAMI",
            "HARAMICROSS",
            "HIGHWAVE",
            "HIKKAKE",
            "HIKKAKEMOD",
            "HOMINGPIGEON",
            "IDENTICAL3CROWS",
            "INNECK",
            "INVERTEDHAMMER",
            "KICKING",
            "KICKINGBYLENGTH",
            "LADDERBOTTOM",
            "LONGLEGGEDDOJI",
            "LONGLINE",
            "MARUBOZU",
            "MATCHINGLOW",
            "MATHOLD",
            "MORNINGDOJISTAR",
            "MORNINGSTAR",
            "ONNECK",
            "PIERCING",
            "RICKSHAWMAN",
            "RISEFALL3METHODS",
            "SEPARATINGLINES",
            "SHOOTINGSTAR",
            "SHORTLINE",
            "SPINNINGTOP",
            "STALLEDPATTERN",
            "STICKSANDWICH",
            "TAKURI",
            "TASUKIGAP",
            "THRUSTING",
            "TRISTAR",
            "UNIQUE3RIVER",
            "UPSIDEGAP2CROWS",
            "XSIDEGAP3METHODS",
        ]

        if output is not None:
            for o in output:
                assert o in default_outputs
        else:
            output = default_outputs

        super(CandlePatterns, self).__init__(outputs=output, name=name)

    def _exec(self,
              open: Union[np.array, pd.Series],
              high: Union[np.array, pd.Series],
              low: Union[np.array, pd.Series],
              close: Union[np.array, pd.Series]):
        return (
            np.array(abstract.CDL2CROWS(open, high, low, close)),
            np.array(abstract.CDL3BLACKCROWS(open, high, low, close)),
            np.array(abstract.CDL3INSIDE(open, high, low, close)),
            np.array(abstract.CDL3LINESTRIKE(open, high, low, close)),
            np.array(abstract.CDL3OUTSIDE(open, high, low, close)),
            np.array(abstract.CDL3STARSINSOUTH(open, high, low, close)),
            np.array(abstract.CDL3WHITESOLDIERS(open, high, low, close)),
            np.array(abstract.CDLABANDONEDBABY(open, high, low, close)),
            np.array(abstract.CDLADVANCEBLOCK(open, high, low, close)),
            np.array(abstract.CDLBELTHOLD(open, high, low, close)),
            np.array(abstract.CDLBREAKAWAY(open, high, low, close)),
            np.array(abstract.CDLCLOSINGMARUBOZU(open, high, low, close)),
            np.array(abstract.CDLCONCEALBABYSWALL(open, high, low, close)),
            np.array(abstract.CDLCOUNTERATTACK(open, high, low, close)),
            np.array(abstract.CDLDARKCLOUDCOVER(open, high, low, close)),
            np.array(abstract.CDLDOJI(open, high, low, close)),
            np.array(abstract.CDLDOJISTAR(open, high, low, close)),
            np.array(abstract.CDLDRAGONFLYDOJI(open, high, low, close)),
            np.array(abstract.CDLENGULFING(open, high, low, close)),
            np.array(abstract.CDLEVENINGDOJISTAR(open, high, low, close)),
            np.array(abstract.CDLEVENINGSTAR(open, high, low, close)),
            np.array(abstract.CDLGAPSIDESIDEWHITE(open, high, low, close)),
            np.array(abstract.CDLGRAVESTONEDOJI(open, high, low, close)),
            np.array(abstract.CDLHAMMER(open, high, low, close)),
            np.array(abstract.CDLHANGINGMAN(open, high, low, close)),
            np.array(abstract.CDLHARAMI(open, high, low, close)),
            np.array(abstract.CDLHARAMICROSS(open, high, low, close)),
            np.array(abstract.CDLHIGHWAVE(open, high, low, close)),
            np.array(abstract.CDLHIKKAKE(open, high, low, close)),
            np.array(abstract.CDLHIKKAKEMOD(open, high, low, close)),
            np.array(abstract.CDLHOMINGPIGEON(open, high, low, close)),
            np.array(abstract.CDLIDENTICAL3CROWS(open, high, low, close)),
            np.array(abstract.CDLINNECK(open, high, low, close)),
            np.array(abstract.CDLINVERTEDHAMMER(open, high, low, close)),
            np.array(abstract.CDLKICKING(open, high, low, close)),
            np.array(abstract.CDLKICKINGBYLENGTH(open, high, low, close)),
            np.array(abstract.CDLLADDERBOTTOM(open, high, low, close)),
            np.array(abstract.CDLLONGLEGGEDDOJI(open, high, low, close)),
            np.array(abstract.CDLLONGLINE(open, high, low, close)),
            np.array(abstract.CDLMARUBOZU(open, high, low, close)),
            np.array(abstract.CDLMATCHINGLOW(open, high, low, close)),
            np.array(abstract.CDLMATHOLD(open, high, low, close)),
            np.array(abstract.CDLMORNINGDOJISTAR(open, high, low, close)),
            np.array(abstract.CDLMORNINGSTAR(open, high, low, close)),
            np.array(abstract.CDLONNECK(open, high, low, close)),
            np.array(abstract.CDLPIERCING(open, high, low, close)),
            np.array(abstract.CDLRICKSHAWMAN(open, high, low, close)),
            np.array(abstract.CDLRISEFALL3METHODS(open, high, low, close)),
            np.array(abstract.CDLSEPARATINGLINES(open, high, low, close)),
            np.array(abstract.CDLSHOOTINGSTAR(open, high, low, close)),
            np.array(abstract.CDLSHORTLINE(open, high, low, close)),
            np.array(abstract.CDLSPINNINGTOP(open, high, low, close)),
            np.array(abstract.CDLSTALLEDPATTERN(open, high, low, close)),
            np.array(abstract.CDLSTICKSANDWICH(open, high, low, close)),
            np.array(abstract.CDLTAKURI(open, high, low, close)),
            np.array(abstract.CDLTASUKIGAP(open, high, low, close)),
            np.array(abstract.CDLTHRUSTING(open, high, low, close)),
            np.array(abstract.CDLTRISTAR(open, high, low, close)),
            np.array(abstract.CDLUNIQUE3RIVER(open, high, low, close)),
            np.array(abstract.CDLUPSIDEGAP2CROWS(open, high, low, close)),
            np.array(abstract.CDLXSIDEGAP3METHODS(open, high, low, close)),
        )


class GreaterThan(Indicator):
    """
    Returns an int array where 1 = a > b, 0 = b <= a
    """

    def __init__(self, name: str):
        assert name != ""
        super(GreaterThan, self).__init__(name=name)

    def _exec(self, a: np.array, b: np.array) -> Tuple[np.array]:
        return (a > b).astype(int),


class LessThan(Indicator):
    """
    Returns an int array where 0 = a > b, 1 = b <= a
    """

    def __init__(self, name: str):
        assert name != ""
        super(LessThan, self).__init__(name=name)

    def _exec(self, a: np.array, b: np.array) -> Tuple[np.array]:
        return (a < b).astype(int),


class Diff(Indicator):
    """
    Returns difference between two series
    """

    def __init__(self, name: str):
        assert name != ""
        super(Diff, self).__init__(name=name)

    def _exec(self, a: np.array, b: np.array) -> Tuple[np.array]:
        return a - b,


class Increasing(Indicator):
    """
    Returns an array of ints where value at x = 1 if value at x is greater than value at x-1
    """

    def __init__(self, name: str, pad_val: np.nan):
        assert name != ""
        self.p_val = pad_val
        super(Increasing, self).__init__(name=name)

    def _exec(self, v: np.array) -> Tuple[np.array]:
        diff = (v[1:] > v[:-1]).astype(np.int)
        if self.p_val is not None:
            pad = np.empty(1)
            pad[:] = self.p_val
            diff = np.concatenate(pad, diff)
        return diff,


class Decreasing(Indicator):
    """
    Returns an array of ints where value at x = 1 if value at x is less than value at x-1
    """

    def __init__(self, name: str, pad_val=np.nan):
        assert name != ""
        self.p_val = pad_val
        super(Decreasing, self).__init__(name=name)

    def _exec(self, v: np.array) -> Tuple[np.array]:
        diff = (v[:-1] > v[1:]).astype(np.int)
        if self.p_val is not None:
            pad = np.empty(1)
            pad[:] = self.p_val
            diff = np.concatenate(pad, diff)
        return diff,


class Compare(Indicator):
    """
    Returns an array of ints where value at x = 1 if comparison(a,b) is true
    """

    def __init__(self, name: str, comparison=">"):
        assert name != ""
        assert comparison in [">", ">=", "=", "<", "<=", "!="]
        self.comparison = comparison
        super(Compare, self).__init__(name=name)

    def _exec(self, a: np.array, b: np.array) -> Tuple[np.array]:
        if self.comparison == ">=":
            x = a >= b
        elif self.comparison == ">":
            x = a > b
        elif self.comparison == "=":
            x = a == b
        elif self.comparison == "<":
            x = a < b
        elif self.comparison == "<=":
            x = a <= b
        elif self.comparison == "!=":
            x = a != b
        return x.astype(np.int),


class Custom(Indicator):
    """
    Indicator that uses a custom function
    """

    def __init__(self, name: str, func: Callable):
        assert name != ""
        self.f = func
        super(Custom, self).__init__(name=name)

    def _exec(self, *args):
        return self.f(*args)
