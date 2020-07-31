import os
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
# import mplfinance as mpf
import yfinance as yf
from talib import abstract
# add base to the modules
from indicators import indicators as ind
from indicators import normalize as nor
from matplotlib import pyplot as plt
import math
import pathlib
from typing import Dict, List, Tuple, Union
import mplfinance as mpf

from datetime import datetime
# module_path = os.path.abspath(os.path.join('..'))
# if module_path not in sys.path:
#     sys.path.append(module_path)
from GoogleNews import GoogleNews
from nltk.sentiment.vader import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()
gn = GoogleNews()
gn.setlang('en')
gn.setperiod('d')

NEWS_TERMS = ["yahoo finance", "seeking alpha", "thestreet", "bloomberg", "marketwatch", "Motley", "Zacks"]


def get_google_news_scores(sym: str, date: datetime) -> Dict[str, float]:
    date_str = date.strftime("%m/%d/%Y")
    gn.setTimeRange(date_str, date_str)
    gn.search(sym)
    results: list = gn.result()
    gn.clear()
    for ext in NEWS_TERMS:
        gn.search(sym + ext)
        results.extend(gn.result())
        gn.clear()
    text = ""
    for r in results:
        text += r['title'] + ' ' + r['desc']
    scores = analyzer.polarity_scores(text)
    return scores


def load_ticker(sym, period="5y", interval="1d") -> pd.DataFrame:
    ydf = yf.Ticker(sym).history(period, interval).reset_index()
    scores = pd.DataFrame(index=ydf.index, columns=["news_neg", "news_neu", "news_pos", "news_cmp"])
    for idx, date in ydf["Date"].items():
        dt = date.to_pydatetime()
        gn_scores = get_google_news_scores(sym, date.to_pydatetime())
        print(f"Loaded news scoring for {sym} on {dt}: {gn_scores['compound']}")
        scores.loc[idx] = [gn_scores["neg"], gn_scores["neu"], gn_scores["pos"], gn_scores["compound"]]
    return pd.concat((ydf, scores), axis=1)


def load_tickers(symbols: list, period="5y", interval="1d") -> Dict[str, pd.DataFrame]:
    frames: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        frames[sym] = load_ticker(sym, period, interval)
    return frames


def plot_tickers(frames: Dict[str, pd.DataFrame], figsize=(18, 6)):
    for symbol, df in frames.items():
        plt.figure(figsize=figsize)
        df["Close"].plot(title=f"{symbol} Close")
    plt.show()


def add_indicators(df: pd.DataFrame) -> Tuple[Dict[str, List[ind.Indicator]], pd.DataFrame]:
    indicators = {
        'CNDL': [],  # candle patterns
        'PRICE': [],  # price relative indicators
        'BOOL': [],  # boolean indicators (1,0)
        'MACD': [],
        'ADX': [],
        'EOM': [],
        'PFC': [],  # the target for the nn

    }
    # set the mid price
    mid = ind.Mid()
    indicators['Mid'] = [mid]
    df = mid.concat(df, df['Open'], df['Close'])

    # numpy-ize
    o = df['Open'].to_numpy()
    h = df['High'].to_numpy()
    l = df['Low'].to_numpy()
    c = df['Close'].to_numpy()
    m = df['Mid'].to_numpy()

    # add candles patterns
    i_cndl = ind.CandlePatterns()
    indicators['CNDL'] = [i_cndl]
    df = i_cndl.concat(df, o, h, l, c)

    for i in [3, 5, 8, 13, 21, 34, 55, 89]:
        # EMA
        i_ema = ind.EMA(i)
        indicators['PRICE'].append(i_ema)
        df = i_ema.concat(df, m)
        # SMA
        i_sma = ind.SMA(i)
        indicators['PRICE'].append(i_sma)
        df = i_sma.concat(df, m)
        # triple EMA
        i_tema = ind.TEMA(i)
        indicators['PRICE'].append(i_tema)
        df = i_tema.concat(df, m)
        # comparison
        i_tema_gt_ema = ind.GreaterThan("TEMA>EMA")
        # reg slope
        i_lin = ind.RollingLin(i)
        indicators['PRICE'].append(i_lin)
        df = i_lin.concat(df, m)

    indicators['BOOL'] = []
    for i in [13, 21, 34, 55]:
        # BBANDS
        i_bb2 = ind.BBANDS(i)
        i_bb3 = ind.BBANDS(i)
        indicators['PRICE'].append(i_bb2)
        indicators['PRICE'].append(i_bb3)
        df = i_bb2.concat(df, c)
        # KEL
        i_kel = ind.KelCh(i)
        indicators['PRICE'].append(i_kel)
        df = i_kel(df, c)
        # squeeze
        i_sqz = ind.Squeeze(i, 2, 2)
        indicators['BOOL'].append(i_sqz)
        df = i_sqz(df, c)

    indicators['SAR'] = []
    for i in [.2, .3, .5, .8]:
        i_sar = ind.SAR(i, 1.)
        indicators['SAR'].append(i_sar)
        df = i_sar.concat(df, h, l)

    return indicators, df


def plot_ticker(df: pd.DataFrame):
    # fig = plt.figure(figsize=(20, 10))
    # gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], top=0.92, bottom=0.08, left=0.05, right=.95)
    # ax1 = fig.add_subplot(gs[0, :])
    # ax2 = fig.add_subplot(gs[1, :])
    # fig.suptitle(f"{ticker} {period} {interval}")
    # df["Close"].plot(title=f"Close", ax=ax1)
    # df["Volume"].plot(title="Volume", ax=ax2)
    # plt.show()
    plt.figure(figsize=(20, 10))
    mpf.plot(df[["Open", "High", "Low", "Close", "Volume"]].set_index(df["Date"]).tail(200),
             figratio=(40, 20),
             volume=True,
             type='candle',
             show_nontrading=False,
             style='yahoo',
             mav=(3, 6, 20, 200))


if __name__ == '__main__':

    symbols = [
        'SPY',
        'QQQ',
        'VOO',
        'SLV',
        'GLD',
        'EEM',
        'IEMG',
        'VTI',
        'IVV',
        'VEA',
        'VTV',
        'VXX',
        'GDX',
        'EFA',
        'XLF',
        'IWM',
        'GDX'
        'UCO',
        'XLP',
        'XLV',
        'IAU',
    ]

    period = '5y'
    interval = '1d'

    for sym in symbols:
        df = load_ticker(sym, period, interval)
        plot_ticker(df)


