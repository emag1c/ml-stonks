import os
import sys
import pandas as pd
import yfinance as yf
from pathlib import Path
from typing import Dict, List, Tuple, Union
import mplfinance as mpf
from multiprocessing import Pool, Queue
from datetime import datetime
# module_path = os.path.abspath(os.path.join('..'))
# if module_path not in sys.path:
#     sys.path.append(module_path)
from GoogleNews import GoogleNews
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from time import sleep

analyzer = SentimentIntensityAnalyzer()
gn = GoogleNews()
gn.setlang('en')
gn.setperiod('d')


NEWS_TERMS = ["stock", "ETF", "yahoo finance", "seeking alpha", "marketwatch", "Zacks"]


def get_google_news_scores(sym: str, date: datetime, pages=1) -> Dict[str, float]:
    date_str = date.strftime("%m/%d/%Y")
    gn.setTimeRange(date_str, date_str)
    results = []
    for ext in NEWS_TERMS:
        for i in range(pages):
            gn.search(sym + " " + ext)
            if i > 0:
                gn.getpage(i + 1)
            results.extend(gn.result())
            gn.clear()

    text = ""
    for r in results:
        text += r['title'] + ' ' + r['desc'] + ' '
    scores = analyzer.polarity_scores(text)
    print(f"{sym} on {date} had {scores['compound']} news score")
    return scores


def get_google_news_scores_with_key(key, sym: str, date:datetime) -> tuple:
    return key, get_google_news_scores(sym, date)


def load_ticker(sym, period="5y", interval="1d", workers=10) -> pd.DataFrame:

    ydf = yf.Ticker(sym).history(period, interval).reset_index()
    scores = pd.DataFrame(index=ydf.index, columns=["news_neg", "news_neu", "news_pos", "news_cmp"])

    tasks = {}

    def process(remainder=False):
        done = []
        if len(tasks) > 0:
            for k, task in tasks.items():
                if task.ready():
                    res = task.get()
                    scores.loc[k] = [res["neg"], res["neu"], res["pos"],  res["compound"]]
                    done.append(k)

        if len(done) > 0:
            for d in done:
                del tasks[d]

        if (remainder is False and workers - len(tasks) > 0) or (remainder is True and len(tasks) == 0):
            return
        else:
            sleep(0.1)
            return process()

    with Pool(workers) as pool:
        for idx, date in ydf["Date"].items():
            # process before adding more to the pool
            process()
            tasks[idx] = pool.apply_async(get_google_news_scores, (sym, date.to_pydatetime()))
        process(True)  # process the rest

    return pd.concat((ydf, scores), axis=1)


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

    data_dir = Path('/Users/eric/Projects/stonks/ml/rnn/data')

    for sym in symbols:
        df = load_ticker(sym, period, interval, 20)
        df.to_csv(data_dir / f'chart_yf_{sym}.csv')


