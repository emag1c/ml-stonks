import os
import sys
import pandas as pd
import yfinance as yf
from pathlib import Path
from typing import Dict, List, Tuple, Union
from multiprocessing import Pool, Queue
from datetime import datetime
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from scrape.scraperapi import GoogleNews
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from time import sleep
import argparse


analyzer = SentimentIntensityAnalyzer()

NEWS_TERMS = ["etf OR stock OR finance OR forecast -guru"]
# NEWS_TERMS = ["stock -guru", "etf -guru", "finance -guru"]
PROXY_API_KEY = ""


def get_google_news_scores(sym: str, date: datetime, pages=1) -> Dict[str, float]:
    gn = GoogleNews(key=PROXY_API_KEY)
    gn.set_lang('en')
    scores = {'neu': 0.,
              'neg': 0.,
              'pos': 0.,
              'compound': 0.}

    queries = []
    for ext in NEWS_TERMS:
        queries.append("*" + sym + " " + ext)

    try:
        gn.search(queries, list(range(1, pages)), start=date, end=date)
    except Exception as e:
        print(f"failed to load {sym} news on {date}: {e}")
        raise e
        # return scores

    text = " ".join(gn.get_text())
    if len(text) > 0:
        scores = analyzer.polarity_scores(text)
        print(f"{sym} on {date} had {scores['compound']} news score")
    else:
        print(f"no {sym} news on {date}")
    return scores


def load_ticker(sym, period="5y", interval="1d", workers=10, rate_limit=0.) -> pd.DataFrame:

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
            sleep(rate_limit)
            tasks[idx] = pool.apply_async(get_google_news_scores, (sym, date.to_pydatetime()))
        process(True)  # process the rest

    return pd.concat((ydf, scores), axis=1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", "--proxy-key", type=str, help="proxy API key used for proxy")
    parser.add_argument("-d", "--download-dir", type=str, help="download path", default=Path(os.getcwd() + "/data"))
    parser.add_argument("-w", "--workers", type=int, help="number of workers", default=10)
    parser.add_argument("-l", "--rate-limit", type=float,
                        help="rate limiting. Seconds to wait between loading google news scores", default=0.1)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    PROXY_API_KEY = args.proxy_key
    nltk.download("vader_lexicon")
    symbols = [
        'SPY',
        'QQQ',
        'TQQQ',
        'VOO',
        'SLV',
        'GLD',
        'IEMG',
        'VTI',
        'IVV',
        'VEA',
        'VTV',
        'EEM',
        'VXX',
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

    data_dir = Path(args.download_dir)

    for sym in symbols:
        df = load_ticker(sym, period, interval, args.workers, args.rate_limit)
        df.to_csv(data_dir / f'chart_yf_{sym}.csv')


