import os
import sys
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import argparse
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from scrape.scraperapi import GoogleNews
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from time import sleep


SHORT_URL = "https://eodhistoricaldata.com/api/shorts/{sym}.{exc}?api_token={token}&from={from_date}"
EOD_URL = "https://eodhistoricaldata.com/api/eod/{sym}.{exc}?api_token={token}&from={from_date}&to={to_date}"
API_KEY = ""


def to_api_date(date: datetime) -> str:
    return date.strftime("%Y-%m-%d")


def short_interest(sym="AAPL", exc="US", token="OeAFFmMliFG5orCUuwAKQ8l4WWFQ67YX", days=365*5) -> pd.DataFrame:
    from_date = datetime.now() - timedelta(days=days)
    url = SHORT_URL.format(sym=sym, exc=exc, token=token, from_date=to_api_date(from_date))
    print(url)
    df = pd.read_csv(url, skipfooter=1)
    return df


def eod(start_date:datetime, sym="AAPL", exc="US", token="OeAFFmMliFG5orCUuwAKQ8l4WWFQ67YX") -> pd.DataFrame:
    to_date = start_date + timedelta(days=100)
    print(f"From: {start_date}, To: {to_date}")
    now = datetime.now()
    df = None
    while start_date < now:
        url = EOD_URL.format(sym=sym, exc=exc, token=token, from_date=to_api_date(start_date), to_date=to_api_date(to_date))
        print(url)
        _df = pd.read_csv(url, skipfooter=1)
        if df is None:
            df = _df
        else:
            df = pd.concat((df, _df))
        start_date += timedelta(days=100)
        to_date += timedelta(days=100)

    return df


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", "--api-key", type=str, help="API key used for proxy")
    parser.add_argument("-p", "--path", type=str, help="download path", default=Path(os.getcwd() + "/data"))
    return parser.parse_args()


def save_with_short(sym: str, exc: str, token: str, file: Path) -> pd.DataFrame:
    short_df = short_interest(sym, exc=exc, token=API_KEY)
    print(short_df.tail(10))
    start_date = pd.to_datetime(short_df['Date'].iloc[0], infer_datetime_format=True)
    start_date: datetime = start_date.to_pydatetime()
    eod_df = eod(start_date, sym, exc=exc, token=API_KEY)
    short_df = short_df.set_index("Date")
    short_df = short_df.join(eod_df["Volume"], lsuffix="_eod")
    short_df["%short"] = short_df["Volume"] / short_df["Volume_eod"]
    df = eod_df.join(short_df, on="Date", lsuffix="_short").reset_index()
    df.ffill(inplace=True)
    df.drop(["Volume_eod"], axis=1, inplace=True)
    df.to_csv(file)
    return df


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-pk", "--proxy-key", type=str, help="proxy API key used for proxy")
    parser.add_argument("-d", "--download-dir", type=str, help="download path", default=Path(os.getcwd() + "/data"))
    parser.add_argument("-w", "--workers", type=int, help="number of workers", default=10)
    parser.add_argument("-l", "--rate-limit", type=float,
                        help="rate limiting. Seconds to wait between loading google news scores", default=0.1)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    PROXY_API_KEY = args.api_key
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

    data_dir = Path(args.path)

    for s in symbols:
        path = data_dir / f"chart_eodhistoric_{s}.csv"
        save_with_short(s, "US", API_KEY, path)
        print(f"Saved {s} data to {path}")
