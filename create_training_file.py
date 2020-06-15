import os
import pandas as pd
import numpy as np
from talib import abstract

SMA = abstract.SMA

if __name__ == '__main__':

    input_file = os.path.join(os.getcwd(), "data", "QQQ_5y_raw.csv")
    output_file = os.path.join(os.getcwd(), "data", "QQQ_5y_train.csv")

    # load the file
    df_raw = pd.read_csv(input_file, header=0)

    # apply indicators the the dataframe
    df_raw["SMA50"] = pd.Series(SMA(df_raw["Close"], timeperiod=50))  # type: pd.Series
    df_raw["SMA20"] = pd.Series(SMA(df_raw["Close"], timeperiod=20))  # type: pd.Series

    df_raw = df_raw.reindex(columns=df_raw.columns.tolist() + ['%change'])

    future_periods = 3

    # set the target to be equal to % change at +3 days
    for i, row in df_raw.iterrows():
        future_i = i + future_periods
        if future_i in df_raw.index:
            now = df_raw.iloc[i]["Close"]
            future = df_raw.iloc[future_i]["Close"]
            df_raw.at[i, "%change"] = (future - now) / abs(now) * 100

    # grab all rows that will have valeus
    df_raw = df_raw.loc[np.r_[50:len(df_raw)-3], :].reset_index(drop=True).drop(columns=["Date"])

    headers = []
    original_headers = df_raw.columns.tolist()

    for i in range(0, 7):
        for h in original_headers:
            headers.append(f"{h}-{i}")

    df = pd.DataFrame(columns=headers)

    # create the trasponsed dataframe
    for i, row in df_raw.iterrows():
        if i % 7 == 0 and i - 7 in df_raw.index:
            # every 7 rows
            sub_df = df_raw.loc[np.r_[i-7:i], original_headers].reset_index(drop=True)
            row = []
            for j in range(0, 7):
                for h in original_headers:
                    row.append(sub_df.iloc[j][h])

            df = df.append(pd.DataFrame([row], columns=headers))

    df.to_csv(output_file, index=False)

    # chunk the dataframe into 7 day chunks

    # convert each chunk's dates into relative days as ints

    # transform df into rows
