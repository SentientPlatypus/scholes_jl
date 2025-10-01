import sys
import certifi
import warnings
import json
import pandas as pd
from urllib.request import urlopen
import os
import platform
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")

dsp = "/"
if platform.system() == "Windows":
    dsp = "\\"

def get_jsonparsed_data(url):
    response = urlopen(url, cafile=certifi.where())
    data = response.read().decode("utf-8")
    return json.loads(data)

def read_historical(tickers:list, apikey):
    df = pd.DataFrame()
    for ticker in tickers:
        # Build the URL for downloading historical data from 2000-01-01
        url = "https://financialmodelingprep.com/api/v3/historical-price-full"
        if ticker.endswith("=X"):
            url += "/{}?apikey={}&from=2000-01-01".format("USD" + ticker[:-2], apikey)
        else:
            url += "/{}?apikey={}&from=2000-01-01".format(ticker, apikey)
        
        json_data = get_jsonparsed_data(url)
        
        # Convert the "historical" data into a DataFrame
        hist_df = pd.DataFrame(json_data["historical"])
        # Sort in ascending order so that pct_change calculates correctly
        hist_df.sort_values("date", inplace=True)
        
        # Compute changeClosePercent as the percentage change in close from the previous day
        # This is calculated as: (today_close - previous_close) / previous_close * 100
        hist_df["changeClosePercent"] = hist_df["close"].pct_change() * 100
        
        # If you prefer to have the rows in descending order (most recent first),
        # you can reverse the DataFrame:
        hist_df = hist_df.iloc[::-1]
        
        # Write the modified DataFrame (which now includes the changeClosePercent column) to CSV.
        hist_df.to_csv(f".{dsp}data{dsp}{ticker}.csv", index=False)
        
        # Merge historical data of all tickers into one CSV file (using adjClose)
        hist = hist_df[["date", "adjClose"]].copy()
        hist = hist.rename(columns={"adjClose": ticker})
        if df.empty:
            df = hist
        else:
            df = df.merge(hist, on="date")
        
    df = df.drop(columns=["date"])


if __name__ == "__main__":
    tickers = sys.argv[1:]
    apikey = open(f".{dsp}apikey", "r").readline().strip()

    read_historical(tickers, apikey)