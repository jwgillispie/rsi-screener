import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import ta 
import os
import yfinance as yf
import datetime 
import warnings
import matplotlib.dates as mdates
import requests
import schedule
import time 
from dotenv import load_dotenv
load_dotenv()
warnings.filterwarnings("ignore")

#  get the data
def getData(sector_list):
    stocks = pd.read_csv("./nasdaq_screener.csv")
    stocks = stocks.sort_values("Market Cap", ascending= False)
    stocks = stocks.loc[stocks["Sector"].isin(sector_list)]
    stocks = stocks.sort_values("Market Cap", ascending=False)
    stocks = stocks.head(200)
    return stocks

stocks = getData(["Technology", "Industrials", "Biotechnology", "Finance"])


# go thru the ticker list and get stock data from the records
# i can use yfinance to make a chart for each of the stocks 
# should try the entire process with a single stock first 
ticker_list = stocks.Symbol 
ticker_list[:5]
# get data from 2020 to yesterday
today = datetime.date.today()
yesterday = today - datetime.timedelta(1)
stock_dict = {}
df_index = 0

for i in ticker_list:
    current = yf.Ticker(i)
    history = current.history(period= "1y").reset_index()[['Date','Close']]

    # really i should calculate the rsi here and then add it to the dataframe
    rsi = ta.momentum.RSIIndicator(history["Close"], window=7)
    # tsi = ta.momentum.tsi(history["Close"])
    # i need to get the df
    history["RSI"] = rsi.rsi()
    # history["TSI"] = tsi.tsi()
    history = history.dropna(subset=['RSI'])
    df_index += 1

    stock_dict[i] = history

# now i have a disctionary where each ticker is mapped to a df of its history

def evaluateRSILow(ticker_to_df, ticker_list, message_dict, rsi_threshold=30, rsi_average_range=25):
    for i in ticker_list:
        current_rsi_list = ticker_to_df[i].tail(rsi_average_range)["RSI"]
        current_df = ticker_to_df[i]
        today_close = current_df.iloc[-1]['Close']
        rsi_average = np.mean(current_rsi_list)
    
        bb = ta.volatility.BollingerBands(current_df['Close'], window=25, window_dev=2)
        # boll_hband = bb.bollinger_hband()


        if rsi_average < rsi_threshold:
            
            # print("Today's close for " + i + " is: " + "{:.2f}".format(today_close))
            # print("Average RSI of previous {days} days: ".format(days=rsi_average_range), str(round(rsi_average, 2)))
 

            # print("\nConsider buying: " + i)
            message_dict[i] = "-" + i + " "


def evaluateRSIHigh(ticker_to_df, ticker_list, message_dict, rsi_threshold=70, rsi_average_range=25):
    count = 0
    for i in ticker_list:
        current_rsi_list = ticker_to_df[i].tail(rsi_average_range)["RSI"]
        current_df = ticker_to_df[i]
        today_close = current_df.iloc[-1]['Close']
        rsi_average = np.mean(current_rsi_list)

        if rsi_average > rsi_threshold:
            
            # print("Today's close for " + i + " is: " + "{:.2f}".format(today_close))
            # print("Average RSI of previous {days} days: ".format(days=rsi_average_range), str(round(rsi_average, 2)))
            # print("\nConsider buying: " + i)

            # print("")
            message_dict[i] = "-" + i + " "


def run():
    num = os.getenv('PHONE')
    stocks = getData(["Technology", "Industrials", "Biotechnology", "Finance"])


    # go thru the ticker list and get stock data from the records
    # i can use yfinance to make a chart for each of the stocks 
    # should try the entire process with a single stock first 
    ticker_list = stocks.Symbol 
    ticker_list[:5]
    # get data from 2020 to yesterday
    today = datetime.date.today()
    yesterday = today - datetime.timedelta(1)
    stock_dict = {}
    df_index = 0

    for i in ticker_list:
        current = yf.Ticker(i)
        history = current.history(period= "1y").reset_index()[['Date','Close']]

        # really i should calculate the rsi here and then add it to the dataframe
        rsi = ta.momentum.RSIIndicator(history["Close"], window=7)
        # tsi = ta.momentum.tsi(history["Close"])
        # i need to get the df
        history["RSI"] = rsi.rsi()
        # history["TSI"] = tsi.tsi()
        history = history.dropna(subset=['RSI'])
        df_index += 1

        stock_dict[i] = history

    # now i have a disctionary where each ticker is mapped to a df of its history

    calls_message_dict = {}
    evaluateRSILow(stock_dict, ticker_list, calls_message_dict)

    puts_message_dict = {}
    evaluateRSIHigh(stock_dict,  ticker_list, puts_message_dict)

    # now we should test this function with some previous data 
    # i should get another dictionary withticker to dataframe structure 
    # and then i should run the analysis on that dictionary 
    # and then i can see what other info may be useful and what i can get from it 
    puts_message = "Puts:"

    for i in puts_message_dict:
        puts_message += puts_message_dict[i]
    # print(puts_message)

    calls_message = "Calls:"
    for i in calls_message_dict:
        calls_message += calls_message_dict[i]
    # print(calls_message)

    message = puts_message + calls_message
    # print(message)

    resp = requests.post('https://textbelt.com/text', {
    'phone': num,
    'message': message,
    'key': 'textbelt',
    })
    print(resp.json())
# schedule.every().saturday.at("22:48", "America/New_York").do(run)
schedule.every().sunday.at("23:00", "America/New_York").do(run)
schedule.every().wednesday.at("10:00", "America/New_York").do(run)

while True:
    schedule.run_pending()
    time.sleep(1)
