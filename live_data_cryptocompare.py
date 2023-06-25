import requests
import pandas as pd
from json import JSONDecoder
import datetime
import json
import time
import math
import random 
import numpy as np

# URL = "https://min-api.cryptocompare.com/data/histohour?fsym=DOT&tsym=USD&limit=24&toTs=1598058000"

def extract_json_objects(text, decoder=JSONDecoder()):
    pos = 0
    while True:
        match = text.find('{', pos)
        if match == -1:
            break
        try:
            result, index = decoder.raw_decode(text[match:])
            yield result
            pos = match + index
        except ValueError:
            pos = match + 1

def get_min_data(start_time,coin):
    # coin = "BTC"
    # time_ = start_time = current_time = time.time() - 120
    end_timestamp   = time.time()
    time_  = start_time
    # coin_data = []
    dot_data = pd.DataFrame()
    time_index = pd.DataFrame()
    while time_ <= end_timestamp:
        try:
            
            print(time_)
            URL = f"https://min-api.cryptocompare.com/data/v2/histominute?fsym={coin}&tsym=USDT&limit=1&toTs={time_}"
            time_ = time_ + 60
            data = requests.get(URL)
            data = extract_json_objects(data.text)
            data = list(data)
            hourly_data = []
            timestamp = []
            for i in range(len(data[0]["Data"]['Data'])):
                hourly_data.append(data[0]["Data"]['Data'][i]["close"])
                timestamp.append(data[0]["Data"]['Data'][i]["time"])
            # coin_data.append(hourly_data)
            dot_data = pd.concat([dot_data,pd.DataFrame(hourly_data)], axis = 0)
            time_index = pd.concat([time_index,pd.DataFrame(timestamp)], axis = 0)
            time.sleep(random.uniform(0, 1))
        except Exception as error:
            print("connection issue")
            time.sleep(random.uniform(0, 1))
            time_ = time_ - 86400
    dot_data.index = pd.to_datetime(time_index.iloc[:,0], unit='s')
    dot_data.rename(columns = {0:"Close"}, inplace = True)
    dot_data.index.name = "Date"


    data = dot_data.tail(1)
    data.reset_index(inplace = True)
    data.columns = ['timestamp','Close']
    data_temp = pd.DataFrame(columns = ['timestamp','Open','High','Low','Close'])
    data      = pd.concat([data_temp,data])
    data[['Open','High','Low']] = 0
    data['timestamp'] = [i.strftime('%d-%m-%Y %H:%M') for i in data['timestamp']]
    data_old = pd.read_csv(f'{coin}_dataset.csv')
    data_final = pd.concat([data_old,data.tail(1)])
    data_final.drop_duplicates(subset = 'timestamp',inplace = True)
    data_final = data_final[['timestamp','Open','High','Low','Close']]
    # data_final.to_csv("Btc_1hr1.csv",index = False)
    
    return data_final

def get_data(start_time,coin):
    # coin = "BTC"
    # start_time = current_time = time.time() - 86400*5
    end_timestamp   = time.time()
    time_  = start_time
    # coin_data = []
    dot_data = pd.DataFrame()
    time_index = pd.DataFrame()
    while time_ <= end_timestamp:
        try:
            print(time_)
            URL1 = f"https://min-api.cryptocompare.com/data/histohour?fsym={coin}&tsym=USDT&limit=12&toTs={time_}"
            time_ = time_ + 86400
            data = requests.get(URL1)
            data = extract_json_objects(data.text)
            data = list(data)
            hourly_data = []
            timestamp = []
            for i in range(len(data[0]["Data"])):
                hourly_data.append(data[0]["Data"][i]["close"])
                timestamp.append(data[0]["Data"][i]["time"])
            # coin_data.append(hourly_data)
            dot_data = pd.concat([dot_data,pd.DataFrame(hourly_data)], axis = 0)
            time_index = pd.concat([time_index,pd.DataFrame(timestamp)], axis = 0)
            time.sleep(random.uniform(0, 1))
        except Exception as error:
            print("connection issue")
            time.sleep(random.uniform(0, 1))
            time_ = time_ - 86400
    dot_data.index = pd.to_datetime(time_index.iloc[:,0], unit='s')
    dot_data.rename(columns = {0:"Close"}, inplace = True)
    dot_data.index.name = "Date"
    # dot_data.to_csv(f"{coin}_1_hr_data_{time.time()}.csv")


    # current_time = time.time() - 86400*5
    # coin = "BTC"
    data = dot_data
    data.reset_index(inplace = True)
    data.columns = ['timestamp','Close']
    data_temp = pd.DataFrame(columns = ['timestamp','Open','High','Low','Close'])
    data      = pd.concat([data_temp,data])
    data[['Open','High','Low']] = 0
    data['timestamp'] = [i.strftime('%d-%m-%Y %H:%M') for i in data['timestamp']]
    data_old = pd.read_csv(f'{coin}_dataset.csv')
    data_final = pd.concat([data_old,data.tail(100)])
    data_final.drop_duplicates(subset = 'timestamp',inplace = True)
    data_final = data_final[['timestamp','Open','High','Low','Close']]
    data_final.to_csv(f'{coin}_dataset.csv',index = False)
    
    return data_final

"""
import websockets
import json
import asyncio

class WSClient():
    def __init__(self):
        self.api_key = "s59Zm69k"
        self.api_secret = "QN3FYdBUFAdgJGNTrHfC8glEDwVJYC368D-NaZWnnXI"
        self.msg = { 
            "jsonrpc": "2.0",
            "id": 0
        }
        self.host = "wss://www.deribit.com/ws/api/v2"

    async def call_api(self, msg):   
        async with websockets.connect(self.host) as websocket:
            # print("Connected to URL:", self.host)
            try:
                await websocket.send(msg)
                while websocket.open:
                    response = await websocket.recv()
                    response_json = json.loads(response)
                    return response_json
            except Exception as e:
                return e

    def request(self, method, params, session=None):
        msg = self.msg
        msg["id"] += 1
        msg["method"] = method
        msg["params"] = params
        if session != None:
            msg["params"]["scope": "session:{}".format(session)]
        return asyncio.get_event_loop().run_until_complete(self.call_api(json.dumps(msg)))

    def get_order_book(self, instrument):
        method = "public/get_order_book"
        params = {
            "instrument_name": instrument
        }
        return self.request(method, params)
    
    def ticker(self, instrument_name):
        params = {"instrument_name" : instrument_name}
        method = "public/ticker"
        return self.request(method, params)
    
    def chart_data(self, instrument_name, start_tmpstmp, end_tmpstmp, freq):
        params = {
            "instrument_name" : instrument_name,
            "start_timestamp" : start_tmpstmp,
            "end_timestamp" : end_tmpstmp,
            "resolution" : freq
            }
        method = "public/get_tradingview_chart_data"
        
        return self.request(method, params)

def get_deribit_data():

    client = WSClient()    
    INTERVAL = 60
    end_tm=time.time()
    strt_tm=end_tm-60*INTERVAL*5 # Why multiplied by 191, because we are tracing last 191 days
    get_data = client.chart_data("BTC-PERPETUAL", int(strt_tm*1000), int(end_tm*1000), INTERVAL)
    df=pd.DataFrame(get_data['result'])  
    df['timestamp'] = [datetime.datetime.fromtimestamp(i/1000) for i in df['ticks']]
    df_new = df[['timestamp','open','high','low','close']]
    df_new.rename(columns = {'open':'Open','high':'High','low':"Low",'close':'Close'}, inplace = True)

    df_new['timestamp'] = [i.strftime('%d-%m-%Y %H:%M') for i in df_new['timestamp']]
    data_old = pd.read_csv("Btc_1hr1.csv")
    data_final = pd.concat([data_old,df_new.tail(100)])
    data_final.drop_duplicates(subset = 'timestamp',inplace = True)
    data_final = data_final[['timestamp','Open','High','Low','Close']]
    data_final.to_csv("Btc_1hr1.csv",index = False)

    return data_final


def get_deribit_data_min():

    client = WSClient()    
    INTERVAL = 1
    end_tm=time.time()
    strt_tm=end_tm-60*INTERVAL*5 # Why multiplied by 191, because we are tracing last 191 days
    get_data = client.chart_data("BTC-PERPETUAL", int(strt_tm*1000), int(end_tm*1000), INTERVAL )
    df=pd.DataFrame(get_data['result'])  
    df['timestamp'] = [datetime.datetime.fromtimestamp(i/1000) for i in df['ticks']]
    df_new = df[['timestamp','open','high','low','close']]
    df_new.rename(columns = {'open':'Open','high':'High','low':"Low",'close':'Close'}, inplace = True)

    df_new['timestamp'] = [i.strftime('%d-%m-%Y %H:%M') for i in df_new['timestamp']]
    data_old = pd.read_csv("Btc_1hr1.csv")
    data_final = pd.concat([data_old,df_new.tail(100)])
    data_final.drop_duplicates(subset = 'timestamp',inplace = True)
    data_final = data_final[['timestamp','Open','High','Low','Close']]
    # data_final.to_csv("Btc_1hr1.csv",index = False)
    return data_final


#!/usr/bin/python
# import ccxt
# import calendar
# from datetime import datetime

# binance = ccxt.binance()

# now = datetime.utcnow()
# unixtime = calendar.timegm(now.utctimetuple())
# since = (unixtime - 60*60*1000) * 1000 # UTC timestamp in milliseconds

# ohlcv = binance.fetch_ohlcv(symbol='BTCUSDT', timeframe='1h', since=since, limit=10000)
# # start_dt = datetime.fromtimestamp(ohlcv[0][0]/1000)
# # end_dt = datetime.fromtimestamp(ohlcv[-1][0]/1000)

# # convert it into Pandas DataFrame
# import pandas as pd

# df = pd.DataFrame(ohlcv, columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
# df['Time'] = [datetime.fromtimestamp(float(time)/1000) for time in df['Time']]
# df.set_index('Time', inplace=True)

"""