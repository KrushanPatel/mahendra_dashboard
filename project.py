import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
from PIL import Image
from helper import *
from backtest_eth_ema_v3_live_krus import backtest


######################## Set-Up PAGE #####################################
st.set_page_config( page_title= "Binance Exchange", page_icon="🧊",
                    menu_items={
                                'Get Help': 'https://www.extremelycoolapp.com/help',
                                'Report a bug': "https://www.extremelycoolapp.com/bug",
                                'About': "# This is a header. This is an *extremely* cool app!"
                                }
)
menu = ["None","Live Market", "Backtest", "Indicators"]
coin = ["BTCUSDT","ETHUSDT","BNBUSDT"]
timeframe = ["1m","5m","30m","1h","1d"]
menu = st.sidebar.selectbox("Select the Options",menu)


place_holder = st.empty()

if menu == "Live Market":

    place_holder.container()
    coin = st.sidebar.selectbox("Search Coin",coin)
    st.write(f"{coin}")
    timeframe = st.sidebar.selectbox("Choose Timeframe",timeframe)

    client = exchange()
    #st.write(client)
    while True:

        with place_holder.container():

            _ltp = client.fetch_ticker(str(coin))["last"]
            order_book = client.fetch_order_book(coin, limit=5)
            bids = order_book['bids'][0][0]
            asks = order_book['asks'][0][0]
            #st.write("Hello")
            data = client.fetch_ohlcv(symbol=str(coin), timeframe=str(timeframe), limit=None, params={})
            data = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
            data["datetime"] = pd.to_datetime(data["timestamp"], unit="ms")

            ltp,bid,ask = st.columns(3)
            ltp.metric(label=str(coin), value=_ltp,delta=str(round(_ltp - data["close"].iloc[-2], 2)))

            bid.metric(label="Bids", value=bids, delta=None)
            ask.metric(label="Asks", value=asks, delta=None)


            data = data.iloc[-100:,::]
            #st.dataframe(data)
            fig = go.Figure(data=[go.Candlestick(x=data["datetime"],
                                                 open=data["open"],
                                                 high=data["high"],
                                                 low=data["low"],
                                                 close=data["close"],
                                                 name=coin)])
            # fig.update_xaxes(type = "TimeStamp")
            # fig.update_layout(height = 60)
            st.plotly_chart(fig, use_container_width=True)
            # st.write(data)
            time.sleep(3)

if menu == "Backtest":

    coin = ["None","BTC/USDT", "ETH/USDT", "BNB/USDT"]
    strategys = ["None","SMA-Crossover","EMA-Crossover","BB-Bands","AO-EMA","RSI","MACD"]
    strategy = st.sidebar.selectbox("Select Strategy",strategys)
    if strategy == "EMA-Crossover":
        st.spinner(text="Wait for it...")

        coin = st.sidebar.selectbox("Search Coin",coin)
        ema_short = st.sidebar.number_input("Enter Short Value in Interger", min_value=1)
        ema_long = st.sidebar.number_input("Enter long Value",min_value=ema_short+1)
        coin_file = json.load(open("coin_file.json"))
        coin_file["coin"] = coin

        with open("coin_file.json", "w") as outfile:
            json.dump(coin_file, outfile)

        input_para = json.load(open("input_para.json"))
        input_para["ema_short"] = ema_short
        input_para["ema_long"] = ema_long

        with open("input_para.json", "w") as outfile:
            json.dump(input_para, outfile)
        if st.button("Start"):
            backtest()
        else:
            None
        st.image(Image.open("Trade_P&L.png"))










