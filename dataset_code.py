#!/usr/bin/python
def get_data_coin():
    import ccxt
    import calendar
    from datetime import datetime
    from dateutil.relativedelta import relativedelta
    import pandas as pd
    import json

    exchange = ccxt.binanceusdm({
        "apiKey": "xBPLNbrLuBqVmYriXB2lVFWa7XEPfUIOyo1Sjvft21SmfZMRUxDz2BcXNFGXGxOw",
        "secret": "W4Pv0VODKY6eT4p4wW7QwW30yXm2ziu7IYQH19M2U9NkmOH7ZBscNi4yLUhAlSFr",
        'options': {'adjustForTimeDifference': True}
    })

    input_coin = json.load(open('coin_file.json','r'))
    coin = input_coin['coin']
    print(f'CURRENT COIN IS {coin}')

    now = datetime.utcnow()
    data_final = pd.DataFrame()
    for i in range(1,15):
        
        now = datetime.utcnow() + relativedelta(days=-30*(i-1))
        unixtime = calendar.timegm(now.utctimetuple())
        print(now)
        
        since = (unixtime - 60*60*1000) * 1000 # UTC timestamp in milliseconds

        ohlcv = ccxt.binance.fetch_ohlcv(exchange,symbol=coin, timeframe='1h', since=since, limit=1000)
        #start_dt = datetime.fromtimestamp(ohlcv[0][0]/1000)
        #end_dt = datetime.fromtimestamp(ohlcv[-1][0]/1000)

        #convert it into Pandas DataFrame
        import pandas as pd

        df = pd.DataFrame(ohlcv, columns = ['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        #df['Time'] = [datetime.fromtimestamp(float(time)/1000) for time in df['timestamp']]
        #df.set_index('Time', inplace=True)
        #df.drop(columns='Time', axis='columns', inplace=True)
        
        data_final = pd.concat([data_final,df],axis = 0)
        
        

    data_final = data_final.drop_duplicates(subset="timestamp")
    data_final = data_final.sort_values(by='timestamp', ascending=True)
    #data_final['Datetime'] = datetime.fromtimestamp(data_final['timestamp']).isoformat()
    data_final['timestamp_'] = data_final['timestamp']
    data_final['timestamp'] = pd.to_datetime(data_final['timestamp'],unit='ms')
    
    data_final['timestamp'] = data_final['timestamp'].dt.strftime('%d-%m-%Y %H:%M')
    data_final.index = data_final['timestamp']
    #data_final.index = pd.DatetimeIndex(data_final['timestamp'])
    data_final.drop(columns=['timestamp'], inplace=True)
    #data_final.to_csv(f'{coin[:3]}_dataset.csv')
    
    
    data_final.to_csv(f'{coin[:3]}_dataset.csv')
    #data_final = data_final.iloc[407:]
    #print(data_final.head())
    #data_final.to_csv(f"C:\Users\Admin\OneDrive\Desktop\Backtest\{coin}_dataset.csv")
    
    return data_final

