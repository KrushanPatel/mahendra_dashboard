def backtest():
    from live_data_cryptocompare import get_min_data
    import os
    #import pdb
    import datetime as dt
    import numpy as np
    import pandas as pd
    import time
    import json
    from dataset_code import get_data_coin
    from bktest_config import config
    import ccxt
    #from ta.momentum import AwesomeOscillatorIndicator
    from  collections.abc import Mapping
    global client
    exchange = ccxt.binanceusdm({
        "apiKey": "xBPLNbrLuBqVmYriXB2lVFWa7XEPfUIOyo1Sjvft21SmfZMRUxDz2BcXNFGXGxOw",
        "secret": "W4Pv0VODKY6eT4p4wW7QwW30yXm2ziu7IYQH19M2U9NkmOH7ZBscNi4yLUhAlSFr",
        'options': {'adjustForTimeDifference': True}
    })
    client = exchange
    
    input_coin = json.load(open('coin_file.json','r'))
    coin = input_coin['coin'][:3]
    print(coin)


    global sl_perc, pnl, vol_threshold
    global trail_perc, trail_price, sl_val, path, file_name
    global equity, leverage, qty, entry_pr, curr_pos, ema_sig
    global vol_window, mul_std, val_sqrt, trade_details, max_equity
    global freq_fac
    global long_period, short_period,ao_win_long,ao_win_short
    import glob
    import warnings

   
    data_final = get_data_coin()
    #data_final = pd.read_csv(r'C:\Users\Admin\OneDrive\Desktop\Krushan_Crypto\python code\BackTesting\ETH_dataset.csv')
    warnings.filterwarnings("ignore")

    path = os.path.abspath(os.getcwd())

    pnl=0;vol_threshold=config['vol_threshold']
    input_para = json.load(open("input_para.json"))

    short_period=input_para['ema_short']; long_period=input_para['ema_long']
    #ao_win_short = config["ao_win_short"]
    #ao_win_long = config["ao_win_long"]
    trail_perc = config['trail_perc']; file_name = config['file_name']
    equity = config['ini_capital']; leverage = config['leverage']; qty = config['qty']
    freq_fac = config['freq_factor']
    vol_window= config['vol_window']; mul_std= config['multiply_std']; val_sqrt= config['value_sqrt']
    
    entry_pr=0; curr_pos=0; ema_sig = None; trail_price=0; sl_val=0
    max_equity = equity

    trade_details = {
        'Date':[],
        'Close':[],
        'Trade':[],
        'Entry_pr':[],
        'Exit_pr':[],
        
        'Open_Pos':[],
        'PnL':[],
        'PnL_Perc':[],
        'Equity':[],
        'Drawdown_Perc':[]
        }

    # trade_stats


    def calc_vol(df):
        global vol_window, mul_std, val_sqrt, freq_fac

        # df1['Close']=pd.to_numeric(df['Close'], downcast='float')
        df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Volatility'] = mul_std*df['Log_Ret'].rolling(window=vol_window).std() * np.sqrt(freq_fac)* np.sqrt(val_sqrt)

        return df

    def fill_total_trades(dt_pt,trade_pnl, trade_qty,trade,entry_pr, exit_pr, open_qty,main_eq):
        global max_equity, equity

        trade_details['PnL'].append(trade_pnl)       
        trade_details['PnL_Perc'].append(round(trade_pnl/(10000)*100,2))
        trade_details['Date'].append(dt_pt['date'])
        trade_details['Close'].append(dt_pt['close'])
        trade_details['Trade'].append( trade ) 
        trade_details['Entry_pr'].append(entry_pr)
        trade_details['Exit_pr'].append(exit_pr)
        trade_details['Open_Pos'].append(open_qty)
        # trade_details['Traded_Qty'].append(trade_qty)
        
        trade_details['Equity'].append(equity)

        if trade_details['Equity'][-1] >= max_equity:
            max_equity = trade_details['Equity'][-1]
        
        _dd = round( (trade_details['Equity'][-1]-max_equity)/max_equity*100,2 )
        trade_details['Drawdown_Perc'].append(_dd)


    def gen_trades(dt_pt,sl_perc, trail ):
        global  pnl, vol_threshold, short_period, long_period
        global trail_perc, trail_price, path, file_name, sl_val
        global equity, leverage, qty, entry_pr, curr_pos

        if ema_sig=='long' and curr_pos <=0:# and dt_pt['vol'] < vol_threshold:
            
            if curr_pos < 0:
               
                pnl = (entry_pr - dt_pt['close'])/ entry_pr* abs(curr_pos)
                equity += pnl
                fill_total_trades(dt_pt,pnl, curr_pos,'Exit_Short',entry_pr, dt_pt['close'], 0,equity)
                curr_pos = 0; entry_pr=0

            if curr_pos == 0:
                pnl = 0
                entry_pr = dt_pt['close']

                sl_val = round(entry_pr * (1 - (sl_perc)/100),4)

                if qty == 0 :
                    curr_pos = round((equity * 0.95)*(leverage)/entry_pr,0)
                else:
                    curr_pos = qty

                equity += pnl
                fill_total_trades(dt_pt,pnl, curr_pos,'New_Long',entry_pr, 0, curr_pos,equity)
            
        if ema_sig=='short' and curr_pos >=0 :#and dt_pt['vol'] < vol_threshold:
            if curr_pos > 0:
               
                pnl = ( dt_pt['close'] - entry_pr)/entry_pr * abs(curr_pos)
                equity += pnl
                fill_total_trades(dt_pt,pnl, curr_pos,'Exit_Long',entry_pr, dt_pt['close'], 0,equity)
                curr_pos = 0; entry_pr=0
                
            if curr_pos == 0:
                pnl = 0
                entry_pr = dt_pt['close']
                sl_val = round(entry_pr * (1 + (sl_perc)/100),4)
                if qty == 0 :
                    curr_pos = -1 * round((equity * 0.95)*(leverage)/entry_pr,0)
                else:
                    curr_pos = -1 * qty
                equity += pnl
                fill_total_trades(dt_pt,pnl, curr_pos,'New_Short',entry_pr, 0, curr_pos,equity)
   
        monitor_sl(dt_pt,sl_perc, trail = trail)
        
        #print(pnl)
        """############################# Profit Booking ################################
        
        if curr_pos > 0 and round(100*pnl/equity,2) > 0.5:
            #print("profit Booked")
            
            if curr_pos >= 5000:
                   
                pnl = (dt_pt['close'] - entry_pr)/entry_pr * abs(curr_pos)
                equity += pnl
                fill_total_trades(dt_pt,pnl, curr_pos,'Pos_Closed',entry_pr, dt_pt['close'], 0,equity)
                curr_pos = 0; entry_pr=0
                
            
            if curr_pos >= 10000:
                   
                pnl = ( dt_pt['close'] - entry_pr)/entry_pr * abs(curr_pos)
                equity += pnl*(0.5)
                fill_total_trades(dt_pt,pnl, curr_pos,'Profit_Booked',entry_pr, dt_pt['close'], 0,equity)
                curr_pos = 5000; entry_pr=entry_pr
                                
            
        if curr_pos < 0 and round(100*pnl/equity,2) > 0.5:
            #print("profit Booked")
            if curr_pos <= -10000:
                   
                pnl = ( dt_pt['close'] - entry_pr)/entry_pr * abs(curr_pos)
                equity += pnl*0.5
                fill_total_trades(dt_pt,pnl, curr_pos,'Profit_Booked',entry_pr, dt_pt['close'], 0,equity)
                curr_pos = -5000; entry_pr=entry_pr
                
            if curr_pos <= -5000:
                   
                pnl = ( dt_pt['close'] - entry_pr)/entry_pr * abs(curr_pos)
                equity += pnl
                fill_total_trades(dt_pt,pnl, curr_pos,'Pos_Closed',entry_pr, dt_pt['close'], 0,equity)
                curr_pos = 0; entry_pr=0
          
          
                
            #####################################################################################################
"""


    def monitor_sl(dt_pt,sl_perc, trail = False):
        global  pnl, vol_threshold, short_period, long_period
        global trail_perc, trail_price, path, file_name, sl_val
        global equity, leverage, qty, entry_pr, curr_pos, ema_sig

        if ema_sig=='long' and dt_pt['close'] < sl_val:
            if abs(( dt_pt['close'] - entry_pr)/entry_pr)*100 > sl_perc:
                # print("Correct")
                pnl = -sl_perc*100
                equity += pnl
                fill_total_trades(dt_pt,pnl, curr_pos,'SL_Long',entry_pr, dt_pt['close'], 0,equity)
                curr_pos = 0; entry_pr=0; ema_sig = None
            else:

                pnl = ( dt_pt['close'] - entry_pr)/entry_pr * abs(curr_pos)
                equity += pnl
                fill_total_trades(dt_pt,pnl, curr_pos,'SL_Long',entry_pr, dt_pt['close'], 0,equity)
                curr_pos = 0; entry_pr=0; ema_sig = None
                
        if ema_sig=='short' and dt_pt['close'] > sl_val:
            
            if abs(( entry_pr - dt_pt['close'])/entry_pr)*100 > sl_perc:
                #print("Correct")
                pnl = -sl_perc*100
                equity += pnl
                fill_total_trades(dt_pt,pnl, curr_pos,'SL_Short',entry_pr, dt_pt['close'], 0,equity)
                curr_pos = 0; entry_pr=0; ema_sig = None
            else:
                pnl = ( entry_pr - dt_pt['close'])/entry_pr * abs(curr_pos)
                equity += pnl
                fill_total_trades(dt_pt,pnl, curr_pos,'SL_Short',entry_pr, dt_pt['close'], 0,equity)
                curr_pos = 0; entry_pr=0; ema_sig = None
                
                


        if ema_sig == 'long' and trail == True:
            if trail_price == 0 and dt_pt['close'] > round((1+trail_perc/100) * entry_pr,0): 
                sl_val=round((1-sl_perc/100) * dt_pt['close'],4) 
                trail_price = dt_pt['close'] 
            if trail_price != 0 and dt_pt['close'] > round((1+trail_perc/100) * trail_price,0): 
                sl_val=round((1-sl_perc/100) * dt_pt['close'],4) 
                trail_price = dt_pt['close'] 

        if ema_sig == 'short'and trail == True:
            if trail_price == 0 and dt_pt['close'] < round((1-trail_perc/100) * entry_pr,0): 
                sl_val=round((1+sl_perc/100) * dt_pt['close'],4) 
                trail_price = dt_pt['close'] 
            if trail_price != 0 and dt_pt['close'] < round((1-trail_perc/100) * trail_price,0): 
                sl_val=round((1+sl_perc/100) * dt_pt['close'],4) 
                trail_price = dt_pt['close']


    def publish_data(file_name,sl_perc, trail):
        global short_period, long_period, path, ema_sig
        global trade_details

        df= pd.read_csv(file_name)
        df= df.rename(columns = {'timestamp.1':'Unix Timestamp','timestamp':'Date'})

        df=df.reset_index(drop=True)

        # pdb.set_trace()
        df = calc_vol(df)

        prev_short_ema=0; curr_short_ema=0
        prev_long_ema=0; curr_long_ema=0

        """
        ################################ Awesome Indicator ########################################
        
        df_ao = AwesomeOscillatorIndicator(high=df["High"],low=df["Low"],window1=ao_win_short,window2=ao_win_long)
        df_ao = df_ao.awesome_oscillator()

        df = pd.concat([df,df_ao],axis=1)
        df.dropna()
        ############################## Awesome Indiactor Ends ####################################
        """
        for k, val in enumerate(df['Close']):
            prev_short_ema = curr_short_ema
            prev_long_ema = curr_long_ema

            if k==short_period:
                curr_short_ema = sum(df['Close'][-k:])/short_period
            if k>short_period:
                curr_short_ema = ((2/(short_period+1)) * val) + ((1-(2/(short_period+1))) * prev_short_ema) # ema(df,EMATFSIGNAL)
            if k==long_period:
                curr_long_ema = sum(df['Close'][-k:])/long_period
            if k>long_period:
                curr_long_ema = ((2/(long_period+1)) * val) + ((1-(2/(long_period+1))) * prev_long_ema) # ema(df,EMATFSIGNAL)

            """
            if prev_short_ema != 0 or prev_long_ema != 0:
                if curr_short_ema > curr_long_ema and prev_short_ema < prev_long_ema:
                    ema_sig='long'
                    #print(val)
                    #print(k)
                elif curr_short_ema < curr_long_ema and prev_short_ema > prev_long_ema:
                    ema_sig='short'
                    #print(val)
                    #print(k)
                  # else:
                #     ema_sig=None
            """
            
            ##################################### ONLY EMA ####################################################  
            if prev_short_ema != 0 or prev_long_ema != 0:
                if curr_short_ema > curr_long_ema and prev_short_ema < prev_long_ema:
                    ema_sig='long'
                    #print(val)
                    #print(k)
                elif curr_short_ema < curr_long_ema and prev_short_ema > prev_long_ema:
                    ema_sig='short'
           
            ##AWS + EMA  Signal Generator ##
            
            #print(df,k,df["ao"][k])
            """
            #####################################AO-EMA####################################################
        
            ao_signal = np.where(float(df["ao"][k]) > 0.00,"long","short")
            
            if prev_short_ema != 0 or prev_long_ema != 0:
                if curr_short_ema > curr_long_ema and prev_short_ema < prev_long_ema and ao_signal == "long":
                    ema_sig='long'
                    #print(val)
                    #print(k)
                elif curr_short_ema < curr_long_ema and prev_short_ema > prev_long_ema and ao_signal == "short":
                    ema_sig='short'
            """
            
            data_point = {
                'date':df['Date'][k],
                'open':df['Open'][k],
                'high':df['High'][k],
                'low':df['Low'][k],
                'close':df['Close'][k],
                'vol':df['Volatility'][k]
                }
            
            gen_trades(data_point,sl_perc, trail)
        return pd.DataFrame(trade_details)
                    

    def gen_trade_stats(df):
        global freq_fac, val_sqrt

        ini_cap = config['ini_capital']
        

        df1=pd.DataFrame(columns=['Metric', 'Value'])
            
        rem_na_col = df['Equity'].dropna()
        max_dd_perc=min(df['Drawdown_Perc'])
        
        eqy_pl=round(df['Equity'][len(df)-1]-ini_cap,2)
        PnL_real=round(max(df['PnL']),2)

        # df[['Day','Time']] = df['Date'].astype(str).str.split(' ',expand=True)  ## KRUS .astype(str) added ##
        df[['Day','Time']] = df['Date'].str.split(' ',expand=True)
        df['Date']=pd.to_datetime(df['Day'])#, format="%d-%m-%Y")
        # # print(df['Date'][len(df)-1])
        num_days = (df['Date'][len(df)-1] - df['Date'][0]).days
        avg_dail_pl = round(eqy_pl/num_days,2)
        avg_wk_pl = round(eqy_pl/(num_days/5),2)
        avg_mon_pl = round(eqy_pl/(num_days/28),2)
        avg_ann_pl = round(eqy_pl/(num_days/342),2)
        avg_tr_pl = round(eqy_pl/len(df),2)

        

        pnl_df=df[df['PnL']!=0]
        pnl_df['score']=np.where(pnl_df['PnL']>0,1,-1)
        pnl_df['streak'] = pnl_df['score'].groupby((pnl_df['score'] != pnl_df['score'].shift()).cumsum()).cumcount() + 1

        pnl_df['pnl_sum'] = pnl_df['PnL'].groupby((pnl_df['score'] != pnl_df['score'].shift()).cumsum()).cumsum()
        pnl_df['pnl_perc_sum'] = pnl_df['PnL_Perc'].groupby((pnl_df['score'] != pnl_df['score'].shift()).cumsum()).cumsum()
        win_df=pnl_df[pnl_df['score']==1]; los_df=pnl_df[pnl_df['score']==-1]

        win_perc = round(len(win_df)/len(pnl_df),2); loss_perc = round(len(los_df)/len(pnl_df),2)
        avg_win = round( sum(win_df['PnL'])/len(win_df),2 ); avg_loss = round( sum(los_df['PnL'])/len(los_df),2 ); 
        expectancy = win_perc * avg_win + loss_perc * avg_loss

        metric_df = df.copy()
        # pdb.set_trace()
        metric_df = metric_df[['Date','PnL_Perc']]


        metric_df = metric_df.groupby('Date', as_index=False).sum()

        sharpe_ratio = metric_df['PnL_Perc'].mean() /(0.00000000000000001 +  metric_df['PnL_Perc'].std()) * np.sqrt(freq_fac) * np.sqrt(val_sqrt)
        
        df2 = metric_df[metric_df['PnL_Perc']<0] ### negative retuns used for sortino ratio

        sortino_ratio = metric_df['PnL_Perc'].mean() /(0.00000000000000001 +  df2['PnL_Perc'].std()) * np.sqrt(freq_fac) * np.sqrt(val_sqrt)

        # 

        # pdb.set_trace()
        df1=df1.append({'Metric':'Equity pl ($)','Value':eqy_pl}, ignore_index=True)
        df1=df1.append({'Metric':'Equity pl (%)','Value':round(eqy_pl/ini_cap*100,2)}, ignore_index=True)
        df1=df1.append({'Metric':'MaxDD (%)','Value':max_dd_perc}, ignore_index=True)
        df1=df1.append({'Metric':'Max PL_Real','Value':PnL_real}, ignore_index=True)
        df1=df1.append({'Metric':'Max PL_Real (%)','Value':round(PnL_real/ini_cap*100,2)}, ignore_index=True)
        
        df1=df1.append({'Metric':'Total Win Trades','Value':len(win_df)}, ignore_index=True)
        df1=df1.append({'Metric':'Win (%)','Value':win_perc}, ignore_index=True)
        # df1=df1.append({'Metric':'Cont.. Win ($)','Value':cons_win_amount}, ignore_index=True)
        df1=df1.append({'Metric':'Avg Win ($)','Value': avg_win }, ignore_index=True)
        
        df1=df1.append({'Metric':'Total Loss Trades','Value':len(los_df)}, ignore_index=True)
        df1=df1.append({'Metric':'Loss (%)','Value':loss_perc}, ignore_index=True)
        # df1=df1.append({'Metric':'Cont.. Loss ($)','Value':cons_los_amount}, ignore_index=True)
        df1=df1.append({'Metric':'Avg Loss ($)','Value': avg_loss}, ignore_index=True)

        df1=df1.append({'Metric':'Expectancy (%)','Value':expectancy}, ignore_index=True)

        df1=df1.append({'Metric':'Sharpe Ratio','Value':round(sharpe_ratio,2)}, ignore_index=True)
        df1=df1.append({'Metric':'Sortino Ratio','Value':round(sortino_ratio,2)}, ignore_index=True)


        return df1

    path = path
    extension = 'csv'
    os.chdir(path)
    result = glob.glob('*.{}'.format(extension))
    sl_perc=config['stop_loss']
    
    #krus
    for file in result:
        if str(file[0:3]) == coin:  #### Krushan Change
            result = [f'{coin}_dataset.csv']
        else:
            pass

    for coin_file in result:
        directory  = coin_file[0:-4].split('_')[0]
        path_coin  = os.path.join(path, directory)
        # files = glob.glob(path_coin+"/*")
        # for f in files:
        #     os.remove(f)
        try: 
            os.mkdir(path_coin)
        except Exception as error:
        # writer_volatility = pd.ExcelWriter(f'{path_coin}\Return_Volatility_{coin_file}.xlsx')
            trails = [False]#[False, True]
            for trail in trails:
                print(path_coin)
                print(coin_file)
                writer_vol = pd.ExcelWriter(f'{path_coin}\Return_Volatility_{coin_file}_{trail}.xlsx')
                writer_backtest_output = pd.ExcelWriter(f'{path_coin}\BackTest_Output_{coin_file}_{trail}.xlsx')
                writer_stats           = pd.ExcelWriter(f'{path_coin}\Output_Stats_{coin_file}_{trail}.xlsx')
                df_vol = []
                df_bt  = []
                df_os  = []
                
                for sl in sl_perc:
                    
                    #print([coin_file, trail,sl])
                    #print(coin_file)
                    df= pd.read_csv(coin_file)
                    df= df.rename(columns = {'timestamp.1':'Unix Timestamp','timestamp':'Date'})
                    # df=df.sort_values('Unix Timestamp')
                    df=df.reset_index(drop=True)
                    df = calc_vol(df)
                    # df.to_excel(writer_vol, sheet_name = f'{directory}_{sl}')
                    df.to_csv(f'{path_coin}\Return_Volatility_{coin_file}_{trail}.csv')
                    main_df = publish_data(coin_file,sl, trail )
                    #print(main_df)
                    # main_df.to_excel(writer_backtest_output, sheet_name = f'{directory}_{sl}')
                    # main_df.to_csv(f'{path_coin}\BackTest_Output_{coin_file}_{sl}_{trail}.csv')
                    strat_stats = gen_trade_stats(main_df)
                    # strat_stats.to_excel(writer_stats, sheet_name = f'{directory}_{sl}')
                    # strat_stats.to_csv(f'{path_coin}\Output_Stats_{coin_file}_{sl}_{trail}.csv')
                    df_bt.append(main_df)
                    df_os.append(strat_stats)
                    # writer_vol.save()
                    # writer_backtest_output.save()
                    # writer_stats.save()
                    
                    trade_details = {
                    'Date':[],
                    'Close':[],
                    'Trade':[],
                    'Entry_pr':[],
                    'Exit_pr':[],
                    
                    'Open_Pos':[],
                    'PnL':[],
                    'PnL_Perc':[],
                    'Equity':[],
                    'Drawdown_Perc':[]
                    }
                   
                    pnl=0;vol_threshold=config['vol_threshold']
                    #short_period=config['ema_short']; long_period=config['ema_long']
                    input_para = json.load(open("input_para.json","r"))
                    short_period=input_para['ema_short']; long_period=input_para['ema_long']
                    #ao_win_short = config["ao_win_short"]
                    #ao_win_long = config["ao_win_long"]
                    trail_perc = config['trail_perc']
                    equity = config['ini_capital']; leverage = config['leverage']; qty = config['qty']
                    freq_fac = config['freq_factor']
                    vol_window= config['vol_window']; mul_std= config['multiply_std']; val_sqrt= config['value_sqrt']
                    entry_pr=0; curr_pos=0; ema_sig = None; trail_price=0; sl_val=0
                    max_equity = equity
                   
    
                for i, dfs in enumerate(df_bt):
                    # print(i)
                    dfs.to_excel(writer_backtest_output, sheet_name = f'{directory}_{sl_perc[i]}')
                writer_backtest_output.save()
                writer_backtest_output.close()
                
                for i, dfs in enumerate(df_os):
                    # print(i)
                    dfs.to_excel(writer_stats, sheet_name = f'{directory}_{sl_perc[i]}')
                writer_stats.save()
                writer_stats.close()


    import numpy as np
    import pandas as pd
    #"C:\Users\Admin\OneDrive\Desktop\Backtest\BNB"
    path = os.getcwd()
    data = pd.read_excel(r"{2}/{0}/BackTest_Output_{1}_dataset.csv_False.xlsx".format(coin,coin,path),sheet_name=f'{coin}_1.8')
    #"C:\Users\Admin\OneDrive\Desktop\Backtest"
    current_trade = data['Trade'].iloc[-1]
    data.drop(columns='Unnamed: 0', axis='columns',inplace=True)
    data.head()


    data = data.append(data.iloc[-1,:],ignore_index = True)
    #print(data['Close'].iloc[-4:])
    # data = pd.DataFrame(data)
    start_time = time.time() - 120
    data_min = get_min_data(start_time,coin)
    # data_final.reset_index(inplace = True)
    data_min.rename(columns = {"timestamp":"Date"}, inplace = True)
    data_min.set_index('Date', drop = True, inplace = True)
    exit_price = data_min.iloc[-1]['Close']
    print(exit_price,data['Trade'].iloc[-2])
    print(data_min.tail())
    ##
    """data_min = client.fetch_ohlcv(symbol=str(f'{coin}/USDT'),timeframe="1h",limit=1000)
    data_min = pd.DataFrame(data_min,columns=["timestamp","open","high","low","Close","volume"])
    data_min['timestamp'] = pd.to_datetime(data_min["timestamp"],unit="ms")
    data_min.rename(columns = {"timestamp":"Date"}, inplace = True)
    data_min.set_index('Date', drop = True, inplace = True)
    exit_price = data_min.iloc[-1]['Close']
    print(exit_price,data['Trade'].iloc[-2])
    print(data_min.tail())"""
    

    
    if data['Trade'].iloc[-2] == 'New_Short':

        data['Trade'].iloc[-1] = "Exit_Short"
        data['Close'].iloc[-1] = exit_price
        data['Entry_pr'].iloc[-1] = 0
        data['Open_Pos'].iloc[-1] = 0
        data['Exit_pr'].iloc[-1] = exit_price
        data['PnL'].iloc[-1] = 10000*(data['Entry_pr'].iloc[-2] - exit_price)/(exit_price)
        data['PnL_Perc'].iloc[-1] = 100*((data['Entry_pr'].iloc[-2] - exit_price)/exit_price)
        data['Equity'].iloc[-1] = data['Equity'].iloc[-2] + data['PnL'].iloc[-1]
        # data.rename(index = {data.index[-1]:data_final.index[-1]},inplace = True)
        # data.index[-1]        = pd.to_datetime(dt.datetime.now().replace(microsecond=0),format='%Y-%m-%d %H:%M:S')
        data['Day'].iloc[-1]  = dt.datetime.fromtimestamp(time.time()).strftime(format ='%d-%m-%Y')
        data['Time'].iloc[-1] = dt.datetime.fromtimestamp(time.time()).strftime(format ='%H:%M')


    elif data['Trade'].iloc[-2] == 'New_Long':
        
        data['Trade'].iloc[-1] = "Exit_Long"
        data['Open_Pos'].iloc[-1] = 0
        data['Close'].iloc[-1] = exit_price
        data['Entry_pr'].iloc[-1] = 0
        data['Exit_pr'].iloc[-1] = exit_price
        data['PnL'].iloc[-1] =10000*(exit_price - data['Entry_pr'].iloc[-2])/(data['Entry_pr'].iloc[-2])
        data['PnL_Perc'].iloc[-1] = 100*(exit_price-data['Entry_pr'].iloc[-2])/(data['Entry_pr'].iloc[-2])
        data['Equity'].iloc[-1] = data['Equity'].iloc[-2] + data['PnL'].iloc[-1]
        # data.index[-1]        = pd.to_datetime(dt.datetime.now().replace(microsecond=0),format='%Y-%m-%d %H:%M:S')
        # data.rename(index = {data.index[-1]:data_final.index[-1]},inplace = True)
        data['Day'].iloc[-1]  = dt.datetime.fromtimestamp(time.time()).strftime(format ='%d-%m-%Y')
        data['Time'].iloc[-1] = dt.datetime.fromtimestamp(time.time()).strftime(format ='%H:%M')

    else:
        pass

    print(data.iloc[-2:,:])


    data['+ve_cross'] = data['-ve_cross'] = data['stop_loss'] = data['20_Cross_MA']= 0

    data['+ve_cross'] = np.where((data['PnL'] > 0) & (data['Trade'] == 'New_Short'),1,0)
    data['+ve_cross'] = np.where((data['PnL'] > 0) & (data['Trade'] == 'New_Long'),1,data['+ve_cross'])
    data['+ve_cross'] = np.where((data['PnL'] > 0) & (data['Trade'] == 'Exit_Short'),1,data['+ve_cross'])
    data['+ve_cross'] = np.where((data['PnL'] > 0) & (data['Trade'] == 'Exit_Long'),1,data['+ve_cross'])

    data['-ve_cross'] = np.where((data['PnL'] < 0) & (data['Trade'] == 'New_Short'),1,0)
    data['-ve_cross'] = np.where((data['PnL'] < 0) & (data['Trade'] == 'New_Long'),1,data['-ve_cross'])
    data['-ve_cross'] = np.where((data['PnL'] < 0) & (data['Trade'] == 'Exit_Short'),1,data['-ve_cross'])
    data['-ve_cross'] = np.where((data['PnL'] < 0) & (data['Trade'] == 'Exit_Long'),1,data['-ve_cross'])

    data['stop_loss'] = np.where((data['Trade'] == 'SL_Short'),1,0)
    data['stop_loss'] = np.where((data['Trade'] == 'SL_Long'),1,data['stop_loss'])


    data['stop_loss_sum'] = (data['stop_loss'].rolling(20).sum())
    data['-ve_cross_sum'] = (data['-ve_cross'].rolling(20).sum())
    data['+ve_cross_sum'] = (data['+ve_cross'].rolling(20).sum())

    data['20_Cross_MA'] = data['-ve_cross_sum']/(data['stop_loss_sum']+data['-ve_cross_sum']+data['+ve_cross_sum'])
    # data = data.dropna()
    data['Datetime'] = pd.to_datetime(data['Day'] + " " + data['Time'], format = '%d-%m-%Y %H:%M')
    data.set_index('Datetime', drop = True, inplace = True)
    #data.to_csv(f"{path}\{coin}\pnl_1.8.csv")
    data.to_csv(f"{path}\{coin}\pnl_1.8.csv")
    # data.head(60)
    # data.tail(20)
    ## Plotting ##

    import matplotlib.pyplot as plt


    #plt.ylim([math.floor(p_l_dataframe.min().min()/10000)*10000,math.ceil(p_l_dataframe.max().max()/10000)*10000])
    # plt.show()
    plt.rcParams["figure.figsize"] = [15.00, 10.0]
    plt.rcParams["figure.autolayout"] = True
    fig, ax1 = plt.subplots()
    ax1.plot(data['Equity'], color='blue',linewidth=1)


    #plt.plot(data[data['-ve_cross'] == 1]['Equity'], marker = '*', color = 'orange',)
    #plt.plot(data[data['+ve_cross'] == 1]['Equity'], marker = '+', color = 'green')
    ax1.scatter(x=data[data['+ve_cross'] == 1].index, y =data[data['+ve_cross'] == 1]['Equity'], marker = '^', color = 'green',s =25*4, label = '+ve_PNL')
    ax1.scatter(x=data[data['-ve_cross'] == 1].index, y =data[data['-ve_cross'] == 1]['Equity'], marker = 'v', color = 'orange',s =25*4,label = '-ve_PNL')
    ax1.scatter(x=data[data['stop_loss'] == 1].index, y =data[data['stop_loss'] == 1]['Equity'], marker = 'o', color = 'red',s =25*4,label = 'SL')
    #plt.plot(data[data['stop_loss'] == 1]['Equity'], marker = 'o', color = 'red')
    #ax1.set_title(f'Backtest Report for {coin}/USDT at 1.80% Stoploss')
    ax1.set_title(f'Backtest Report for {coin}/USDT at 1.80% Stoploss')
    ax1.legend(loc='lower right',fontsize = 20)
    if current_trade[-1] == "g":
        color = "green"
    else:
        color = "red"
    ax1.text(0.9, 1, f'Current Trade : {current_trade}',
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax1.transAxes,
        color=color, fontsize=15)
    #plt.rcParams['font.size'] = '35'
    ax2 = ax1.twinx()
    ax1.set_xlabel("Date",fontsize = 20)
    ax1.set_ylabel("Equity_USD($)", fontsize = 20)
    ax2.plot(data['20_Cross_MA'], color='Red',linewidth = 1)
    ax2.set_ylabel("20_Cross_MA", fontsize = 20)
    #ax1.set_title(f'Backtest Report for {coin}/USDT at 1.80% Stoploss')
    fig.tight_layout()
    #plt.title("Equity & Signals", fontsize = 20)

    plt.rcParams['font.size'] = '20'
    plt.xticks(fontsize = 10)
    plt.yticks(fontsize = 10)


    plt.savefig("Trade_P&L.png", dpi = 300)
    ax2.legend(loc='lower left')
    #plt.show()
    plt.savefig(f'{coin}_pnl_report.png', dpi = 300)

