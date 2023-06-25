config = {
    #'symbol':'ETH',
    'file_name':'Bittrex_ETHUSD_1h.csv',    ##make sure file is in same folder as backtest code
    #'ema_long':26,                   ##if file in different folder, give entire path
    #'ema_short':12,
    #"ao_win_short":5,
    #"ao_win_long":34,
    'vol_threshold':150,
    'vol_window':10,
    'multiply_std':100,   ##Always 100. we use this to multiply log returns
    'value_sqrt':365,
    'freq_factor': 24,     ##used in volatility calculation. if data is 1hr frequency, we give 24. 
    'stop_loss':[1.8],#[1,5,1.8,2.0,2.3,2.5,2.8,3.0,3.3,3.5,3.8,4.0,4.3,4.5,4.8,5.0],   #in percentage  ~ 2%
    'trail_perc' : 2,     #in percentage
    'qty':10000,          # if 0, it will automatically calculate qty based on capital leaving aside 5% for drawdown
    'ini_capital':10000,    #in dollars
    'leverage': 1        #0.2 means 20% of capital. 1 == 100% of capital, 2 == 2x of capital
    }
