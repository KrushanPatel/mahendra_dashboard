import pandas as pd
import ccxt
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings("ignore")


def exchange():

    client = ccxt.binanceusdm({
        "apiKey": "xBPLNbrLuBqVmYriXB2lVFWa7XEPfUIOyo1Sjvft21SmfZMRUxDz2BcXNFGXGxOw",
        "secret": "W4Pv0VODKY6eT4p4wW7QwW30yXm2ziu7IYQH19M2U9NkmOH7ZBscNi4yLUhAlSFr",
        'options': {'adjustForTimeDifference': True}
    })

    return client


