import pandas as pd
import pandas_datareader.data as web
from datetime import datetime

start_date = '2014-01-01'
end_date ='2024-01-01'

qqq_data = web.DataReader('QQQ', 'stooq', start_date, end_date)
tqqq_data = web.DataReader('TQQQ', 'stooq', start_date, end_date)

print(qqq_data.head())
print(tqqq_data.head())
