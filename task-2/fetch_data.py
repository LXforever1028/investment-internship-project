import pandas as pd
import pandas_datareader.data as web
from datetime import datetime

start_date = '2014-01-01'
end_date ='2024-01-01'

qqq_data = web.DataReader('QQQ', 'stooq', start_date, end_date)
tqqq_data = web.DataReader('TQQQ', 'stooq', start_date, end_date)

qqq_df = qqq_data.reset_index()
qqq_df['symbol'] = 'QQQ'
qqq_df = qqq_df.rename(columns = {'Date': 'date', 'Open': 'open', 
	'Close': 'close', 'High': 'high', 'Low': 'low', 'Volume': 'volume'})
qqq_df = qqq_df[['symbol', 'date', 'open', 'close', 'high', 'low', 'volume']]
qqq_df['date'] = qqq_df['date'].dt.strftime('%m-%d-%Y')

tqqq_df = tqqq_data.reset_index()
tqqq_df['symbol'] = 'TQQQ'
tqqq_df = tqqq_df.rename(columns = {'Date': 'date', 'Open': 'open', 
	'Close': 'close', 'High': 'high', 'Low': 'low', 'Volume': 'volume'})
tqqq_df = tqqq_df[['symbol', 'date', 'open', 'close', 'high', 'low', 'volume']]
tqqq_df['date'] = tqqq_df['date'].dt.strftime('%m-%d-%Y')

print(qqq_df.head())
print(tqqq_df.head())

qqq_df.to_csv("QQQ.csv", index = False)
tqqq_df.to_csv("TQQQ.csv", index = False)
