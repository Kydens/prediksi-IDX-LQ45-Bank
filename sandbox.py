import yfinance as yf
import pandas as pd

ticker = yf.Ticker('BBCA.JK')

df_info = pd.DataFrame.from_dict(ticker.info, orient='index')

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

print(f'Information: {df_info}')
print(f'Full Information: {ticker.info}')

