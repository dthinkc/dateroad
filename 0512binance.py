import pandas as pd
import numpy as np
import ccxt
from  time import sleep
from datetime import datetime,timedelta
from datetime import datetime, timezone
pd.set_option('expand_frame_repr', False)  # 当列太多时不换
pd.set_option("display.max_rows", 5000)  # 显示最大行数
pd.set_option('precision', 3)






# 获取binance的k线数据
def get_binance_candle_data(exchange, symbol, time_interval1):
    # 抓取数据
    content = exchange.fetch_ohlcv(symbol, timeframe=time_interval1,limit=limit)

    df = pd.DataFrame(content, dtype=float)
    df.rename(columns={0: 'date', 1: 'open', 2: 'high', 3: 'low', 4: 'close', 5: 'volume'}, inplace=True)
    df['candle_begin_time'] = pd.to_datetime(df['date'], unit='ms')
    df['candle_begin_time'] = df['candle_begin_time'] + timedelta(hours=8)
    df = df[['candle_begin_time', 'open', 'high', 'low', 'close', 'volume']]

    return df

  #择时策略
def signal_moving_average(df):
    #计算均线
    df['average']=(df['high']-df['low']).rolling(n).mean()
    df['sell_average1'] = df['close'] - df['average'] * k #买入止损

    #市场效率
    '''df['er']=abs(df['close'].diff(n))/abs(df['close'].diff(1)).rolling(n).sum()
    #平滑常数
    df['sc']=df['er']*(2/3-2/31)+2/31
    #移动平均
    df['AMA']=df['close']
    df['AMA']=df['AMA'].shift(1)+pow(df['sc'],2)*(df['close']-df['AMA'].shift(1))'''

    # %%
    def KAMA(price, n=10, pow1=2, pow2=30):
        ''' kama indicator '''
        ''' accepts pandas dataframe of prices '''

        absDiffx = abs(price - price.shift(1))

        ER_num = abs(price - price.shift(n))
        ER_den = absDiffx.rolling(n).sum()
        # ER_den = pd.stats.moments.rolling_sum(absDiffx, n)
        ER = ER_num / ER_den

        sc = (ER * (2.0 / (pow1 + 1) - 2.0 / (pow2 + 1.0)) + 2 / (pow2 + 1.0)) ** 2.0

        answer = np.zeros(sc.size)
        N = len(answer)
        first_value = True

        for i in range(N):
            if sc[i] != sc[i]:
                answer[i] = np.nan
            else:
                if first_value:
                    answer[i] = price[i]
                    first_value = False
                else:
                    answer[i] = answer[i - 1] + sc[i] * (price[i] - answer[i - 1])
        return answer

    # %%

    # %%
    # calculate KAMA
    # ---------------
    kama = KAMA(df['close'], n=n, pow1=2, pow2=30)
    df['kama'] = kama

    #过滤器
    df['lz']=pt*df['kama'].rolling(20).std()

    # 找出买入信号
    df['buy'] = df['kama'] - df['kama'].rolling(n).min()
    condition1 = df['lz'] < df['buy']
    condition2 = df['lz'].shift(1) > df['buy'].shift(1)
    df.loc[condition1 & condition2, 'signal_short'] = 1  # 将产生做多信号的那根K线的signal设置为1，1代表做多


    # 卖出信号
    '''df['sell'] = df['kama'].rolling(n).max() - df['kama']
    condition1 = df['lz'] < df['sell']
    condition2 = df['lz'].shift(1) > df['sell'].shift(1)
    df.loc[condition1 & condition2, 'signal_long'] = -2  # 将产生做空信号的那根K线的signal设置为0，0代表做空'''

    # 均线卖出信号
    df['sell_average'] = (df['close']-df['average']*k).rolling(n).max()
    df['sell_kong'] = (df['close'] + df['average'] * k).rolling(n).min() #做空移动平均卖出
    df['sell_average'].fillna(method='ffill', inplace=True)
    condition1 = df['close'] < df['sell_average']
    condition2 = df['close'].shift(1) > df['sell_average'].shift(1)
    df.loc[condition1 & condition2, 'signal_long2'] = -1  # 将产生做空信号的那根K线的signal设置为0，0代表做空


    # 均线卖出止盈信号
    df['nsell_average'] = df[df['signal_short']==1]['close']+df['average']*w
    df['nsell_average'].fillna(method='ffill', inplace=True)
    condition1 = df['close'] > df['nsell_average']
    condition2 = df['close'].shift(1) < df['nsell_average'].shift(1)
    df.loc[condition1 & condition2, 'signal_long3'] = -1  # 将产生做空信号的那根K线的signal设置为0，0代表做空


    #合并做空去除重复
    df['signal']=df[['signal_short','signal_long2','signal_long3']].sum(axis=1, skipna=True)
    df['signal'] = df[df['signal'] != 0][['signal']]
    temp = df[df['signal'].notnull()][['signal']]
    temp = temp[temp['signal'] != temp['signal'].shift(1)]
    df['signal'] = temp['signal']
    df.drop(['volume'], axis=1, inplace=True)

    # ===由signal计算出实际的每天持有仓位
    # signal的计算运用了收盘价，是每根K线收盘之后产生的信号，到第二根开盘的时候才买入，仓位才会改变。
    df['pos'] = df['signal'].shift(1)
    #df['pos'].fillna(method='ffill', inplace=True)
    df['pos'].fillna(value=0, inplace=True)  # 将初始行数的position补全为0

    return df

exchange = ccxt.binance({'recvWindow': 10000000})
exchange.load_markets()
exchange.apiKey = 'YCCvmMXFaXGcuwGF0GnbheuyWmC4514LvJGy0y41s8R2ZVggW5gJhzKKvY7Af2F9'  # 此处加binance上自己的apikey和secret，都需要开通交易权限
exchange.secret = 'haTexGaG8321IBBzq84e3Zlr8fUmdbea1e32pmJSKgLFfSCLqFH8K8LL6pAQtEe2'
# =====获取账户资产

time_interval='1d'
#time_interval='5m'

limit=10000
n=10
pt=0.24
k=2.4
w=7.2
symbol='BTC/USDT'
base_coin = symbol.split('/')[-1]
trade_coin = symbol.split('/')[0]

df = get_binance_candle_data(exchange, symbol, time_interval)
df=signal_moving_average(df)




tpop=df[df['signal'].notnull()]

tpop = tpop[['candle_begin_time',   'close', 'sell_average', 'nsell_average', 'signal']]
df = df[['candle_begin_time', 'open', 'high', 'low', 'close', 'signal_short',"signal_long2"]]


#print(tpop)
print(df)
