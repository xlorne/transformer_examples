import pandas as pd
import requests
import pytz
import matplotlib.pyplot as plt


def data_format(data):
    data = pd.DataFrame(data)
    data.columns = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'volCcy', 'volCcyQuote', 'confirm']  # 设置列名
    data.drop(['volCcy', 'volCcyQuote', 'confirm'], axis=1, inplace=True)  # 删除无用列

    data['Open'] = data['Open'].astype(float)
    data['Close'] = data['Close'].astype(float)
    data['High'] = data['High'].astype(float)
    data['Low'] = data['Low'].astype(float)
    data['Volume'] = data['Volume'].astype(float)

    data['|'] = '|'
    data['Date'] = pd.to_datetime(data['Timestamp'].astype(float) / 1000, unit='s').dt.tz_localize('UTC').dt.tz_convert(
        pytz.timezone('Asia/Shanghai')).dt.strftime('%Y-%m-%d %H:%M:%S')

    data['Direction'] = data['Close'] - data['Open']  # 计算涨跌幅
    data['Amplitude'] = data['High'] - data['Low']  # 计算最大振幅
    return data


def get_candles(symbol, bar, limit):
    url = "https://www.okex.com/api/v5/market/candles?instId=%s&bar=%s&limit=%s" % (symbol, bar, limit)
    response = requests.get(url)
    data = response.json()['data']
    return data_format(data)


def get_history_candles(symbol, bar, limit, after):
    if after is None or after == 0:
        url = "https://www.okex.com/api/v5/market/history-candles?instId=%s&bar=%s&limit=%s" % (symbol, bar, limit)
    else:
        url = "https://www.okex.com/api/v5/market/history-candles?instId=%s&bar=%s&limit=%s&after=%s" % (symbol, bar, limit, after)
    response = requests.get(url)
    data = response.json()['data']
    return data_format(data)


def fetch_candles(symbol, bar, limit, max_epoch=0, callback=None):
    if callback is None:
        callback = lambda x: print(x.to_string())

    data = get_candles(symbol, bar, limit)
    callback(data)
    last = data.iloc[-1]['Timestamp']
    epoch = 1
    while True:
        data = get_history_candles(symbol, bar, limit, last)
        callback(data)
        last = data.iloc[-1]['Timestamp']
        epoch += 1
        if 0 < max_epoch <= epoch:
            break


def show_candles(data):
    # 可视化展示close数据
    # 转换 'Date' 列为 datetime 类型并设置为索引
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.set_index('Date')

    # 使用 Matplotlib 绘制 'Close' 数据
    plt.figure(figsize=(10, 5))  # Set the figure size
    plt.plot(data['Close'])
    plt.xticks(rotation=45)  # Rotate the x-axis labels for better visibility
    plt.title('Close Data Over Time')
    plt.xlabel('Date')
    plt.ylabel('Close')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    data = []
    symbol = 'BTC-USDT'
    bar = '1m'
    limit = 100
    max_epoch = 10

    def callback(x):
        data.append(x)

    fetch_candles(symbol, bar, limit, max_epoch, callback)
    data = pd.concat(data)
    data.index = range(len(data))

    print(data.to_string())
    show_candles(data)

    data.to_csv('data.csv', index=False)




