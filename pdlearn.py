import pandas as pd
import datetime
import pandas_datareader.data as web

start = datetime.datetime(2018, 1, 1)
end = datetime.datetime(2020, 7, 20)

df = web.DataReader("XOM", "yahoo", start, end)

print(df.head())

import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')
df['High'].plot()
plt.legend()
plt.show()