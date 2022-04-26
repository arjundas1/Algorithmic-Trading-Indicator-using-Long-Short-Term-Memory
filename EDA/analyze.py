import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
df = yf.download(["BTC-USD"], start=dt.datetime(2021, 1, 1), end=dt.datetime.now())
df = df[["Close","Adj Close"]]
plt.style.use("seaborn")
df["Close"].plot(figsize=(15,8), fontsize=14)
plt.legend(fontsize=14)
plt.show()
df["Lag"] = df["Close"].shift(periods=1)
df["Diff"] = df["Close"].sub(df["Lag"])
df["Return"] = ((df["Close"].div(df["Lag"])).sub(1)).mul(100)
df = df[["Close","Return"]]
retn = df["Return"]
retn = retn.dropna()
retn.plot(kind="hist", figsize=(12,8),bins=100)
plt.show()
plot_acf(df["Close"])
plt.show()
plot_pacf(df["Close"])
plt.show()
