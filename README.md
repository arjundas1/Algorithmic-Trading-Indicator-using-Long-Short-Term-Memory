<h1 align="center"> Algorithmic Trading Indicator using Advanced Recurrent Neural Network </h1>
<!--
<p align="center">
  <a href="https://github.com/arjundas1/Algorithmic-Trading-Indicator-using-Recurrent-Neural-Network">
    <img src="" width="650" height="300">
  </a>
</p>
-->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#introduction">Introduction</a></li>
    <li><a href="#project-team">Project Team</a></li>
    <li><a href="#project-objective">Project Objective</a></li>
    <li><a href="#methodology">Background</a></li>
    <li><a href="#implementation">Implementation</a></li>
    <li><a href="#conclusion">Conclusion</a></li>
    <li><a href="#references">References</a></li>
  </ol>
</details>

## Introduction

Stock value prediction is a difficult task that necessitates a solid computational foundation in order to compute longer-term share values. Since stock prices are correlated by nature of the market, it is impossible to estimate the respective costs. Due to the stock market's volatility, it necessitates a great deal of analysis based on historical data. Intra-day trading requires accurate prediction models to efficiently bet on the future trends of the financial market.

The purchasing and selling of financial assets is known as financial trading. Trading types are used to categorise algorithmic trading. Foreign Exchange (FOREX), stock markets, exchange-traded funds (ETFs), bonds, cryptocurrencies, and others (assets, commodities, collateral mortgage, Credit Default Swap (CDS), and Interest Rate Swap (IRS)) are among the trading types.

Over the forecast period, the Algorithmic Trading Market is expected to grow at a CAGR of 10.5 percent (2022-2027). Traders have traditionally used market surveillance technology to maintain track of their trading operations and investment portfolio. Applications with built-in intelligence, such as algorithmic trading, explore the market for opportunities according on the yield and other criteria set by the user. The rise of AI, machine learning, and big data in the financial services sector is predicted to be a major contributor in the algorithmic trading market's expansion. 

In this model, we have made use of The Long Short Term Memory (LSTM) network which is a type of recurrent neural network (RNN) capable of addressing linear problems. LSTM is a deep learning technique and its units are enforced to learn very long sequences. An LSTM module has a cell state and three gates, giving it the ability to learn, unlearn, or retain information from each of the units selectively. By allowing only a few linear interactions, the cell state in LSTM allows information to travel across the units without being altered. Each unit contains an input, output, and a forget gate that adds or removes data from the cell state. The forget gate utilises a sigmoid function to determine whether information from the previous cell state should be ignored. The input gate uses a point-wise multiplication operation of'sigmoid' and 'tanh' to control the information flow to the current cell state. Finally, the output gate determines which data should be sent to the next concealed state. 

## Project Team
Guidance Professor: Dr. Varalakshmi M, School of Computer Science and Engineering, VIT Vellore.

Team members:

|Sl.No. | Name  | Registration No. |
|-| ------------- |:-------------:|
|1|   Arjun Das      | 20BDS0129     |
|2| Dharmik Naicker  | 20BCB0148     |
|3|  Isha Agarwal    | 20BCB0103     |
|4|  Manav Malhotra  | 20BCB0133     |

## Project Objective

To build a reliable intra day trading indicator for the use of future market prediction which is
computationally less costly and highly efficient as well as insensitive to changes in the market
trends. 

## Methodology

1) Using API’s finding suitable dataset from financial websites.
2) Using statistical methods and suitable charting software to visualize the data.
3) Selecting the reliable column from the dataset on which the model is to be trained.
4) On comparing various ML models,LSTM has been chosen as the best model out of all.
5) Loading the test data in the LSTM.
6) Checking the accuracy and plotting graphs to analyze the model’s reliability.
7) Deploying the model on trading platforms like tradingview, tradinglite etc.
8) Manually backtesting the strategy on a large sample size to with appropriate Risk to Reward
ratio.
9) Calculating the winrate to analyze the reliability of the model.
10) Using it on live markets for intra day trading. 

## Implementation

1) Financial data can be fetched from Yahoo Finance using various APIs as well as Python
wrapper packages such as Pandas DataReader, Yfinance and FXCM. The primary
financial asset chosen is Bitcoin USD.

```python
import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
df = yf.download(["BTC-USD"], start=dt.datetime(2021, 1, 1), end=dt.datetime.now())
df
```

2) Fetching the data and plotting the close bid of the time series data to understand the
complexity of the dataset.

```python
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("seaborn")
df["Close"].plot(figsize=(15,8), fontsize=14)
plt.legend(fontsize=14)
plt.show()
```
![image](https://user-images.githubusercontent.com/72820515/165214615-495fcf8c-6d8b-4fb8-b704-8341f6f551aa.png)

3) Realising the simple and logarithmic return of the dataset to under the risk factor of the
market to efficiently build an insensitive model. 

```python
df = df[["Close","Adj Close"]]
df["Lag"] = df["Close"].shift(periods=1)
df["Diff"] = df["Close"].sub(df["Lag"])
df["Return"] = ((df["Close"].div(df["Lag"])).sub(1)).mul(100)
df = df[["Close","Return"]]
df
```

4) There are various statistical and machine learning models that can be used on a time
series data. Auto correlation and moving average based models work efficiently but
needs immense preprocessing which makes the program computationally costly.

```python
retn = df["Return"]
retn = retn.dropna()
retn.plot(kind="hist", figsize=(12,8),bins=100)
plt.show()
```

![image](https://user-images.githubusercontent.com/72820515/165214691-87b457a2-c0ef-4a73-bf76-b78375237fce.png)

5) Since moving average models need to have a compulsory stationary data, additional
preprocessing will be required on our dataset. This was realised by plotting the ACF
and PACF graphs. 

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(df["Close"])
plt.show()
```
![image](https://user-images.githubusercontent.com/72820515/165214771-2ae1d41f-e205-45bd-bd02-6f2b69bf3f2c.png)

```python
plot_pacf(df["Close"])
plt.show()
```
![image](https://user-images.githubusercontent.com/72820515/165214813-987a6ad3-a387-4782-8d82-d4fcdf21fa54.png)

6) Preprocessing the data to fit the data efficiently in the model. The data needs to be
transformed to a lower value so that the neural network cannot have any probability of
biasing.

```python
import pandas_datareader as web
from sklearn.preprocessing import MinMaxScaler
company = 'BTC-USD'
start = dt.datetime(2020, 1, 1)
end = dt.datetime(2021,1,1)
data = web.DataReader(company, 'yahoo', start, end)
scaler=MinMaxScaler(feature_range=(0,1))
df=scaler.fit_transform(data['Close'].values.reshape(-1,1))
```

7) Reshaping the training data so that it can be in-sync with the DL model requirements.
This is done by creating a prediction days variable which will be used to look back the
last n days and will predict the n+1 day value on the training data.

```python
prediction_days = 30
x_train=[]
y_train=[]
for x in range(prediction_days, len(df)):
x_train.append(df[x-prediction_days:x, 0])
y_train.append(df[x, 0])
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
```

8) In the model creation module we are creating a LSTM model with one input layer, two
hidden layers, and one dense layer which will then predict the next day price of the
data.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')
```

9) The most suitable model was found for 25 epochs with a batch size of 32.

```python
model.fit(x_train, y_train, epochs=25, batch_size=32)
```

10)Now finally we will load our test data which is also reshaped and preprocessed like the
train data. The model has never seen this test data before.

```python
test_start=dt.datetime(2022, 1, 1)
test_end=dt.datetime.now()
test_data = web.DataReader(company, 'yahoo', test_start, test_end)
actual_prices=test_data['Close'].values
total_dataset=pd.concat((data['Close'], test_data['Close']), axis=0)
model_inputs=total_dataset[len(total_dataset)-len(test_data)-prediction_days:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs)
x_test=[]
for x in range(prediction_days, len(model_inputs)):
x_test.append(model_inputs[x-prediction_days:x, 0])
x_test=np.array(x_test)
x_test=np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
predicted_prices=model.predict(x_test)
predicted_prices=scaler.inverse_transform(predicted_prices)
real_data = [model_inputs[len(model_inputs)+1-prediction_days:len(model_inputs+1), 0]]
real_data = np.array(real_data)
real_data=np.reshape(real_data, (real_data.shape[0], real_data.shape[1],1))
```

11) Based on how intensive the training model has been gone through, it will predict the
next day price.

```python
prediction=model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(f"Prediction: {prediction}")
```

12)The model will be plotted on a 2D X-Y graph which will display the curves of actual
price and the predicted price.

```python
plt.plot(actual_prices, color = "black", label=f"Actual {company} Price")
plt.plot(predicted_prices, color="green", label=f"Predicted {company} Price")
plt.title(f"{company} Crypto Price")
plt.xlabel("Time")
plt.ylabel(f"{company} Crypto Price")
plt.legend()
plt.show()
```
![image](https://user-images.githubusercontent.com/72820515/165214863-2a632f58-b148-45c2-8242-dbf9357c27c6.png)

13) The graph can thus then aid us in visualizing the overall movement of the market. This
predicted closing price is then imported in a custom indicator on a trading platform (such as TradingView). This further generates the buy sell signal which can be used in
intra day trading with variable timeframes. (Preferably 5min - 1Hr).

## Conclusion

With recent advances in the development of sophisticated machine learning-based
approaches, particularly deep learning algorithms, these techniques are gaining favour
among academics from a variety of disciplines. This paper carries out a share analysis,
which can be repeated for several shares in the future. The research presented in this
paper argues for the advantages of using LSTM to analyse economic and financial data.

In the future, we plan to add sentiment analysis from social media to our programme to
better understand what the market thinks about price variations for specific stocks. We
can do this by adding Twitter and Facebook APIs to our programme, as Facebook is a
popular social media platform with a lot of market trend information posted by users.

Chart: BTC-USDT BYBIT Perpetual

Time Frame: 8 Minutes

Platform: Trading View

Additional 50(Red)/200(Orange) Exponential moving average indicator is used to analyze the overall trend of the market. When the 50 is above 200 EMA we are looking for buy/long trades. Vice versa for short trades.

![image](https://user-images.githubusercontent.com/72820515/165215022-a9d60dc1-eb9c-4d39-b9ef-61d4d3192e3c.png)
<p align="center">
  <b>Screenshot of BTC-USDT chart before applying Indicator</b>
</p>

The above picture shows chart of BTC-USDT pair on 8 minutes timeframe. The highest price that pair reached in that time frame is shown by Upper Shadow. Similarly the lowest price is shown by Lower Shadow. Real body is made around the opening price and closing price in that time frame. The same is shown in th e figure below.

![image](https://user-images.githubusercontent.com/72820515/165215338-0b217759-f4dc-4c0f-a7c0-d17e12dfdd59.png)
<p align="center">
  <b>Scheme of candlestick</b>
</p>

After the analysis of the algorithm, the same algorithm is used on trading exchange to generate signals. When a trade is placed, it is either 'buying' or 'selling' a financial instrument.

Buy Trade or Long: When a Long trade is undertaken in expectation its price will rise.

Sell Trade or Short: When a Short trade is undertaken in expectation its price will fall.

Below are few screenshots of the indicator after integrating it in Trading View. Three examples of each Long and Short Trades are given:

### Long Trades:

![image](https://user-images.githubusercontent.com/72820515/165215432-fed86f88-cb0e-4be6-ba3c-24b59011efe8.png)
<p align="center">
  <b>Buy Signal Generated on 21st April at 13:14PM using LSTM Indicator</b>
</p><br><br><br>

![image](https://user-images.githubusercontent.com/72820515/165215481-308affe0-404b-47b1-a870-dfc4def1c7a4.png)
<p align="center">
  <b>Buy Signal Generated on 20th April at 14:26PM using LSTM Indicator</b>
</p><br><br><br>

![image](https://user-images.githubusercontent.com/72820515/165216382-1eafe7e0-9864-45b0-8545-ecf735050eb7.png)
<p align="center">
  <b>Buy Signal generated on 19th April 2022 at 17:14PM using LSTM Indicator</b>
</p><br><br>

### Short Trades:

![image](https://user-images.githubusercontent.com/72820515/165216459-94c595c1-1af6-4934-a132-f1bd0d10c064.png)
<p align="center">
  <b>Sell Signal generated on 25th April at 11:30AM using LSTM Indicator</b>
</p><br><br><br>

![image](https://user-images.githubusercontent.com/72820515/165216488-273c52fc-86e6-4125-869c-cef1964c9b94.png)
<p align="center">
  <b>Sell Signal Generated on 25th April at 4:02AM using LSTM Indicator</b>
</p><br><br><br>

![image](https://user-images.githubusercontent.com/72820515/165216542-987a0dfc-4741-47cc-8ab5-f73e3be09698.png)
<p align="center">
  <b>Sell Signal generated on 24th April at 16:50PM using LSTM Indicator</b>
</p><br>

## References

[Siami-Namini, S., Tavakoli, N., & Namin, A. S. (2018, December). A comparison of ARIMA and LSTM in forecasting time series. In 2018 17th IEEE international conference on machine learning and applications (ICMLA) (pp. 1394-1401). IEEE.]()

[Selvin, S., Vinayakumar, R., Gopalakrishnan, E. A., Menon, V. K., & Soman, K. P. (2017, September). Stock price prediction using LSTM, RNN and CNN-sliding window model. In 2017 international conference on advances in computing, communications and informatics (icacci) (pp. 1643-1647). IEEE.]()

[Alzheev, A. V., & Kochkarov, R. A. (2020). Comparative analysis of ARIMA and lSTM predictive models: Evidence from Russian stocks. Finan.: Theory Pract, 4(1), 14-23.]()

[Liu, S., Liao, G., & Ding, Y. (2018, May). Stock transaction prediction modeling and analysis based on LSTM. In 2018 13th IEEE Conference on Industrial Electronics and Applications (ICIEA) (pp. 2787-2790). IEEE.]()

[Kwon, D. H., Kim, J. B., Heo, J. S., Kim, C. M., & Han, Y. H. (2019). Time series classification of cryptocurrency price trend based on a recurrent LSTM neural network. Journal of Information Processing Systems, 15(3), 694-706.]()

[Banik, S., Sharma, N., Mangla, M., Mohanty, S. N., & Shitharth, S. (2022). LSTM based decision support system for swing trading in stock market. Knowledge-Based Systems, 239, 107994.]()

[Zou, Z., & Qu, Z. (2020). Using LSTM in Stock prediction and Quantitative Trading. CS230: Deep Learning, Winter.]()

[Chen, G., Chen, Y., & Fushimi, T. (2021). Application of deep learning to algorithmic trading. Stanford University, Tech. Rep, Tech. Rep., 2017.[Online]. Available: http://cs229. stanford. edu/proj2017/final-reports/5241098. Pdf.]()

[Moghar, A., & Hamiche, M. (2020). Stock market prediction using LSTM recurrent neural network. Procedia Computer Science, 170, 1168-1173.]()

[Pramod, B. S., & Pm, M. S. (2020). Stock price prediction using LSTM. Test Engineering and Management, 83, 5246-5251.]()

[Shin, H. G., Ra, I., & Choi, Y. H. (2019, October). A deep multimodal reinforcement learning system combined with CNN and LSTM for stock trading. In 2019 International conference on information and communication technology convergence (ICTC) (pp. 7-11). IEEE.]()

[Wu, J. M. T., Li, Z., Herencsar, N., Vo, B., & Lin, J. C. W. (2021). A graph-based CNN-LSTM stock price prediction algorithm with leading indicators. Multimedia Systems, 1-20.]()
