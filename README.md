# Algorithmic Trading Indicator using Recurrent Neural Network
Literature Survey
Siami-Namini, S. et. al (2018) The research topic addressed in this article is whether and how newly discovered deep learning-based algorithms for predicting time series data, such as "Long Short-Term Memory (LSTM)," are superior to existing methods for forecasting time series data. The authors did this by comparing the performance of LSTM and ARIMA in terms of error rate minimization in prediction. Deep learning-based algorithms, such as LSTM, outperform traditional-based algorithms, such as the ARIMA model, according to the empirical tests undertaken and reported. When compared to ARIMA, the average reduction in error rates attained by LSTM was between 84 and 87 percent, demonstrating that LSTM is better to ARIMA.

Selvin, S. et. al (2017) In this article, a model-independent strategy is suggested. They are employing deep learning architectures to find the hidden dynamics in the data, rather than fitting the data to a predefined model. They compare the performance of three distinct deep learning architectures for price prediction of NSE listed businesses in this paper. They're using a sliding window method to forecast future values in the short term. Percentage error was used to assess the models' performance.

Alzheev, A. V. et. al (2020) The goal of this article is to develop the optimal time series predictive model that minimises mistakes while maintaining high forecast accuracy. The authors compared the most common "conventional" econometric model ARIMA with the deep learning model LSTM (Long short-term memory) based on a recurrent neural network in this study. These prediction models are mathematically described in the paper. The LSTM model was shown to be superior than the ARIMA model in terms of RMSE error, which is 65 percent lower. As a result, an LSTM model-based approach is preferable for higher time series prediction quality.

Liu, S. et. al (2018) The authors of this study describe how LSTM (Term Memory Long-Short) is a type of time recurrent neural network that is capable of analysing and predicting critical events in time series with interval and long delays. This study utilises LSTM recurrent neural networks to filter, extract feature value, and analyse stock data, and set up the prediction model of the corresponding stock transaction, based on the temporal features of stock and the LSTM neural network technique.

Kwon, D. H. et. al (2019) The authors used the long short-term memory (LSTM) model to categorise bitcoin price time series in this study. They gathered historical bitcoin price time series data and cleaned it up so it could be used as train and target data. Following this preprocessing, the price time series data was methodically encoded into a three-dimensional price tensor that represented historical cryptocurrency price movements. Finally, their research found that the LSTM model beats the gradient boosting model, a general machine learning model recognised for having strong prediction ability, for time series categorization of the bitcoin price trend, based on the comparison of f1-score values. When compared to the GB model, they experienced a performance boost of roughly 7% utilising the LSTM model.

Banik, S. et al (2022) A Long Short Term Memory enforced Decision Support System for swing traders is created in this work to properly assess and anticipate future stock prices. The Decision Support System creates a report that includes the firm stock's expected values for the following 30 days. The trader can supplement his investment selections with the investment success score determined in the report. The suggested model achieves 4.13, 3.24, and 1.21 percent in terms of Root Mean Square Error, Mean Absolute Error, and Mean Absolute Percentage Error, respectively, demonstrating the usefulness of the proposed approach when compared to state-of-the-art techniques.

Zou, Z. et. al (2020) The authors built and used the state-of-the-art deep learning sequential models, such as the Long Short Term Memory Model (LSTM), StackedLSTM, and Attention-Based LSTM, as well as the classic ARIMA model, to forecast stock prices for the following day in this study. Furthermore, they developed two trading strategies based on their forecast and compared them to the benchmark. Their input data comprises company financial figures, which are carefully picked and implemented into the models, in addition to standard end-day pricing and trade volumes. The results reveal that Attention-LSTM outperforms all other models in terms of prediction error and returns in our trading strategy when compared to other models.

Chen, G. et. al (2021) The authors use a Long Short-Term Memory Network to forecast INTC stock price and apply the trained network to an algorithmic trading problem in this research. They show how LSTM can properly anticipate INTC's price the next day, even when there is no trend in the stock price. The strategy based on the LSTM prediction outperforms the other two methods in terms of cumulative daily returns.

Moghar, A. et. al (2020) The goal of this article is to develop a model that uses Recurrent Neural Networks (RNN) and, in particular, the Long-Short Term Memory model (LSTM) to estimate future stock market values and adjusted closing prices for a portfolio of assets. The major goal of this study is to investigate how precise a Machine Learning algorithm can forecast future values for their portfolio and how much the epochs may enhance the model to produce the most accurate trained algorithm. The results of the tests show that their algorithm can track the evolution of opening prices for both assets.

Pramod, B. S. et. al (2020) The proposed approach uses market data to forecast share price using machine learning techniques such as recurrent neural networks with Long Short Term Memory, with weights modified for each data point using stochastic gradient descent. In compared to currently available stock price prediction algorithms, this method will deliver reliable results. To encourage the graphical outputs, the network is trained and assessed with varied sizes of input data.

Shin, H. G. et. al (2019) For stock price prediction, this study offers a deep multimodal reinforcement learning approach that combines convolutional neural networks (CNN) with long short-term memory (LSTM) neural networks. The proposed machine learning system leverages stock trading data to create multiple charts, which are then fed into the CNN layer. The CNN layer extracts features, which are then separated into column vectors and sent into the LSTM layer. A multi-modal structure is used to employ multiple types of charts, in which the input layer and the concealed layer are separated, and the individual operation results are merged. Finally, the model's performance is great in both bear and bull markets.

Wu, J. M. T. et. al (2021) This article presents a novel framework structure that blends Convolution Neural Network (CNN) and Long–Short–Term Memory Neural Network to obtain a more accurate stock price forecast (LSTM). Stock sequence array convolutional LSTM is the term given to this innovative approach (SACLSTM). It creates a sequence array of historical data and its leading indicators (options and futures), and uses the array as the input image of the CNN framework, extracting certain feature vectors through the convolutional layer and the layer of pooling, and as the input vector of LSTM, with ten stocks in the United States of America and Taiwan as the experimental data. When compared directly to earlier approaches, the prediction performance of the suggested algorithm in this paper produces better results.




References
Siami-Namini, S., Tavakoli, N., & Namin, A. S. (2018, December). A comparison of ARIMA and LSTM in forecasting time series. In 2018 17th IEEE international conference on machine learning and applications (ICMLA) (pp. 1394-1401). IEEE.

Selvin, S., Vinayakumar, R., Gopalakrishnan, E. A., Menon, V. K., & Soman, K. P. (2017, September). Stock price prediction using LSTM, RNN and CNN-sliding window model. In 2017 international conference on advances in computing, communications and informatics (icacci) (pp. 1643-1647). IEEE.

Alzheev, A. V., & Kochkarov, R. A. (2020). Comparative analysis of ARIMA and lSTM predictive models: Evidence from Russian stocks. Finan.: Theory Pract, 4(1), 14-23.

Liu, S., Liao, G., & Ding, Y. (2018, May). Stock transaction prediction modeling and analysis based on LSTM. In 2018 13th IEEE Conference on Industrial Electronics and Applications (ICIEA) (pp. 2787-2790). IEEE.

Kwon, D. H., Kim, J. B., Heo, J. S., Kim, C. M., & Han, Y. H. (2019). Time series classification of cryptocurrency price trend based on a recurrent LSTM neural network. Journal of Information Processing Systems, 15(3), 694-706.

Banik, S., Sharma, N., Mangla, M., Mohanty, S. N., & Shitharth, S. (2022). LSTM based decision support system for swing trading in stock market. Knowledge-Based Systems, 239, 107994.

Zou, Z., & Qu, Z. (2020). Using LSTM in Stock prediction and Quantitative Trading. CS230: Deep Learning, Winter.

Chen, G., Chen, Y., & Fushimi, T. (2021). Application of deep learning to algorithmic trading. Stanford University, Tech. Rep, Tech. Rep., 2017.[Online]. Available: http://cs229. stanford. edu/proj2017/final-reports/5241098. Pdf.

Moghar, A., & Hamiche, M. (2020). Stock market prediction using LSTM recurrent neural network. Procedia Computer Science, 170, 1168-1173.

Pramod, B. S., & Pm, M. S. (2020). Stock price prediction using LSTM. Test Engineering and Management, 83, 5246-5251.

Shin, H. G., Ra, I., & Choi, Y. H. (2019, October). A deep multimodal reinforcement learning system combined with CNN and LSTM for stock trading. In 2019 International conference on information and communication technology convergence (ICTC) (pp. 7-11). IEEE.

Wu, J. M. T., Li, Z., Herencsar, N., Vo, B., & Lin, J. C. W. (2021). A graph-based CNN-LSTM stock price prediction algorithm with leading indicators. Multimedia Systems, 1-20.
