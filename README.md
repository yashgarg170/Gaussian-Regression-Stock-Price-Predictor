Predicting Price of Apple Stock Exchange using GPR and Random Forest.

Daily high price for about 1000 trading days was sourced from Yahoo Finance. Initial data looked like this: 

![image](https://github.com/aditya8503/Stock-price-Prediction/assets/168825142/b4d0aad4-1b07-4137-986d-18338de0251d)

Since our intention was to perform time series forecasting and none of these methods were ideal time series, we had to improve. 

During the initial data processing, for every input row into the algo, 30 days high prices were feeded as X and the next day price as Y. This ensured that every input contained a time series of the last 30 days. 

![image](https://github.com/aditya8503/Stock-price-Prediction/assets/168825142/bec2a13f-0514-4ef5-80f8-2b70c9a17693)


A 30/70 test/train split was made to validate the algorithm and then predict data 30 days into the future. 

This helped us input the data for a duration of time as a row and easily build a model for forecasting price.

Here's how forecasted data compared with the actual data.

![GPR 30% Test vs Train ](https://github.com/aditya8503/Stock-price-Prediction/assets/168825142/22ac3132-695a-4946-9dc7-71c3c95a22fb)

This it the forecast price for 15 days into the future.

![GPR 30Day Forecasting](https://github.com/aditya8503/Stock-price-Prediction/assets/168825142/799c403f-e16a-4ca4-a0cb-7ff7ff8091c3)



