# Importing required libraries
import pandas as pd
import numpy as np
import os
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct, WhiteKernel
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


#Set the number of days for forecasting. 
Days = 30

# Load the dataset
dirname = os.path.dirname(__file__)
Data = pd.read_csv( os.path.join(dirname, 'APPL.csv'))
Data = Data.drop('Date', axis=1)
Data = Data.drop('Adj Close', axis=1)


# Data Transformation | Transforming Initial CSV to calculate Additional Columns
for i in range(1,Days+1):
    Temp = ([0] * i)  
    Close = Data['Close'].values.tolist()
    Close = (Close[i:])
    Values = Close
    for x in Temp:
        Values.append(x)

    if(Days == i):
        Data["Forecast"] = Values
    else:
        Data["Day " + str(i)] = Values


Data.to_csv(os.path.join(dirname, "Transform.csv"))

dataY = Data['Forecast'].values
dataX = Data.copy()
dataX = Data.drop('Forecast', axis=1)



# Values of Test Data split into X and Y components
TrainX, TestX, TrainY, TestY  = train_test_split(dataX, dataY, test_size=0.3, random_state=42, shuffle=False)


# Ignore | Just for plotting graphs, nothing to do with actual forecasting
PlotY = []
for i in TestY: 
    PlotY.append(i)

FutureX = []
for i in range(3000):
    FutureX.append(i)
    i = i+1










ForecastGPR = []
GPRSigma = []
# GPR 30% Test Data Testing using GPR
TrainX, TestX, TrainY, TestY  = train_test_split(dataX, dataY, test_size=0.1, random_state=4200, shuffle=False)
NewTestY = []
InitialTextX = TestX
ScaledTrainX = scaler.fit_transform(TrainX)
ScaledTestX = scaler.transform(TestX)



# Ignore | Just for plotting graphs, nothing to do with actual forecasting
PlotY = []
for i in TestY: 
    PlotY.append(i)

FutureX = []
for i in range(3000):
    FutureX.append(i)
    i = i+1


kernel =  RBF()+WhiteKernel() + DotProduct() 
gpr = GaussianProcessRegressor(kernel=kernel)
gpr.fit(ScaledTrainX, TrainY)


for i in range(1, len(TestX)):
    TestingArray = TestX.iloc[i].tolist()
    CheckingIndex = -Days
    while CheckingIndex < 0:
        if (TestingArray[CheckingIndex] <= 0):
            TestingArray[CheckingIndex] = float(NewTestY[CheckingIndex])

        CheckingIndex = CheckingIndex +1
    TestX.iloc[i] = TestingArray
    ScaledTestX = scaler.transform(TestX)

    ForecastGPRVal, SigmaGPRVal = gpr.predict(ScaledTestX[i].reshape(1, -1), return_std=True)
    NewTestY.append(ForecastGPRVal)
    ForecastGPR.append(float(ForecastGPRVal))
    GPRSigma.append(float(SigmaGPRVal))



fig = plt.figure(figsize = (8, 8))


# GPR | Plotting 30% Test vs Forecast
plt.fill_between(FutureX[:len(ForecastGPR)-Days], np.array(ForecastGPR[:-Days]) - 2*np.array(GPRSigma[:-Days]), np.array(ForecastGPR[:-Days])+ 2*np.array(GPRSigma[:-Days]), color='blue', alpha=0.3, label='Variation (2σ)')
plt.plot(FutureX[:len(PlotY) - Days], PlotY[:-Days], color="Black", label='actual test',  linewidth = 0.7)
plt.plot(FutureX[:len(ForecastGPR)- Days], ForecastGPR[:-Days], label='Predicted', color='blue')
plt.xlabel("Time (Days)")
plt.ylabel("Closing Price (USD)")
plt.title("GPR Predictions vs Actual Values (30% Test data)", y = -0.15)
plt.legend(['Variation', 'Actual Price', 'Predicted Price (GPR))'], bbox_to_anchor=(0., 1.02, 1., .102), loc=3, mode="expand", borderaxespad=0)
plt.show()
print("GPR R-squared", "{:.2%}".format(r2_score(TestY[:len(ForecastGPR)-Days], ForecastGPR[:len(ForecastGPR)-Days])))
print("The Root Mean Squared Error:", round(np.sqrt(mean_squared_error(TestY[:len(ForecastGPR)-Days], ForecastGPR[:len(ForecastGPR)-Days])),2))
print("The Mean Absolute Error:", round(mean_absolute_error(TestY[:len(ForecastGPR)-Days], ForecastGPR[:len(ForecastGPR)-Days]),2))




# Random Forest


RF = RandomForestRegressor(n_estimators=500, random_state=42, bootstrap=False)
RF.fit(ScaledTrainX, TrainY)

# Can't directly Fit the TestX in Model
# After each Row forecast we will be modyfing "zero" cells in the subsequent rows 

TextX = InitialTextX
TestingArray = []
ForecastRF = []
for i in range(1, len(TestX)):
    TestingArray = TestX.iloc[i].tolist()
    CheckingIndex = -Days
    while CheckingIndex < 0:
        if (TestingArray[CheckingIndex] <= 0):
            TestingArray[CheckingIndex] = float(NewTestY[CheckingIndex])

        CheckingIndex = CheckingIndex +1
    TestX.iloc[i] = TestingArray
    ScaledTestX = scaler.transform(TestX)

    ForecastRFVal = RF.predict(ScaledTestX[i].reshape(1, -1))
    NewTestY.append(ForecastRFVal)
    ForecastRF.append(float(ForecastRFVal))

fig = plt.figure(figsize = (8, 8))


#RF  | Plotting 30% Test vs Forecast
plt.plot(FutureX[:len(PlotY) - Days], PlotY[:-Days], color="Black", label='actual test',  linewidth = 0.7)
plt.plot(FutureX[:len(ForecastRF)- Days], ForecastRF[:-Days], label='Predicted', color='blue')
plt.xlabel("Time (Days)")
plt.ylabel("Closing Price (USD)")
plt.title("RF Predictions vs Actual Values (30% Test data)", y = -0.15)
plt.legend(['Actual Price', 'Predicted Price (RF))'], bbox_to_anchor=(0., 1.02, 1., .102), loc=3, mode="expand", borderaxespad=0)
plt.show()
print("RF R-squared", "{:.2%}".format(r2_score(TestY[:len(ForecastRF)-Days], ForecastRF[:len(ForecastRF)-Days])))
print("The Root Mean Squared Error:", round(np.sqrt(mean_squared_error(TestY[:len(ForecastRF)-Days], ForecastRF[:len(ForecastRF)-Days])),2))
print("The Mean Absolute Error:", round(mean_absolute_error(TestY[:len(ForecastRF)-Days], ForecastRF[:len(ForecastRF)-Days]),2))

