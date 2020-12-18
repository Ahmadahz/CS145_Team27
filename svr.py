import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
import csv


def handleDates(x):
    print(x)
    y = x[0].split('-')
    if int(y[0]) in [1,3,5,7,8,10,12]:
        return int(y[0]) * 31 + int(y[1])
    elif int(y[0]) == 2:
        return int(y[0]) * 29 + int(y[1])
    else:
        return int(y[0]) * 30 + int(y[1]) 


def plotConfirmed(state, result, train, test):
    trainStates = train.iloc[stea::50, :]
    X = np.array([[handleDates(x)] for x in trainStates.iloc[:, 0:1].values])
    y = trainStates.iloc[:, 1].values
    # Plotting
    plt.plot(X, y)
    testStates = test.iloc[::50, :]
    testDates = np.array([[handleDates(x)] for x in testStates.iloc[:, 0:1].values])
    testConfirmed = result[state]
    plt.plot(testDates, testConfirmed)
    plt.show()

def cases(state, train, test):
    trainState = train.iloc[state::50, :]
    dates = np.array([[handleDates(x)] for x in trainState.iloc[:, 0:1].values])
    confirmedCases = trainState.iloc[:, 1].values

    newCases = np.copy(confirmedCases)
    orig_state = newCases[len(y) - 1]
    for i in range(1, len(y)):
        newCases[len(copy_y) - i] = newCases[len(newCases) - i] - newCases[len(newCases) - i - 1]
    newCases[0] = newCases[1]

    
    reg = SVR(C=2000, kernel='rbf')
    reg.fit(X, newCases)

    testStates = test.iloc[::50, :] 
    testDates = np.array([[handleDates(x)] for x in testStates.iloc[:, 0:1].values])
    testCases = reg.predict(testDates)

    testCases[0] = orig_state
    for i in range(1, len(testCases)):
        testCases[i] += testCases[i - 1]
    
    return predicted_y


def deaths(state, train, test):

    trainState = train.iloc[state::50, :]
    dates = np.array([[handleDates(x)] for x in trainState.iloc[:, 0:1].values])
    confirmedDeaths = trainState.iloc[:, 1].values

    reg = SVR(C=1000)
    reg.fit(dates, confirmedDeaths)

    testState = test.iloc[::50, :]
    testDates = np.array([[handleDates(x)] for x in testState.iloc[:, 0:1].values])
    testCases = reg.predict(test_x)

    temp = np.floor(len(testCases) / 1.5)
    newreg = LinearRegression()
    newX = dates[len(dates) - 14:]
    newY = confirmedDeaths[len(confirmedDeaths) - 14:]
    newreg.fit(newX, newY)
    predicted_y_2 = newreg.predict(testDates)


    diff = newY[-1] - 2 * predicted_y_2[0] + predicted_y_2[1]
    predicted_y_2 += diff
    if testCases[-1] < testCases[int(temp)] or predicted_y_2[-1] > predicted_y[-1]:
        predicted_y=predicted_y_2
    return predicted_y


data = pd.read_csv("train.csv")
data = data.fillna(data.mean())
deathsData = data[['Date', 'Deaths']]
print(deathsData)


data = data[['Date', 'Confirmed']]
testData = pd.read_csv("test.csv")
testData = test_data[['Date']]

dead = np.array([deaths(i, deathsData, test_data) for i in range(50)])
confirmed = np.array([cases(i, data, test_data) for i in range(50)])

plotConfirmed(15, confirmed, data, testData)
plotConfirmed(15, dead, data_dead, testData)
