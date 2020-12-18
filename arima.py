import numpy as np
import pandas as pd
import sklearn 
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import scipy
import math
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)

def arima(train, test):
    train = pd.read_csv(train)
    test = pd.read_csv(test)
    statesdata = {}
    states = pd.Series.unique(train['Province_State'])
    num_states = len(states)
    for s in states:
        statesdata[s] = train.loc[train['Province_State'] == s ,:]

    predictions = {}
    for s in states:
        temp = statesdata[s].reset_index()
        confirmed_cases = temp['Confirmed'].values
        deaths = temp['Deaths'].values
        
        caseModel = SARIMAX(confirmed_cases, order=(3,2,1), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        predictedCases = caseModel.forecast(26)

        deathModel = SARIMAX(deaths, order=(4,2,3), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        predictedDeaths = deathModel.forecast(26)

        df = {'Confirmed': predictedCases, 'Deaths': predictedDeaths}
        predictions[s] = pd.DataFrame(df)

    state_order = test.loc[0:49,'Province_State']
    pred_cases = []
    pred_dead = []
    for i in range(0,26):
        for j in state_order:
            day = predictions[j].iloc[i]
            pred_cases.append(int(day['Confirmed']))
            pred_dead.append(int(day['Deaths']))

    test['Confirmed'] = pred_cases
    test['Deaths'] = pred_dead
    submission = test.drop(columns=['Province_State', 'Date'])
    submission.to_csv('arimatest.csv', index = False, header = True)


arima("train.csv", "test.csv")
