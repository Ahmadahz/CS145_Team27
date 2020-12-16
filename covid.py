# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 16:24:58 2020

@author: Leighton
"""

import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn import neural_network
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import statsmodels.api as sm
import pmdarima as pm

# Reads the data from CSV files, each attribute column can be obtained via its name, e.g., y = data['y']
def getDataframe(filePath):
    data = pd.read_csv(filePath)
    return data

def to_csv(predicted_cases, predicted_deaths, outfile):   
    ids = np.arange(0, predicted_cases.shape[0])
    id_col = np.array(ids)
    
    df = pd.DataFrame(list(zip(id_col, predicted_cases, predicted_deaths)),
                      columns=('ForecastID', 'Confirmed', 'Deaths'))
    df.to_csv(path_or_buf=outfile, index=False)
    
# @param predicted, actual: numpy arrays of the predicted and target values
def MAPE(predicted, actual):
    print(predicted.shape, actual.shape)
    assert predicted.shape == actual.shape
    n = predicted.shape[0]
    return np.sum(np.abs((predicted - actual) / actual)) / n

class CovidClassifier:
    def __init__(self):
        pass
    
    def load_data(self, train_path, validation_path, test_path):
        dropped_cases_features = ['Province_State', 'Date',
                                  ]
        
        dropped_deaths_features = ['Province_State', 'Date',
                                   ]
        
        self.train_data = getDataframe(train_path)
        self.train_data = self.train_data.loc['0':'5549']
        
        self.train_cases_features = self.train_data.drop(dropped_cases_features, axis = 1).to_numpy()
        self.train_cases_features = np.nan_to_num(self.train_cases_features, nan=0.0)
        
        self.train_death_features = self.train_data.drop(dropped_deaths_features, axis = 1).to_numpy()
        self.train_death_features = np.nan_to_num(self.train_death_features, nan=0.0)
        
        self.train_cases = self.train_data['Confirmed'].to_numpy()
        self.train_deaths = self.train_data['Deaths'].to_numpy()
        
        """Validation Data"""
        self.val_data = getDataframe(validation_path)
        self.val_cases_features = self.val_data.drop(dropped_cases_features, axis = 1).to_numpy()
        self.val_cases_features = np.nan_to_num(self.val_cases_features, nan=0.0)
        
        self.val_death_features = self.val_data.drop(dropped_deaths_features, axis = 1).to_numpy()    
        self.val_death_features = np.nan_to_num(self.val_death_features, nan=0.0)
        
        self.val_cases = self.val_data['Confirmed'].to_numpy()
        self.val_deaths = self.val_data['Deaths'].to_numpy()
        
        """Test Data"""
        self.test_data = getDataframe(test_path)
        self.test_cases = self.test_data['Confirmed'].to_numpy()
        self.test_deaths = self.test_data['Deaths'].to_numpy()
        
    def train(self, classifier, target):    
        if target == 'cases':
            classifier.fit(self.train_cases_features, y=self.train_cases)
        elif target == 'deaths':
            classifier.fit(self.train_death_features, y=self.train_deaths)
        else:
            print('Invalid target specified')
            
    def predict(self, classifier, target):
        if target == 'cases':
            self.predicted_cases = np.rint(classifier.predict(self.val_cases_features))
            print(f'Shape of predicted cases: {self.predicted_cases.shape}')
            print(self.predicted_cases)
        elif target == 'deaths':
            self.predicted_deaths = np.rint(classifier.predict(self.val_death_features))
            print(f'Shape of predicted deaths: {self.predicted_deaths.shape}')
            print(self.predicted_deaths)
        else:
            print('Invalid target specified')
            
    def forecast(self, target, train_type):    
        round1file = './ucla2020-cs145-covid19-prediction/train.csv'
        round2file = './ucla2020-cs145-covid19-prediction/train_round2_current.csv'
        forecast_n = 0
        
        testfile = ""
        if train_type == 2:
            testfile = round2file
            forecast_n = 180
        else:
            testfile = round1file
            forecast_n = 26
            
        if target == 'cases':                     
            state_cases = getDataframe(testfile)
            state_cases = state_cases[['Province_State', 'Date', 'Confirmed']]
            state_cases['Date'] = pd.to_datetime(state_cases['Date'])
            state_cases = state_cases.groupby('Province_State')
            
            self.predicted_cases = []
            i = 1
            for case in state_cases.groups:
                print(i)
                group = state_cases.get_group(case)
                              
                # Best round1: (1,1,1), (2,1,1,12)
                # Best round2: 
                order = (1,1,1)
                seasonal_order = (1,1,1,12)
                
                model = sm.tsa.SARIMAX(group['Confirmed'],
                            order=order, seasonal_order=seasonal_order,
                            dates=group['Date'], freq='D')
                
                results = model.fit()
                predicted = np.rint(np.asarray(results.forecast(forecast_n)))
                self.predicted_cases.append(predicted)
                print(f'Length: {len(self.predicted_cases)}')
                
                i += 1
        elif target == 'deaths':
            state_deaths = getDataframe(testfile)
            state_deaths = state_deaths[['Province_State', 'Date', 'Deaths']]
            state_deaths['Date'] = pd.to_datetime(state_deaths['Date'])
            state_deaths = state_deaths.groupby('Province_State')
            
            self.predicted_deaths = []
            i = 1
            for case in state_deaths.groups:
                print(i)
                
                group = state_deaths.get_group(case)
                
                # Best round1: (2,1,2), (1,1,0,12)
                # Best round2: 
                order = (2,1,2)
                seasonal_order = (1,1,0,12)
                
                model = sm.tsa.SARIMAX(group['Deaths'],
                            order=order, seasonal_order=seasonal_order,
                            dates=group['Date'], freq='D')
                
                results = model.fit()
                predicted = np.rint(np.asarray(results.forecast(forecast_n)))
                self.predicted_deaths.append(predicted)
                print(f'Length: {len(self.predicted_deaths)}')
                
                i += 1
        else:
            print('Invalid target specified')

    def getMAPE(self, target):
        if target == 'cases':
            return MAPE(self.predicted_cases, self.test_cases)
        elif target == 'deaths':
            return MAPE(self.predicted_deaths, self.test_deaths)
        else:
            print('Invalid target specified')

    def normalize(self):
        preprocessing.scale(self.train_cases_features)
        preprocessing.scale(self.val_cases_features)
        
        preprocessing.scale(self.train_death_features)
        preprocessing.scale(self.val_death_features)














            