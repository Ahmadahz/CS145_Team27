import numpy as np
import pandas as pd
import glob

# Displays full dataframe table in terminal
#pd.set_option('display.max_rows', None)

# I wasn't sure how to parse only the training data sets, so I moved all the nontraining data into a different folder
path = r'./Data/csse_covid_19_daily_reports_us'
all_files = sorted(glob.glob(path + "/*.csv"))
li = []
for file_name in all_files:
    temp_df = pd.read_csv(file_name, usecols=['Province_State', 'Confirmed', 'Deaths'])
    li.append(temp_df)
df = pd.concat(li, axis=0, ignore_index=True)

# List of provinces in the .csv files that aren't part of the 50 states for which we are predicting cases for
non_states = ['American Samoa', 'Diamond Princess', 'District of Columbia', 'Grand Princess', 'Guam', 'Northern Mariana Islands', 'Puerto Rico', 'Recovered', 'Virgin Islands']

for non_state in non_states:
    index = df[df['Province_State'] == non_state].index
    df.drop(index, inplace=True)
df.reset_index(drop=True, inplace=True)

# Transforms dataframe into a 3D array indexed by (state_id, date, type of information(cases/deaths)
num_dates = int(len(df.index)/50)
data_set = np.zeros(shape=(50, num_dates, 2))
for i in range(len(df.index)):
    data_set[i%50][int(i/50)][0] = df.iloc[i, 1]    
    data_set[i%50][int(i/50)][1] = df.iloc[i, 2]
    
#for i in range(50):
    #for j in range(num_dates):
        #print('i=', i, "j=", j)
        #print('Confirmed:', data_set[i][j][0], 'Deaths', data_set[i][j][1])
        
# y = a(r^t) where t is the number of days since April 12th 2020
a_confirmed, r_confirmed, a_deaths, r_deaths = ([] for i in range(4))
for i in range(50):
    a_confirmed = np.append(a_confirmed, data_set[i][0][0])
    r_confirmed = np.append(r_confirmed, 0)
    a_deaths = np.append(a_deaths, data_set[i][0][1])
    r_deaths = np.append(r_deaths, 0)
    for j in range(num_dates-1):
        if (data_set[i][j][0] != 0):
            r_confirmed[i] = r_confirmed[i] + data_set[i][j+1][0]/data_set[i][j][0]
        else:
            r_confirmed[i] = r_confirmed[i] + r_confirmed[i]/(max(1, j))
        if (data_set[i][j][1] != 0):
            r_deaths[i] = r_deaths[i] + data_set[i][j+1][1]/data_set[i][j][1]
        else: 
            r_deaths[i] = r_deaths[i] + r_confirmed[i]/(max(1, j))
r_confirmed = np.true_divide(r_confirmed, num_dates-1)
r_deaths = np.true_divide(r_deaths, num_dates-1)

# Special case since Wyoming had 0 deaths on April 12th
a_deaths[49] = 1
    
#for i in range(50):
    #print("i:", i, "a_confirmed:", a_confirmed[i], "r_confirmed:", r_confirmed[i], "a_deaths:", a_deaths[i], "r_deaths:", r_deaths[i])
    
# Calculates MAPE on training data
APE = 0
count = 0
for i in range(50):
    for j in range(num_dates):
        predicted_confirmed = a_confirmed[i] * (r_confirmed[i]**j)
        predicted_deaths = a_deaths[i] * (r_deaths[i]**j)
        if (data_set[i][j][0] != 0):
            APE += abs(predicted_confirmed - data_set[i][j][0])/data_set[i][j][0]
            count += 1
        else:
            #print('Zero for i:', i, 'j:', j, 'k: 0')
            pass
        if (data_set[i][j][1] != 0):
            APE += abs(predicted_deaths - data_set[i][j][1])/data_set[i][j][1]
            count += 1
        else:
            #print('Zero for i', i, 'j:', j, 'k: 1')
            pass
        #print('i', i, 'j:', j)
        #print('predicted_confirmed:', predicted_confirmed)
        #print('actual_confirmed:', data_set[i][j][0])
        #print('MPE for confirmed:', abs(predicted_confirmed - data_set[i][j][0])/data_set[i][j][0])
        #print('predicted_deaths:', predicted_deaths)
        #print('actual_deaths:', data_set[i][j][1])
MAPE = APE/(count)
print('The MAPE score is:', MAPE)
print('The MAPE score was computed using:', count, 'samples') 

#print(df)
#print('num_dates: ', num_dates)
#print(df.iloc[7099,2])