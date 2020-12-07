CS145 Team 27

./Data/csse_covid_19_daily_reports_us/*.csv contains the data used for the naive_exponential_model.py file

./Data2/train_round2/csv contains the data used for the time_series_analysis.ipynb



To generate the predictions for September using the naive exponential model:
$ python3 naive_exponential_model.py

To view the analysis/comparison of the AR/MA/ARIMA models:
$ jupyter-notebook
(Click into 'time_series_analysis.ipynb')
(Run the code blocks sequentially; make sure all imported libraries are present)