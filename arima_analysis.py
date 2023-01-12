from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
from pandas.plotting import autocorrelation_plot
from bokeh.plotting import figure, show
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
import math

income_df = pd.read_csv("BEA_CAINC1_converted.csv")
medincome = pd.pivot_table(income_df, values='CAINC1-3', index=['TimePeriod'], columns=['GeoName'])
employment_rate_df = pd.read_csv("BLS_LN_Rate_Series.csv")
employment_level_df = pd.read_csv("BLS_LN_Level_Series.csv")

# Extract individual series and the index/x values and then do arima, and plot that.
forecast_years = 6
start_year = min(medincome.index)
max_year = max(medincome.index)
predict_year = (max_year + 1) - forecast_years

# A quick exploration of autocorrelation since Arima leverages largely that
all_albany = medincome.loc[:, 'Albany, NY']
autocorrelation_plot(all_albany)
pyplot.show()

# Splitting data into test and train
train_a, test_a = medincome.loc[start_year:predict_year, 'Albany, NY'],    medincome.loc[predict_year:, 'Albany, NY']
train_b, test_b = medincome.loc[start_year:predict_year, 'Allegany, NY'],  medincome.loc[predict_year:, 'Allegany, NY']
train_c, test_c = medincome.loc[start_year:predict_year, 'Bronx, NY'],     medincome.loc[predict_year:, 'Bronx, NY']

# Fitting models
model_alb = ARIMA(train_a.values, order=(6,1,0))
model_all = ARIMA(train_b.values, order=(6,1,0))
model_brx = ARIMA(train_c.values, order=(6,1,0))

# Look at one particular model
predictions = list()
alb_fit = model_alb.fit()
residuals = alb_fit.resid
# Plot residuals
fg = figure(title="Residuals", width=800, height=600, tools="pan, reset, zoom_in, zoom_out, save")
fg.line(list(range(len(residuals))), residuals, line_color='blue')
show(fg)
# Show density plot of residuals
alb_fit.summary()
alb_rplot = pd.DataFrame(residuals).plot.kde()
pyplot.show()
# Get predictions, root mean squared error
predictions = alb_fit.forecast(forecast_years)
rmse = math.sqrt(mean_squared_error(test_a, predictions))

# Plot the two time series
import pdb; pdb.set_trace()
model_values = list()
model_values.extend(alb_fit.predict()[1:])
model_values.extend(alb_fit.forecast(forecast_years))
actual = all_albany.values
years = all_albany.index
fg = figure(title="Actual vs Model Albany, NY", width=800, height=600, tools="pan, reset, zoom_in, zoom_out, save")
fg.line(years, model_values, line_color='red')
fg.line(years, actual, line_color='blue')
show(fg)

