#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objects as go
from prophet.plot import add_changepoints_to_plot
from prophet.plot import plot_plotly, plot_components_plotly
import warnings
warnings.filterwarnings("ignore")


def getProphet(prior = 0.05, monthly = False):
    """
    Return a Prophet type compilated with holidays 
    """
    
    holidays = pd.read_csv("data/Festius_generals_de_Catalunya.csv")
    holidays = holidays.query("Any > 2018")
    holidays = holidays[['Data', 'Nom del festiu']]
    holidays.columns = ['ds', 'holiday']
    lockdowns = pd.DataFrame([
    {'holiday': 'lockdown_1', 'ds': '2020-03-21', 'lower_window': 0, 'ds_upper': '2020-06-06'},
    {'holiday': 'lockdown_2', 'ds': '2020-07-09', 'lower_window': 0, 'ds_upper': '2020-10-27'},
    {'holiday': 'lockdown_3', 'ds': '2021-02-13', 'lower_window': 0, 'ds_upper': '2021-02-17'},
    {'holiday': 'lockdown_4', 'ds': '2021-05-28', 'lower_window': 0, 'ds_upper': '2021-06-10'},
    ])
    for t_col in ['ds', 'ds_upper']:
        lockdowns[t_col] = pd.to_datetime(lockdowns[t_col])
    lockdowns['upper_window'] = (lockdowns['ds_upper'] - lockdowns['ds']).dt.days
    
    
    if monthly:
        return Prophet(changepoint_prior_scale = prior, holidays = lockdowns, seasonality_mode='multiplicative')
    else:
        return Prophet(changepoint_prior_scale = prior, holidays = lockdowns)

def analyze(Temp, name, gt_0 = True, max_ = None, days = 365, prior = 0.05, monthly = False):
    """
    In:
    
    Temp[Pandas DataFrame] - with columns ds and y
    name[str] - name of the variable 
    gt_0[bool] - 'greater than 0' True will filter the data
    max_[int or float] - max value to filter. If None, it does nothing
    days[int] - number of days to extrapole
    """
    
    
    if gt_0:
        Temp = Temp[Temp.y > 0]
    
    if type(max_) == int or type(max_) == float:
        Temp = Temp[Temp.y < max_]
    
    m = getProphet(prior, monthly)
    m.fit(Temp)
    future = m.make_future_dataframe(periods = days)
    forecast = m.predict(future)
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

    f = plot_components_plotly(m, forecast)
    f.update_layout(title = f"Componentes para {name}")
    f.show()
    
    fig = m.plot(forecast)
    a = add_changepoints_to_plot(fig.gca(), m, forecast)
    fig.show()

