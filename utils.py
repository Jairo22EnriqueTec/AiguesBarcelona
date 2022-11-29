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

import matplotlib.pyplot as plt


def CalculateFences(T):
    upper = int(T.y.quantile([0.75]))  + 1.5 * (int(T.y.quantile([0.75])) - int(T.y.quantile([0.25])))
    lower = int(T.y.quantile([0.25]))  - 1.5 * (int(T.y.quantile([0.75])) - int(T.y.quantile([0.25])))
    return max(lower, 0), upper


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

def get_covid_prophet(df):
    df2 = df.copy()
    df2['pre_covid'] = pd.to_datetime(df2['ds']) < pd.to_datetime('2020-03-21')
    df2['post_covid'] = ~df2['pre_covid']

    lockdowns = pd.DataFrame([
        {'holiday': 'lockdown_1', 'ds': '2020-03-21', 'lower_window': 0, 'ds_upper': '2020-06-06'},
        {'holiday': 'lockdown_2', 'ds': '2020-07-09', 'lower_window': 0, 'ds_upper': '2020-10-27'},
        {'holiday': 'lockdown_3', 'ds': '2021-02-13', 'lower_window': 0, 'ds_upper': '2021-02-17'},
        {'holiday': 'lockdown_4', 'ds': '2021-05-28', 'lower_window': 0, 'ds_upper': '2021-06-10'},
    ])
    for t_col in ['ds', 'ds_upper']:
        lockdowns[t_col] = pd.to_datetime(lockdowns[t_col])
    lockdowns['upper_window'] = (lockdowns['ds_upper'] - lockdowns['ds']).dt.days

    m4 = Prophet(holidays=lockdowns, weekly_seasonality=False)

    m4.add_seasonality(
        name='weekly_pre_covid',
        period=7,
        fourier_order=3,
        condition_name='pre_covid',
    )
    m4.add_seasonality(
        name='weekly_post_covid',
        period=7,
        fourier_order=3,
        condition_name='post_covid',
    );

    m4 = m4.fit(df2)

    future4 = m4.make_future_dataframe(periods=366)
    future4['pre_covid'] = pd.to_datetime(future4['ds']) < pd.to_datetime('2020-03-21')
    future4['post_covid'] = ~future4['pre_covid']

    forecast4 = m4.predict(future4)
    m4.plot(forecast4)
    plt.axhline(y=0, color='red')
    plt.title('Lockdowns as one-off holidays + Conditional weekly seasonality')
    plt.show()

    fig = plot_components_plotly(m4, forecast4)
    fig.update_layout(title = f"Componentes para algo")
    fig.show()

    fig = m4.plot(forecast4)
    a = add_changepoints_to_plot(fig.gca(), m4, forecast4)
    fig.show()

    return m4


def analyze(Temp, name, gt_0 = True, max_ = None, days = 365, prior = 0.05, monthly = False):
    """
    In:
    
    Temp[Pandas DataFrame] - with columns ds and y
    name[str] - name of the variable 
    gt_0[bool] - 'greater than 0' True will filter the data
    max_[int or float] - max value to filter. If None, it does nothing
    days[int] - number of days to extrapole
    """
    
    
   
    Temp = Temp[Temp.y > gt_0]
    
    if type(max_) == int or type(max_) == float:
        Temp = Temp[Temp.y < max_]


    #m = getProphet(prior, monthly)
    m = getProphet()
    #m.add_regressor('covid')
    m.fit(Temp)
    future = m.make_future_dataframe(periods = 15)
    forecast = m.predict(future)
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

    print("before error")

    f = plot_components_plotly(m, forecast)
    f.update_layout(title = f"Componentes para {name}")
    f.show()
    
    fig = m.plot(forecast)
    a = add_changepoints_to_plot(fig.gca(), m, forecast)
    fig.show()

    return m, forecast


