import pandas as pd
from prophet import Prophet
from datetime import datetime
import matplotlib.pyplot as plt
import time
from prophet.plot import plot_plotly, plot_components_plotly

dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d')
dateparse_slash = lambda x: datetime.strptime(x, '%d/%m/%Y')
df_district = pd.read_csv("ConsumoPorTerritorio/ConsumPerDistricte.csv", parse_dates=['DIA'], date_parser=dateparse)
df_districts = df_district.groupby(['DIA']).sum().reset_index()

print(df_districts)

df_districts = df_districts.rename({'DIA': 'ds', 'CONSUM': 'y'}, axis='columns')

prophet = Prophet()
prophet.fit(df_districts)

future = prophet.make_future_dataframe(periods=365)
future.tail()

forecast = prophet.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

fig1 = prophet.plot(forecast)
fig1.show()
plt.show()

fig2 = prophet.plot_components(forecast)
plt.show()

# source for public holidays: https://datos.gob.es/es/catalogo/a09002970-festivos-generales-de-cataluna
df_festivos = pd.read_csv("Festius_generals_de_Catalunya.csv", parse_dates=['Data'], date_parser=dateparse_slash)
df_festivos = df_festivos[df_festivos['Any'] > 2018]


def is_public_holiday(date):
    if date in df_festivos['Data']:
        return 1
    elif date.weekday == 6:
        return 1
    else:
        return 0

df_districts['public_holiday'] = df_districts['ds'].apply(is_public_holiday)

m = Prophet()
m.add_regressor('public_holiday')
m.fit(df_districts)

future['public_holiday'] = future['ds'].apply(is_public_holiday)

forecast = m.predict(future)
fig = m.plot_components(forecast)
fig.show()

plt.show()

