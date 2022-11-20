import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import utils
import prophet

df_covid = pd.read_csv("data/covid_data.csv")
# Solo España
Ddf_covidata = df_covid.query("countriesAndTerritories == 'Spain'")
df_covid = df_covid.reset_index()

print(df_covid.info())

# Solo fecha y casos
df_covid = df_covid[['dateRep','cases']]

# Solo hasta 2021
df_covid.dateRep = pd.to_datetime(df_covid.dateRep)
df_covid = df_covid.loc[df_covid['dateRep'] < pd.to_datetime('2022-01-01'),:]
df_covid.columns = ['ds', 'covid']
df_covid = df_covid.sort_values(by = ['ds'])
print(df_covid)

dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d')
dateparse_slash = lambda x: datetime.strptime(x, '%d/%m/%Y')
df_economica = pd.read_csv("data/consumo/AEDataConsum_agrupat.csv", parse_dates=['FECHA'], date_parser=dateparse)
df_economica = df_economica.groupby(['FECHA']).sum().reset_index()

df_economica = df_economica.rename({'FECHA' : 'ds', 'CONSUMO': 'y'}, axis='columns')

df_economica_covid = df_economica.join(df_covid.set_index('ds'), on='ds')
df_economica_covid['covid'] = df_economica_covid['covid'].fillna(0)

print(df_economica_covid)

prophet_economia_covid, forecast_economia_covid = utils.analyze(df_economica_covid, 'consum economico - covid')
#prophet_economia, forecast_economia = utils.analyze(df_economica, 'consum economico')

prophet_economia_covid.plot(forecast_economia_covid)
plt.show()

