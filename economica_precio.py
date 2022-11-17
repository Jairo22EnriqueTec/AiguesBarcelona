import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import utils

dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d')
dateparse_slash = lambda x: datetime.strptime(x, '%d/%m/%Y')
df_economica = pd.read_csv("data/consumo/AEDataConsum_agrupat.csv", parse_dates=['FECHA'], date_parser=dateparse)
df_economica = df_economica.groupby(['FECHA']).sum().reset_index()

#print(len(np.unique(df_economica['FECHA'])))
#print(np.unique(df_economica['FECHA']))


df_precio = pd.read_csv("data/precio_agua.csv", parse_dates=['fecha_inicio', 'fecha_final'], date_parser=dateparse_slash)
df_precio['precio'] = pd.to_numeric(df_precio['precio'])
df_precio_tramo1 = df_precio[(df_precio['suministro_max'] == 9) & (df_precio['tipo'] == 'comercial/industrial')]


df_precio_comercial = pd.DataFrame()
df_precio_comercial['fecha'] = df_economica['FECHA']
df_precio_comercial['precio'] = np.nan

date_mapping = {}
for date in df_precio_tramo1['fecha_inicio']:
    date_mapping[date] = min(df_economica['FECHA'], key = lambda x : abs(x - date))

#print(date_mapping)

for date in df_precio_tramo1['fecha_inicio']:
    closest_date = date_mapping[date]
    df_precio_comercial['precio'][df_precio_comercial['fecha'] == closest_date] = df_precio_tramo1[df_precio_tramo1['fecha_inicio'] == date]['precio'].values[0]

df_precio_comercial['precio'] = df_precio_comercial['precio'].interpolate(method='pad')


df_economica = df_economica.join(df_precio_comercial.set_index('fecha'), on='FECHA')

df_economica = df_economica.rename({'FECHA': 'ds', 'CONSUMO': 'y'}, axis='columns')
df_economica = df_economica.drop(['Unnamed: 0', 'ID_CLIENTE'], axis=1)

print(df_economica)

utils.analyze(df_economica, 'price')