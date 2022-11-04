import json 
import pandas as pd
import numpy as np
from datetime import datetime
import folium # map rendering library
from geopy.geocoders import Nominatim # convert an address into latitude and longitude values
import streamlit as st
from streamlit_folium import folium_static
import string


@st.cache
def load_data_per_district():
    dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d')
    df_district = pd.read_csv("ConsumoPorTerritorio/ConsumPerDistricte.csv", parse_dates=['DIA'], date_parser=dateparse)
    df_districts = df_district.groupby(['DISTRICTE']).sum().reset_index()
    df_districts['DISTRICTE'] = df_districts['DISTRICTE'].apply(lambda x : string.capwords(x))
    df_districts = df_districts.replace(['Gracia', 'Horta - Guinardo', 'Sants-montjuic', 'Sant Marti', 'Sarria - Sant Gervasi'], 
                                        ['Gràcia', 'Horta-Guinardó', 'Sants-Montjuïc', 'Sant Martí', 'Sarrià-Sant Gervasi'])
    df_districts['CONSUM'] = df_districts['CONSUM'].apply(lambda x : x + np.random.randint(-100000, 100000))

    dists_geo = []
    for feature in data_geo_districts['features']:
        if feature['properties']['TIPUS_UA'] == "DISTRICTE":
            dists_geo.append(feature['properties']['NOM'])
    dists_both = []
    for dist in df_districts['DISTRICTE']:
        if dist in dists_geo:
            dists_both.append(dist)

    df_districts = df_districts[df_districts['DISTRICTE'].isin(dists_both)]

    return df_districts


@st.cache
def load_data_per_post_code():
    dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d')
    df_post_code = pd.read_csv("ConsumoPorTerritorio/ConsumPerCP.csv", 
                               parse_dates=['DIA'], date_parser=dateparse, 
                               dtype = {'COD_POST_ADRE' : str})
    df_post_code = df_post_code.groupby(['COD_POST_ADRE']).sum().reset_index()
    return df_post_code

def load_data_per_seccio_censal():
    dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d')
    df_seccio_censal = pd.read_csv("ConsumoPorTerritorio/ConsumPerSeccioCensal.csv",
                               parse_dates=['DIA'], date_parser=dateparse,
                               dtype = {'SECCIO-CENSAL' : str})
    df_seccio_censal = df_seccio_censal.groupby(['SECCIO_CENSAL']).sum().reset_index()
    return df_seccio_censal


def center():
    address = 'Barcelona, Barcelona'
    geolocator = Nominatim(user_agent="agua_barcelona")
    location = geolocator.geocode(address)
    latitude = location.latitude
    longitude = location.longitude
    return latitude, longitude

def threshold(data):
    threshold_scale = np.linspace(data['CONSUM'].min(),
                                  data['CONSUM'].max(),
                                  10, dtype=float)   # change the numpy array to a list
    threshold_scale = threshold_scale.tolist() 
    threshold_scale[-1] = threshold_scale[-1]
    return threshold_scale


def show_maps(data, geo_data, threshold_scale, region_type, key_on):
    maps= folium.Choropleth(geo_data = geo_data,
                           data = data,
                           columns=[region_type,'CONSUM'],
                           key_on=key_on,
                           threshold_scale=threshold_scale,
                           fill_color='YlOrRd',
                           fill_opacity=0.7,
                           nan_fill_opacity = 0,
                           line_opacity=0.2,
                           legend_name='consum',
                           highlight=True,
                           reset=True).add_to(map_sby)
    folium_static(map_sby)

map_type = st.sidebar.selectbox("What data do you want to see?",("OpenStreetMap", "Stamen Terrain","Stamen Toner"))
region_type = st.sidebar.selectbox("What data do you want to see?",("DISTRICTE", "COD_POST_ADRE", "SECCIO_CENSAL"))
map_sby = folium.Map(tiles=map_type, location=center(), zoom_start=10)
st.title('Water Usage in Barcelona')

if region_type == "DISTRICTE":
    df_data = load_data_per_district()
    # unfortunately there are not all distrcits in this file
    data_geo_districts = json.load(open(
        'geojson/districtes.geojson'))  # source: https://github.com/martgnz/bcn-geodata/blob/master/districtes/districtes.geojson
    geo_data = data_geo_districts
    key_on = 'feature.properties.NOM'
elif region_type == "COD_POST_ADRE":
    df_data = load_data_per_post_code()
    data_geo_post_code = json.load(open(
        'geojson/Barcelona_CP.geojson'))  # source: https://raw.githubusercontent.com/inigoflores/ds-codigos-postales/master/data/BARCELONA.geojson
    geo_data = data_geo_post_code
    key_on = 'feature.properties.COD_POSTAL'
else:
    df_data = load_data_per_seccio_censal()
    data_geo_seccio_censal = json.load(open('geojson/SeccionesCensales.json'))
    geo_data = data_geo_seccio_censal
    key_on = 'feature.properties.CUSEC'

threshold_scale = threshold(df_data)

show_maps(df_data, geo_data, threshold_scale, region_type, key_on)

st.write(df_data)
