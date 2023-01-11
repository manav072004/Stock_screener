from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
from matplotlib import pyplot as plt
import pandas as pd
import time
import datetime
import streamlit as st


st.set_page_config(page_title="Future Prediction", page_icon="📈")

st.sidebar.write('Manav Patel')
st.sidebar.write('Email: manavpatel3674@gmail.com')
st.sidebar.markdown(f'''
<a href={'https://www.linkedin.com/in/manavpatel0720/'}><button style="background-color:darkgrey;">My Linkedin</button></a>
''',
unsafe_allow_html=True)

stock = st.text_input('Enter a stock symbol to analyze')
days_prediction = st.slider("Number of days of Prediction", 30, 730)
time_stamp = datetime.datetime.now()
time_stamp = time_stamp.strftime("%Y-%m-%d")
time_stamp = time_stamp.split('-')
p1 = int(time.mktime(datetime.datetime(2015, 1, 1, 23, 59).timetuple()))
p2 = int(time.mktime(datetime.datetime(int(time_stamp[0]), int(time_stamp[1]), int(time_stamp[2]), 23, 59).timetuple()))
interval = '1d'
url_string = f'https://query1.finance.yahoo.com/v7/finance/download/{stock}?period1={p1}&period2={p2}&interval={interval}&events=history&includeAdjustedClose=true'
try:
    df = pd.read_csv(url_string)
except:
    st.stop()

df_train = df[['Date', 'Close']]
df_train = df_train.rename(columns={"Date":"ds", "Close": "y"})

m = Prophet(daily_seasonality=True)
m.fit(df_train)
future = m.make_future_dataframe(periods=days_prediction)
forecast = m.predict(future)

st.subheader('Forecast Data')
st.write(forecast)


st.write('Forecast Data')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write('forecast Components')
fig2 = m.plot_components(forecast)
st.write(fig2)



