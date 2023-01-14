import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
import streamlit as st
import pandas as pd
import pandas_ta as ta
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ta.trend import MACD
from ta.momentum import StochasticOscillator
import re
import numpy as np
import datetime
import time
import yfinance as yf
import streamlit.components.v1 as components
from bs4 import BeautifulSoup as soup
from urllib.request import Request, urlopen
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
from matplotlib import pyplot as plt


st.set_page_config(layout="wide")
components.html("""<!-- TradingView Widget BEGIN -->
    <div class="tradingview-widget-container">
      <div class="tradingview-widget-container__widget"></div>
      <div class="tradingview-widget-copyright"><a href="https://in.tradingview.com/markets/" rel="noopener" target="_blank"><span class="blue-text">Markets today</span></a> by TradingView</div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-ticker-tape.js" async>
      {
      "symbols": [
        {
          "proName": "FOREXCOM:SPXUSD",
          "title": "S&P 500"
        },
        {
          "proName": "BITSTAMP:BTCUSD",
          "title": "Bitcoin"
        },
        {
          "proName": "BITSTAMP:ETHUSD",
          "title": "Ethereum"
        },
        {
          "description": "",
          "proName": "NASDAQ:TSLA"
        },
        {
          "description": "",
          "proName": "NASDAQ:META"
        },
        {
          "description": "",
          "proName": "SP:SPX"
        },
        {
          "description": "",
          "proName": "NASDAQ:GOOGL"
        },
        {
          "description": "",
          "proName": "NASDAQ:NFLX"
        },
        {
          "description": "",
          "proName": "NASDAQ:MSFT"
        },
        {
          "description": "",
          "proName": "PEPPERSTONE:XAUUSD"
        },
        {
          "description": "",
          "proName": "MCX:CRUDEOIL1!"
        },
        {
          "description": "",
          "proName": "MCX:COPPER1!"
        },
        {
          "description": "",
          "proName": "TVC:US10Y"
        },
        {
          "description": "",
          "proName": "NASDAQ:LCID"
        },
        {
          "description": "",
          "proName": "NASDAQ:RIVN"
        },
        {
          "description": "",
          "proName": "NYSE:PBR"
        }
      ],
      "showSymbolLogo": true,
      "colorTheme": "light",
      "isTransparent": false,
      "displayMode": "regular",
      "locale": "in"
    }
      </script>
    </div>
    <!-- TradingView Widget END -->""", height=50, width=1200)

def MACD_Strategy(data):
    macd = ta.macd(data['Close'])*100
    macd.rename(columns={'MACD_12_26_9': 'MACD', 'MACDh_12_26_9': 'Histogram', 'MACDs_12_26_9': 'Signal'}, inplace=True)
    data = pd.concat([data, macd], axis=1).reindex(data.index)
    
    Buy=[]
    Sell=[]
    Record=[]
    Buy_position = False
    Sell_position = False

    for i in range(len(data['Close'])):
        if i == 0:
            Buy.append(np.nan)
            Sell.append(np.nan)
        elif pd.notna(data['Histogram'][i-1]):
            if ((data['Histogram'][i-1] < 0) and (data['Histogram'][i] > 0)) and \
                ((data['MACD'][i] < 0) and (data['Signal'][i] < 0)):
                if Sell_position == True:
                    Buy.append(data['Low'][i] - data['Low'][i] * 0.003)
                    Sell.append(np.nan)
                    Buy_position = False
                    Sell_position = False
                    Record.append([i, data['Close'][i], 'Buy'])
                elif Buy_position == False:
                    Buy.append(data['Low'][i] - data['Low'][i] * 0.003)
                    Sell.append(np.nan)
                    Buy_position = True
                    Sell_position = False
                    Record.append([i, data['Close'][i], 'Buy'])
                else:
                    Buy.append(np.nan)
                    Sell.append(np.nan)
            elif ((data['Histogram'][i-1] > 0) and (data['Histogram'][i] < 0)) and \
                ((data['MACD'][i] > 0) and (data['Signal'][i] > 0)):
                if Buy_position == True:
                    Buy.append(np.nan)
                    Sell.append(data['High'][i] + data['High'][i] * 0.003)
                    Buy_position = False
                    Sell_position = False
                    Record.append([i, data['Close'][i], 'Sell'])
                elif Sell_position == False:
                    Buy.append(np.nan)
                    Sell.append(data['High'][i] + data['High'][i] * 0.003)
                    Buy_position = False
                    Sell_position = True
                    Record.append([i, data['Close'][i], 'Sell'])
                else:
                    Buy.append(np.nan)
                    Sell.append(np.nan)
            else:
                Buy.append(np.nan)
                Sell.append(np.nan)
        else:
            Buy.append(np.nan)
            Sell.append(np.nan)

    return Buy, Sell

def BB_Strategy(data):
    bb = ta.bbands(data['Close'], length = 20, std = 2)
    bb.rename(columns={'BBU_20_2.0': 'BBU', 'BBL_20_2.0': 'BBL'}, inplace=True)
    data = pd.concat([data, bb], axis = 1).reindex(data.index)

    Buy=[]
    Sell=[]
    Record=[]
    Buy_position = False
    Sell_position = False

    for i in range(len(data['Close'])):
        if i == 0:
            Buy.append(np.nan)
            Sell.append(np.nan)
        elif pd.notna(data['BBU'][i]):
            if (data['BBL'][i] > (data['Close'][i] + data['Open'][i])/2):
                if Sell_position == True:
                    Buy.append(data['Low'][i] - data['Low'][i] * 0.002)
                    Sell.append(np.nan)
                    Buy_position = False
                    Sell_position = False
                    Record.append([i, data['Close'][i], 'Buy'])
                elif Buy_position == False:
                    Buy.append(data['Low'][i] - data['Low'][i] * 0.002)
                    Sell.append(np.nan)
                    Buy_position = True
                    Sell_position = False
                    Record.append([i, data['Close'][i], 'Buy'])
                else:
                    Buy.append(np.nan)
                    Sell.append(np.nan)
            elif (data['BBL'][i] < (data['Close'][i] + data['Open'][i])/2):
                if Buy_position == True:
                    Buy.append(np.nan) 
                    Sell.append(data['High'][i] + data['High'][i] * 0.002)
                    Buy_position = False
                    Sell_position = False
                    Record.append([i, data['Close'][i], 'Sell'])
                elif Sell_position == False:
                    Buy.append(np.nan)
                    Sell.append(data['High'][i] + data['High'][i] * 0.002)
                    Buy_position = False
                    Sell_position = True
                    Record.append([i, data['Close'][i], 'Sell'])
                else:
                    Buy.append(np.nan)
                    Sell.append(np.nan)
            else:
                Buy.append(np.nan)
                Sell.append(np.nan)
        else:
            Buy.append(np.nan)
            Sell.append(np.nan)

    
    return Buy, Sell


def RSI_Strategy(data):
    data['RSI'] = ta.rsi(data['Close'], timeperiod=14)

    Buy=[]
    Sell=[]
    Record=[]
    Buy_position = False
    Sell_position = False

    for i in range(len(data['Close'])):
        if i == 0:
            Buy.append(np.nan)
            Sell.append(np.nan)
        elif pd.notna(data['RSI'][i-1]):
            if (data['RSI'][i-1] < 30) and (data['RSI'][i] > 30):
                if Sell_position == True:
                    Buy.append(data['Low'][i] - data['Low'][i] * 0.003)
                    Sell.append(np.nan)
                    Buy_position = False
                    Sell_position = False
                    Record.append([i, data['Close'][i], 'Buy'])
                elif Buy_position == False:
                    Buy.append(data['Low'][i] - data['Low'][i] * 0.003)
                    Sell.append(np.nan)
                    Buy_position = True
                    Sell_position = False
                    Record.append([i, data['Close'][i], 'Buy'])
                else:
                    Buy.append(np.nan)
                    Sell.append(np.nan)
            elif (data['RSI'][i-1] > 70) and (data['RSI'][i] < 70):
                if Buy_position == True:
                    Buy.append(np.nan)
                    Sell.append(data['High'][i] + data['High'][i] * 0.003)
                    Buy_position = False
                    Sell_position = False
                    Record.append([i, data['Close'][i], 'Sell'])
                elif Sell_position == False:
                    Buy.append(np.nan)
                    Sell.append(data['High'][i] + data['High'][i] * 0.003)
                    Buy_position = False
                    Sell_position = True
                    Record.append([i, data['Close'][i], 'Sell'])
                else:
                    Buy.append(np.nan)
                    Sell.append(np.nan)
            else:
                Buy.append(np.nan)
                Sell.append(np.nan)
        else:
            Buy.append(np.nan)
            Sell.append(np.nan)
    
    return Buy, Sell
    

global driver
def xpath_element(xpath):
    try:
        element = driver.find_element(By.XPATH, xpath)
    except NoSuchElementException:
        element = []

    return element

def real_time_price(stock_code):
    try:
        url = 'https://finance.yahoo.com/quote/'+stock_code+'?p='+stock_code+'&.tsrc=fin-srch'
        driver.get(url)
        
        xpath ='//*[@id="quote-header-info"]/div[3]/div[1]/div[1]'
        stock_price = xpath_element(xpath)
        price=[]
        change=[]

        if stock_price != []:
            stock_price_temp = stock_price.text.split()[0]
            if stock_price_temp.find('+')!=-1:
                price = stock_price_temp.split('+')[0]
                try:
                    change = '+' + stock_price_temp.split('+')[1] + ' ' + stock_price.text.split()[1]
                except IndexError:
                    change = ''
            elif stock_price_temp.find('-')!=-1:
                price = stock_price_temp.split('-')[0]
                try:
                    change = '-' + stock_price_temp.split('-')[1] + ' ' + stock_price.text.split()[1]
                except IndexError:
                    change = ''
            else:
                price, change = '', ''
        else:
            price, change = '', ''


        xpath = '//*[@id="quote-summary"]/div[1]'
        volume_temp = xpath_element(xpath)
        volume = []
        if volume_temp != []:

            #volume = volume_temp.text.split()[-4]
            for i, text in enumerate(volume_temp.text.split()):
                if text == 'Volume':
                    volume = volume_temp.text.split()[i+1]
                    break
                else:
                    volume = ''
        else:
            volume = ''
    except:
        stocks = yf.Ticker(stock_code)
        price = stocks.info['regularMarketPrice']
        volume = stocks.info['regularMarketVolume']
        temp = int(stocks.info['regularMarketPrice']) - int(stocks.info['previousClose'])
        percent = (float(temp) / float(stocks.info['previousClose'])) * 100
        change = f'{str(temp)} ({percent:.2f}%)'


    return price, change, volume



def check_convert_str_float(df):
    df[0] = float(df[0])
    if isinstance(df[2], str):
        df[2] = re.sub(",", "", df[2])
    df[2] = int(float(df[2]))
    return df

def sqmomentum():
    length = 20
    mult = 2
    length_KC = 20
    mult_KC = 1.5

    # calculate BB
    m_avg = data['Close'].rolling(window=length).mean()
    m_std = data['Close'].rolling(window=length).std(ddof=0)
    data['upper_BB'] = m_avg + mult * m_std
    data['lower_BB'] = m_avg - mult * m_std

    # calculate true range
    data['tr0'] = abs(data["High"] - data["Low"])
    data['tr1'] = abs(data["High"] - data["Close"].shift())
    data['tr2'] = abs(data["Low"] - data["Close"].shift())
    data['tr'] = data[['tr0', 'tr1', 'tr2']].max(axis=1)

    # calculate KC
    range_ma = data['tr'].rolling(window=length_KC).mean()
    data['upper_KC'] = m_avg + range_ma * mult_KC
    data['lower_KC'] = m_avg - range_ma * mult_KC

    # calculate bar value
    highest = data['High'].rolling(window = length_KC).max()
    lowest = data['Low'].rolling(window = length_KC).min()
    m1 = (highest + lowest)/2
    data['value'] = (data['Close'] - (m1 + m_avg)/2)
    fit_y = np.array(range(0,length_KC))
    data['value'] = data['value'].rolling(window = length_KC).apply(lambda x: 
                            np.polyfit(fit_y, x, 1)[0] * (length_KC-1) + 
                            np.polyfit(fit_y, x, 1)[1], raw=True)

    # check for 'squeeze'
    data['squeeze_on'] = (data['lower_BB'] > data['lower_KC']) & (data['upper_BB'] < data['upper_KC'])
    data['squeeze_off'] = (data['lower_BB'] < data['lower_KC']) & (data['upper_BB'] > data['upper_KC'])

    # buying window for long position:
    # 1. black cross becomes gray (the squeeze is released)
    long_cond1 = (data['squeeze_off'][-2] == False) & (data['squeeze_off'][-1] == True) 
    # 2. bar value is positive => the bar is light green k
    long_cond2 = data['value'][-1] > 0
    enter_long = long_cond1 and long_cond2

    # buying window for short position:
    # 1. black cross becomes gray (the squeeze is released)
    short_cond1 = (data['squeeze_off'][-2] == False) & (data['squeeze_off'][-1] == True) 
    # 2. bar value is negative => the bar is light red 
    short_cond2 = data['value'][-1] < 0
    enter_short = short_cond1 and short_cond2
    return data


@st.experimental_singleton
def get_driver():
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

options = Options()
options.add_argument('--disable-gpu')
options.add_argument('--headless')
driver = get_driver()


today = datetime.date.today()
year = today.year


ticker = st.text_input(
    'Enter a Stock Symbol to Analyze'
)

st.sidebar.write('Manav Patel')
st.sidebar.write('Email: manavpatel3674@gmail.com')
st.sidebar.markdown(f'''
<a href={'https://www.linkedin.com/in/manavpatel0720/'}><button style="background-color:darkgrey;">My Linkedin</button></a>
''',
unsafe_allow_html=True)


ma1 = 10
ma2 = 20
try:
    stock_val = yf.Ticker(ticker)
    name = stock_val.info['shortName']
    st.title(f"{name}")
except:
    pass

tab1, tab2 = st.tabs(['Chart', 'Latest Stock News'])


with tab2:
    st.markdown('### Latest News')
    try:
        val=ticker
        url = ("http://finviz.com/quote.ashx?t=" + val.lower())
        req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        webpage = urlopen(req).read()
        html = soup(webpage, "html.parser")
        news = pd.read_html(str(html), attrs = {'class': 'fullview-news-outer'})[0]
        links = []
        for a in html.find_all('a', class_="tab-link-news"):
            links.append(a['href'])

        # Clean up news dataframe
        news.columns = ['Date', 'News Headline']
        news['Article Link'] = pd.Series(links)
        news = news.set_index('Date')
        st.dataframe(news, width=1500)
    except:
        try:
            stocks = yf.Ticker(ticker)
            df = pd.DataFrame(stocks.news)
            df.drop(columns=['uuid'], axis=1, inplace=True)
            df.drop(columns=['providerPublishTime'], axis=1, inplace=True)
            df.drop(columns=['type'], axis=1, inplace=True)
            df.drop(columns=['relatedTickers'], axis=1, inplace=True)
            df.drop(columns=['thumbnail'], axis=1, inplace=True)
            st.dataframe(df)
        except:
            pass
            st.error('Symbol not found. Please try again')
        
       

    
    try:
        st.markdown('### Insider Trades')
        # Find insider table
        insider = pd.read_html(str(html), attrs = {'class': 'body-table'})[0]
        
        # Clean up insider dataframe
        insider = insider.iloc[1:]
        insider.columns = ['Trader', 'Relationship', 'Date', 'Transaction', 'Cost', '# Shares', 'Value ($)', '# Shares Total', 'SEC Form 4']
        insider = insider[['Date', 'Trader', 'Relationship', 'Transaction', 'Cost', '# Shares', 'Value ($)', '# Shares Total', 'SEC Form 4']]
        insider = insider.set_index('Date')
        st.write(insider)

    except Exception as e:
        try:
            stock_val = yf.Ticker(ticker)
            name = stock_val.info['shortName']
            st.write(f'Insider Trades unavailable for {name}')
        except:
            pass



with tab1:
    placeholder = st.empty()
    while True: 
        try:
            data = yf.download(tickers = ticker, period ='1d', interval = '1m')
        except:
            pass
            st.stop()
            
        vol = []
        price, change, vol = real_time_price(ticker)
        if price=='':
            st.stop()
        vol = ' '.join(map(str, [vol]))
        price = ' '.join(map(str, [price]))
        #declare figure
        data['Date'] = data.index
        data['Date'] = pd.to_datetime(data['Date'])
        data['RSI'] = ta.rsi(data['Close'], timeperiod=14)
        data['RSI'] = data['RSI'].fillna(0)
        data[f'{ma1}_ma'] = data['Close'].rolling(ma1).mean()
        data[f'{ma2}_ma'] = data['Close'].rolling(ma2).mean()
        # MACD
        macd = MACD(close=data['Close'], 
                    window_slow=26,
                    window_fast=12, 
                    window_sign=9)
        # stochastics
        stoch = StochasticOscillator(high=data['High'],
                                    close=data['Close'],
                                    low=data['Low'],
                                    window=14, 
                                    smooth_window=3)
        data['MACD Buy'], data['MACD Sell'] = MACD_Strategy(data) 
        data['RSI Buy'], data['RSI Sell'] = RSI_Strategy(data) 
        
        data['BB Buy'], data['BB Sell'] = BB_Strategy(data) 
        
        data = sqmomentum()

        colorsSq = []
        color=''
        for ind, val in enumerate(data['value']):
            if val >= 0:
                color = 'green'
                if val > data['value'][ind-1]:
                    color = 'lime'
            else:
                color = 'maroon'
                if val < data['value'][ind-1]:
                    color='red'
            colorsSq.append(color)

        with placeholder.container():
            
            kpi1, kpi2 = st.columns(2)
            # fill in those three columns with respective metrics or KPIs
            kpi1.metric(
                label="Price",
                value=price,
                delta=change
            )
            
            kpi2.metric(
                label="Volume",
                value=vol,
            )
        
            fig_col1, fig_col2 = st.columns((10,3), gap='large')
            with fig_col1:
                st.markdown("### Stock Chart")
                
                fig = make_subplots(rows=6, cols=1, shared_xaxes=True,
                                    vertical_spacing=0.05, 
                                    row_heights=[6, 2.0, 5, 2.0, 2.0, 2.0])
                # Plot OHLC on 1st subplot (using the codes from before)
                fig.add_trace(go.Candlestick(x=data['Date'],
                                            open=data['Open'],
                                            high=data['High'],
                                            low=data['Low'],
                                            close=data['Close'], 
                                            showlegend=False))
                # add moving average traces
                fig.add_trace(go.Scatter(x=data['Date'], 
                                        y=data[f'{ma1}_ma'], 
                                        line=dict(color='grey', width=2), 
                                        name='MA 5'))
                fig.add_trace(go.Scatter(x=data['Date'], 
                                        y=data[f'{ma2}_ma'], 
                                        line=dict(color='orange', width=2), 
                                        name='MA 20'))
                fig.add_trace(go.Scatter(mode='markers', x=data['Date'], visible=True,
                                        y=data['MACD Buy'], name='MACD Buy', marker=dict(size=8, symbol="triangle-up-open", line=dict(width=2, color="aqua")) 
                                        ))
                fig.add_trace(go.Scatter(mode='markers', x=data['Date'], visible=True,
                                        y=data['MACD Sell'], name='MACD Sell', marker=dict(size=8, symbol="triangle-down-open", line=dict(width=2, color="gold"))
                                        ))
                fig.add_trace(go.Scatter(mode='markers', x=data['Date'], visible=True,
                                        y=data['RSI Buy'], name='RSI Buy', marker=dict(size=8, symbol="hourglass-open", line=dict(width=2, color="aqua")) 
                                        ))
                fig.add_trace(go.Scatter(mode='markers', x=data['Date'], visible=True,
                                        y=data['RSI Sell'], name='RSI Sell', marker=dict(size=8, symbol="hourglass-open", line=dict(width=2, color="gold"))
                                        ))
                fig.add_trace(go.Scatter(mode='markers', x=data['Date'], visible=True,
                                        y=data['BB Buy'], name='BB Buy', marker=dict(size=8, symbol="diamond-open", line=dict(width=2, color="aqua")) 
                                        ))
                fig.add_trace(go.Scatter(mode='markers', x=data['Date'], visible=True,
                                        y=data['BB Sell'], name='BB Sell', marker=dict(size=8, symbol="diamond-open", line=dict(width=2, color="gold"))
                                        ))
                
                # Plot volume trace on 2nd row 
                colors = ['green' if row['Open'] - row['Close'] >= 0 
                        else 'red' for index, row in data.iterrows()]
                fig.add_trace(go.Bar(x=data['Date'], 
                                    y=data['Volume'],
                                    marker_color=colors
                                    ), row=2, col=1)
                fig.add_trace(go.Bar(x=data['Date'],
                                     y=data['value'],
                                     marker_color=colorsSq
                                     ), row=3, col=1)
                fig.add_trace(go.Scatter(mode='markers', x=data['Date'], visible=True,
                                        y=[0]*len(data), marker=dict(size=1, symbol="cross", line=dict(width=2, color=['lightgray' if s else 'black' for s in data['squeeze_off']])) 
                                        ), row=3, col=1)
                # Plot MACD trace on 3rd row
                colors = ['green' if val >= 0 
                        else 'red' for val in macd.macd_diff()]
                fig.add_trace(go.Bar(x=data['Date'], 
                                    y=macd.macd_diff(),
                                    marker_color=colors
                                    ), row=4, col=1)
                fig.add_trace(go.Scatter(x=data['Date'],
                                        y=macd.macd(),
                                        line=dict(color='lightblue', width=2)
                                        ), row=4, col=1)
                fig.add_trace(go.Scatter(x=data['Date'],
                                        y=macd.macd_signal(),
                                        line=dict(color='darkorange', width=1)
                                        ), row=4, col=1)
                # Plot stochastics trace on 4th row 
                fig.add_trace(go.Scatter(x=data['Date'],
                                        y=stoch.stoch(),
                                        line=dict(color='orange', width=2)
                                        ), row=5, col=1)
                fig.add_trace(go.Scatter(x=data['Date'],
                                        y=stoch.stoch_signal(),
                                        line=dict(color='blue', width=1)
                                        ), row=5, col=1)
                fig.add_hrect(y0=20, y1=80, line_width=0, fillcolor="lightblue", opacity=0.2, row=6, col=1)
                fig.add_trace(go.Scatter(x=data['Date'],
                                        y=data['RSI'],
                                        line=dict(color='purple', width=1)
                                        ), row=6, col=1)
                fig.add_hrect(y0=30, y1=70, line_width=0, fillcolor="violet", opacity=0.2, row=6, col=1)

                
                # update y-axis label
                fig.update_yaxes(title_text="Price", row=1, col=1)
                fig.update_yaxes(title_text="Volume", row=2, col=1)
                fig.update_yaxes(title_text="TTM Squeze Indicator", row=3, col=1)
                fig.update_yaxes(title_text="MACD", showgrid=False, row=4, col=1)
                fig.update_yaxes(title_text="Stoch", row=5, col=1)
                fig.update_yaxes(title_text="RSI", row=6, col=1)
                fig.update_xaxes(
                    rangeslider_visible=False,
                    rangebreaks=[
                            # NOTE: Below values are bound (not single values), ie. hide x to y
                        dict(bounds=["sat", "mon"]),  # hide weekends, eg. hide sat to before mon
                        dict(bounds=[16, 9.5], pattern="hour"), 
                        # hide hours outside of 9.30am-4pm
                        dict(values=[f"{year}-12-25", f"{year}-12-31", f"{year}-1-1"])  # hide holidays (Christmas and New Year's, etc)
                    ]
                )
                fig.update_layout(height=1200, width = 900, margin=go.layout.Margin(
                        l=20, #left margin
                        r=20, #right margin
                        b=20, #bottom margin
                        t=20  #top margin
                    ))
                st.plotly_chart(fig, height=1200, width=900)
            with fig_col2:
                st.markdown("### Market Overview")
                components.html("""<!-- TradingView Widget BEGIN -->
                    <div class="tradingview-widget-container">
                    <div class="tradingview-widget-container__widget"></div>
                    <div class="tradingview-widget-copyright"><a href="https://www.tradingview.com" rel="noopener" target="_blank"><span class="blue-text">Stocks</span></a> by TradingView</div>
                    <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-market-overview.js" async>
                    {
                    "colorTheme": "light",
                    "dateRange": "12M",
                    "showChart": true,
                    "locale": "en",
                    "largeChartUrl": "",
                    "isTransparent": false,
                    "showSymbolLogo": true,
                    "showFloatingTooltip": false,
                    "width": "270",
                    "height": "600",
                    "plotLineColorGrowing": "rgba(41, 98, 255, 1)",
                    "plotLineColorFalling": "rgba(41, 98, 255, 1)",
                    "gridLineColor": "rgba(240, 243, 250, 0)",
                    "scaleFontColor": "rgba(106, 109, 120, 1)",
                    "belowLineFillColorGrowing": "rgba(41, 98, 255, 0.12)",
                    "belowLineFillColorFalling": "rgba(41, 98, 255, 0.12)",
                    "belowLineFillColorGrowingBottom": "rgba(41, 98, 255, 0)",
                    "belowLineFillColorFallingBottom": "rgba(41, 98, 255, 0)",
                    "symbolActiveColor": "rgba(41, 98, 255, 0.12)",
                    "tabs": [
                        {
                        "title": "Indices",
                        "symbols": [
                            {
                            "s": "FOREXCOM:SPXUSD",
                            "d": "S&P 500"
                            },
                            {
                            "s": "FOREXCOM:NSXUSD",
                            "d": "US 100"
                            },
                            {
                            "s": "FOREXCOM:DJI",
                            "d": "Dow 30"
                            },
                            {
                            "s": "INDEX:NKY",
                            "d": "Nikkei 225"
                            },
                            {
                            "s": "INDEX:DEU40",
                            "d": "DAX Index"
                            },
                            {
                            "s": "FOREXCOM:UKXGBP",
                            "d": "UK 100"
                            }
                        ],
                        "originalTitle": "Indices"
                        },
                        {
                        "title": "Futures",
                        "symbols": [
                            {
                            "s": "CME_MINI:ES1!",
                            "d": "S&P 500"
                            },
                            {
                            "s": "CME:6E1!",
                            "d": "Euro"
                            },
                            {
                            "s": "COMEX:GC1!",
                            "d": "Gold"
                            },
                            {
                            "s": "NYMEX:CL1!",
                            "d": "Crude Oil"
                            },
                            {
                            "s": "NYMEX:NG1!",
                            "d": "Natural Gas"
                            },
                            {
                            "s": "CBOT:ZC1!",
                            "d": "Corn"
                            }
                        ],
                        "originalTitle": "Futures"
                        },
                        {
                        "title": "Bonds",
                        "symbols": [
                            {
                            "s": "CME:GE1!",
                            "d": "Eurodollar"
                            },
                            {
                            "s": "CBOT:ZB1!",
                            "d": "T-Bond"
                            },
                            {
                            "s": "CBOT:UB1!",
                            "d": "Ultra T-Bond"
                            },
                            {
                            "s": "EUREX:FGBL1!",
                            "d": "Euro Bund"
                            },
                            {
                            "s": "EUREX:FBTP1!",
                            "d": "Euro BTP"
                            },
                            {
                            "s": "EUREX:FGBM1!",
                            "d": "Euro BOBL"
                            }
                        ],
                        "originalTitle": "Bonds"
                        },
                        {
                        "title": "Stocks",
                        "symbols": [
                            {
                            "s": "AMEX:SPY"
                            },
                            {
                            "s": "NASDAQ:TSLA"
                            },
                            {
                            "s": "NASDAQ:AAPL"
                            },
                            {
                            "s": "NASDAQ:AMZN"
                            },
                            {
                            "s": "NASDAQ:QQQ"
                            },
                            {
                            "s": "NASDAQ:NVDA"
                            },
                            {
                            "s": "NASDAQ:MSFT"
                            },
                            {
                            "s": "NYSE:BA"
                            }
                        ]
                        }
                    ]
                    }
                    </script>
                    </div>
                    <!-- TradingView Widget END -->""", width=270, height=580)
                                
                st.markdown("### Economic Calender")
                components.html("""<!-- TradingView Widget BEGIN -->
                    <div class="tradingview-widget-container">
                    <div class="tradingview-widget-container__widget"></div>
                    <div class="tradingview-widget-copyright"><a href="https://www.tradingview.com/economic-calendar/" rel="noopener" target="_blank"><span class="blue-text">Economic calendar</span></a> by TradingView</div>
                    <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-events.js" async>
                    {
                    "colorTheme": "light",
                    "isTransparent": false,
                    "width": "270",
                    "height": "600",
                    "locale": "en",
                    "importanceFilter": "0,1",
                    "currencyFilter": "AUD,CAD,EUR,FRF,DEM,ITL,JPY,MXN,NZD,ZAR,ESP,CHF,TRL,GBP,USD"
                    }
                    </script>
                    </div>
                    <!-- TradingView Widget END -->""", height=590, width=270)
                
            time.sleep(2)


        



