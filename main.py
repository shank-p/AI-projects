"""

"""

# Importing necessary files
from statistics import mode
import streamlit as st
from datetime import date as dt
import yfinance as yf
from plotly import graph_objs as go
import plotly.express as ex

from prophet import Prophet
from prophet.plot import plot_plotly

START_DATE = '2022-01-01'
TODAY_DATE = dt.today().strftime('%Y-%m-%d')

# web app 
st.title("Stock Web App")

stocks = ("AAPL", "AMZN", "GOOG", "MSFT", "AMD")
selected_stock = st.selectbox("Select a Stock : ", stocks)

prediction_days = st.slider("Days of Prediction:", 1, 10)
period = prediction_days

@st.cache
def load_data(ticker):
    data = yf.download(ticker, START_DATE, TODAY_DATE)
    #print(data)
    data.reset_index(inplace=True)
    return data

data_state = st.text("Getting data .....")
data = load_data(selected_stock)
data_state.text("Downloading data ..... Complete!")

# visulaise the raw data
st.subheader("Raw Data:")
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open']))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close']))
    fig.layout.update(title_text='Time Series Data', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()


# prophet -forecasting
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={'Date':'ds', 'Close':'y'})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader("Prophet forecast:")
st.write(forecast.tail())

# prophet chart
st.write("Prophet forecast chart:")
prophet_fig = plot_plotly(m, forecast)
st.plotly_chart(prophet_fig)






