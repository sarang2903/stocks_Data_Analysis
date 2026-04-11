import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Page config
st.set_page_config(page_title="Advanced Stock Dashboard", layout="wide")

st.title("📊 Advanced Stock Analytics Dashboard")

# Sidebar
st.sidebar.header("⚙️ Settings")

stock = st.sidebar.text_input("Enter Stock Symbol (e.g. RELIANCE.NS)", "RELIANCE.NS")
compare_stock = st.sidebar.text_input("Compare With (optional)", "TCS.NS")
forecast_days = st.sidebar.slider("Forecast Days", 1, 30, 10)

# ---------------- LIVE DATA ----------------
@st.cache_data
def load_data(stock):
    df = yf.download(stock, period="2y")
    return df

df = load_data(stock)

st.subheader(f"📈 {stock} Stock Data")
st.line_chart(df['Close'])

# ---------------- CANDLESTICK ----------------
st.subheader("🕯️ Candlestick Chart")

import plotly.graph_objects as go

fig = go.Figure(data=[go.Candlestick(
    x=df.index,
    open=df['Open'],
    high=df['High'],
    low=df['Low'],
    close=df['Close']
)])
st.plotly_chart(fig, use_container_width=True)

# ---------------- MULTI STOCK ----------------
if compare_stock:
    df2 = load_data(compare_stock)
    st.subheader("📊 Stock Comparison")

    compare_df = pd.DataFrame({
        stock: df['Close'],
        compare_stock: df2['Close']
    })

    st.line_chart(compare_df)

# ---------------- MOVING AVERAGE ----------------
st.subheader("📊 Moving Averages")

df['MA20'] = df['Close'].rolling(20).mean()
df['MA50'] = df['Close'].rolling(50).mean()

fig1, ax1 = plt.subplots()
ax1.plot(df['Close'], label='Close')
ax1.plot(df['MA20'], label='MA20')
ax1.plot(df['MA50'], label='MA50')
ax1.legend()

st.pyplot(fig1)

# ---------------- BUY/SELL SIGNAL ----------------
st.subheader("💡 Buy/Sell Signals")

df['Signal'] = 0
df['Signal'][20:] = np.where(df['MA20'][20:] > df['MA50'][20:], 1, 0)
df['Position'] = df['Signal'].diff()

fig2, ax2 = plt.subplots()
ax2.plot(df['Close'], label='Price')

buy = df[df['Position'] == 1]
sell = df[df['Position'] == -1]

ax2.plot(buy.index, buy['Close'], '^', markersize=10, label='Buy')
ax2.plot(sell.index, sell['Close'], 'v', markersize=10, label='Sell')

ax2.legend()
st.pyplot(fig2)

# ---------------- ARIMA ----------------
st.subheader("🔮 ARIMA Forecast")

model = ARIMA(df['Close'], order=(5,1,0))
model_fit = model.fit()
forecast = model_fit.forecast(steps=forecast_days)

future_dates = pd.date_range(start=df.index[-1], periods=forecast_days+1, freq='B')[1:]

fig3, ax3 = plt.subplots()
ax3.plot(df['Close'], label='Actual')
ax3.plot(future_dates, forecast, linestyle='dashed', label='Forecast')
ax3.legend()

st.pyplot(fig3)

# ---------------- LSTM MODEL ----------------
st.subheader("🤖 LSTM Deep Learning Prediction")

data = df[['Close']]
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

train_data = scaled_data[:int(len(scaled_data)*0.8)]

X_train = []
y_train = []

for i in range(60, len(train_data)):
    X_train.append(train_data[i-60:i])
    y_train.append(train_data[i])

X_train, y_train = np.array(X_train), np.array(y_train)

model_lstm = Sequential()
model_lstm.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1],1)))
model_lstm.add(LSTM(50))
model_lstm.add(Dense(1))

model_lstm.compile(optimizer='adam', loss='mean_squared_error')

model_lstm.fit(X_train, y_train, epochs=2, batch_size=32, verbose=0)

# Prediction
test_data = scaled_data[int(len(scaled_data)*0.8)-60:]
X_test = []

for i in range(60, len(test_data)):
    X_test.append(test_data[i-60:i])

X_test = np.array(X_test)

predictions = model_lstm.predict(X_test)
predictions = scaler.inverse_transform(predictions)

fig4, ax4 = plt.subplots()
ax4.plot(df.index[-len(predictions):], predictions, label='LSTM Prediction')
ax4.plot(df['Close'], label='Actual')
ax4.legend()

st.pyplot(fig4)

# ---------------- INSIGHTS ----------------
st.subheader("🧠 Smart Insights")

latest = df['Close'].iloc[-1]
avg = df['Close'].mean()

trend = "Uptrend 📈" if latest > avg else "Downtrend 📉"

st.write(f"""
- Trend: **{trend}**
- Price is {'above' if latest > avg else 'below'} average
- Buy signals indicate potential entry points
- ARIMA = short-term prediction
- LSTM = deep learning prediction (more powerful)
""")
