import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objects as go

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Stock Dashboard", layout="wide")

st.title("📊 Stock Analytics Dashboard")

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Settings")

stock = st.sidebar.text_input("Enter Stock Symbol", "RELIANCE.NS")
compare_stock = st.sidebar.text_input("Compare With", "TCS.NS")
forecast_days = st.sidebar.slider("Forecast Days", 1, 30, 10)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data(stock):
    df = yf.download(stock, period="2y")
    df.dropna(inplace=True)
    return df

df = load_data(stock)

# Safety check
if df.empty:
    st.error("❌ No data found. Check stock symbol.")
    st.stop()

# ---------------- KPI ----------------
latest = df['Close'].iloc[-1]
avg = df['Close'].mean()
high = df['Close'].max()
low = df['Close'].min()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest Price", round(latest,2))
c2.metric("Average Price", round(avg,2))
c3.metric("Max Price", round(high,2))
c4.metric("Min Price", round(low,2))

# ---------------- LINE CHART ----------------
st.subheader("📈 Price Trend")
st.line_chart(df['Close'])

# ---------------- CANDLESTICK ----------------
st.subheader("🕯️ Candlestick Chart")

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
    try:
        df2 = load_data(compare_stock)
        compare_df = pd.DataFrame({
            stock: df['Close'],
            compare_stock: df2['Close']
        })
        st.subheader("📊 Stock Comparison")
        st.line_chart(compare_df)
    except:
        st.warning("⚠️ Comparison stock not found")

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
df.loc[20:, 'Signal'] = np.where(df['MA20'][20:] > df['MA50'][20:], 1, 0)
df['Position'] = df['Signal'].diff()

fig2, ax2 = plt.subplots()
ax2.plot(df['Close'], label='Price')

buy = df[df['Position'] == 1]
sell = df[df['Position'] == -1]

ax2.plot(buy.index, buy['Close'], '^', markersize=10, label='Buy')
ax2.plot(sell.index, sell['Close'], 'v', markersize=10, label='Sell')

ax2.legend()
st.pyplot(fig2)

# ---------------- MONTHLY BAR GRAPH (FINAL FIX) ----------------
st.subheader("📊 Monthly Average Price")

try:
    df_copy = df.copy()

    # Ensure datetime index
    df_copy.index = pd.to_datetime(df_copy.index, errors='coerce')

    # Drop invalid dates
    df_copy = df_copy[~df_copy.index.isna()]

    # Sort index
    df_copy = df_copy.sort_index()

    # Resample safely
    monthly = df_copy['Close'].resample('ME').mean()

    st.bar_chart(monthly)

except:
    st.warning("⚠️ Monthly chart failed")

# ---------------- RETURNS ----------------
st.subheader("📉 Daily Returns")

df['Returns'] = df['Close'].pct_change()

fig3, ax3 = plt.subplots()
df['Returns'].hist(bins=50, ax=ax3)
st.pyplot(fig3)

# ---------------- ARIMA ----------------
st.subheader("🔮 ARIMA Forecast")

try:
    model = ARIMA(df['Close'], order=(5,1,0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=forecast_days)

    future_dates = pd.date_range(start=df.index[-1], periods=forecast_days+1, freq='B')[1:]

    fig4, ax4 = plt.subplots()
    ax4.plot(df['Close'], label='Actual')
    ax4.plot(future_dates, forecast, linestyle='dashed', label='Forecast')
    ax4.legend()

    st.pyplot(fig4)

    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted Price': forecast
    }).set_index('Date')

    st.dataframe(forecast_df)

except:
    st.warning("⚠️ ARIMA model failed")

# ---------------- INSIGHTS ----------------
st.subheader("🧠 Key Insights")

trend = "Uptrend 📈" if latest > avg else "Downtrend 📉"

st.write(f"""
- Trend: **{trend}**
- Current price is {'above' if latest > avg else 'below'} average
- Moving averages show trend direction
- Buy/Sell signals indicate entry/exit points
- ARIMA gives short-term prediction
""")
