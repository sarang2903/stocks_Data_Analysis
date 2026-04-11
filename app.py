import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Page config
st.set_page_config(page_title="Stock Analytics Dashboard", layout="wide")

# Title
st.title("📊 Stock Analytics Dashboard")
st.markdown("Advanced Stock Insights & Prediction using ARIMA")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('Stocks_Data')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    return df

df = load_data()

# Sidebar
st.sidebar.header("⚙️ Settings")
stock_list = df['stock'].unique()
selected_stock = st.sidebar.selectbox("Select Stock", stock_list)
forecast_days = st.sidebar.slider("Forecast Days", 1, 30, 10)

# Filter data
st_data = df[df['stock'] == selected_stock][['Close']]

# ---------------- KPI SECTION ----------------
latest_price = st_data['Close'].iloc[-1]
avg_price = st_data['Close'].mean()
max_price = st_data['Close'].max()
min_price = st_data['Close'].min()

col1, col2, col3, col4 = st.columns(4)
col1.metric("💰 Latest Price", round(latest_price, 2))
col2.metric("📊 Average Price", round(avg_price, 2))
col3.metric("📈 Max Price", round(max_price, 2))
col4.metric("📉 Min Price", round(min_price, 2))

# ---------------- LINE CHART ----------------
st.subheader("📈 Price Trend")
st.line_chart(st_data)

# ---------------- MOVING AVERAGE ----------------
st.subheader("📊 Moving Average")

st_data['MA20'] = st_data['Close'].rolling(20).mean()
st_data['MA50'] = st_data['Close'].rolling(50).mean()

fig1, ax1 = plt.subplots()
ax1.plot(st_data['Close'], label='Close Price')
ax1.plot(st_data['MA20'], label='MA20')
ax1.plot(st_data['MA50'], label='MA50')
ax1.legend()

st.pyplot(fig1)

# ---------------- BAR GRAPH (MONTHLY AVG) ----------------
st.subheader("📊 Monthly Average Price")

monthly = st_data.resample('M').mean()

fig2, ax2 = plt.subplots()
monthly.plot(kind='bar', ax=ax2)
st.pyplot(fig2)

# ---------------- RETURNS ANALYSIS ----------------
st.subheader("📉 Daily Returns")

st_data['Returns'] = st_data['Close'].pct_change()

fig3, ax3 = plt.subplots()
st_data['Returns'].hist(bins=50, ax=ax3)
st.pyplot(fig3)

# ---------------- ARIMA MODEL ----------------
st.subheader("🔮 Forecast Prediction")

with st.spinner("Training Model..."):
    model = ARIMA(st_data['Close'], order=(5,1,0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=forecast_days)

future_dates = pd.date_range(start=st_data.index[-1], periods=forecast_days+1, freq='B')[1:]

# Forecast Plot
fig4, ax4 = plt.subplots()
ax4.plot(st_data['Close'], label='Actual')
ax4.plot(future_dates, forecast, linestyle='dashed', label='Forecast')
ax4.legend()

st.pyplot(fig4)

# Forecast Table
forecast_df = pd.DataFrame({
    'Date': future_dates,
    'Predicted Price': forecast
}).set_index('Date')

st.dataframe(forecast_df)

# Download
csv = forecast_df.to_csv().encode('utf-8')
st.download_button("📥 Download Forecast", csv, "forecast.csv", "text/csv")

# ---------------- INSIGHTS ----------------
st.subheader("🧠 Key Insights")

trend = "Uptrend 📈" if latest_price > avg_price else "Downtrend 📉"

st.write(f"""
- Current Trend: **{trend}**
- Price is {'above' if latest_price > avg_price else 'below'} average
- Volatility (returns spread) shows market risk
- Moving averages help identify trend direction
- Forecast suggests possible future movement (short-term only)
""")
