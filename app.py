# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Page config
st.set_page_config(page_title="Stock Price Predictor", layout="wide")

# Title
st.title("📈 Stock Price Prediction App")
st.write("Predict future stock prices using ARIMA model")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('Stocks_Data')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    return df

df = load_data()

# Sidebar for user input
st.sidebar.header("⚙️ Settings")

stock_list = df['stock'].unique()
selected_stock = st.sidebar.selectbox("Select Stock", stock_list)

forecast_days = st.sidebar.slider("Select Forecast Days", 1, 30, 10)

# Filter data
st_data = df[df['stock'] == selected_stock][['Close']]

# Show data
st.subheader(f"📊 {selected_stock} Stock Data")
st.line_chart(st_data)

# Show raw data option
if st.checkbox("Show Raw Data"):
    st.write(st_data.tail())

# Train ARIMA model
with st.spinner("Training model..."):
    model = ARIMA(st_data['Close'], order=(5,1,0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=forecast_days)

# Create future dates
future_dates = pd.date_range(start=st_data.index[-1], periods=forecast_days+1, freq='B')[1:]

# Plot results
st.subheader("📉 Prediction Results")

fig, ax = plt.subplots(figsize=(10,5))
ax.plot(st_data['Close'], label='Actual Prices')
ax.plot(future_dates, forecast, label='Predicted Prices', linestyle='dashed')
ax.set_xlabel("Date")
ax.set_ylabel("Stock Price")
ax.legend()

st.pyplot(fig)

# Show forecast values
st.subheader("🔮 Forecast Data")
forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': forecast})
forecast_df.set_index('Date', inplace=True)
st.dataframe(forecast_df)

# Download option
csv = forecast_df.to_csv().encode('utf-8')
st.download_button("📥 Download Forecast", csv, "forecast.csv", "text/csv")
