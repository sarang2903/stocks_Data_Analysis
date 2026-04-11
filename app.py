import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Page config
st.set_page_config(page_title="Advanced Stock Predictor", layout="wide")

st.title("🚀 Advanced Stock Price Prediction Dashboard")
st.write("Interactive stock analysis with insights")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('./Stocks_Data')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    return df

df = load_data()

# Sidebar
st.sidebar.header("⚙️ Controls")
stock_list = df['stock'].unique()
selected_stock = st.sidebar.selectbox("Select Stock", stock_list)
forecast_days = st.sidebar.slider("Forecast Days", 1, 30, 10)

# Filter data
st_data = df[df['stock'] == selected_stock][['Close']]

# Returns
st_data['Returns'] = st_data['Close'].pct_change()
st_data.dropna(inplace=True)

# =======================
# 💡 Key Insights
# =======================
st.subheader("💡 Key Insights")
col1, col2, col3 = st.columns(3)

col1.metric("Average Price", round(st_data['Close'].mean(), 2))
col2.metric("Highest Price", round(st_data['Close'].max(), 2))
col3.metric("Lowest Price", round(st_data['Close'].min(), 2))

trend = "Uptrend 📈" if st_data['Close'].iloc[-1] > st_data['Close'].iloc[0] else "Downtrend 📉"
st.info(f"Overall Trend: {trend}")

# =======================
# 📊 Subplots (FIXED)
# =======================
fig = make_subplots(
    rows=2, cols=2,
    specs=[
        [{"type": "xy"}, {"type": "domain"}],
        [{"type": "xy"}, {"type": "xy"}]
    ],
    subplot_titles=("Price Trend", "Market Sentiment", "Price Distribution", "Prediction")
)

# 📈 Line Chart
fig.add_trace(
    go.Scatter(x=st_data.index, y=st_data['Close'], name='Price'),
    row=1, col=1
)

# 🥧 Pie Chart
pos_days = (st_data['Returns'] > 0).sum()
neg_days = (st_data['Returns'] <= 0).sum()

fig.add_trace(
    go.Pie(labels=['Positive', 'Negative'], values=[pos_days, neg_days]),
    row=1, col=2
)

# 📊 Histogram
fig.add_trace(
    go.Histogram(x=st_data['Close'], nbinsx=30, name='Histogram'),
    row=2, col=1
)

# =======================
# 🔮 Prediction
# =======================
model = ARIMA(st_data['Close'], order=(5,1,0))
model_fit = model.fit()
forecast = model_fit.forecast(steps=forecast_days)

future_dates = pd.date_range(start=st_data.index[-1], periods=forecast_days+1, freq='B')[1:]

# Actual
fig.add_trace(
    go.Scatter(x=st_data.index, y=st_data['Close'], name='Actual'),
    row=2, col=2
)

# Predicted
fig.add_trace(
    go.Scatter(x=future_dates, y=forecast, name='Predicted'),
    row=2, col=2
)

# Layout
fig.update_layout(height=700, showlegend=True)

st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# =======================
# 📏 Model Performance
# =======================
st.subheader("📏 Model Performance")

train = st_data['Close'][:-forecast_days]
test = st_data['Close'][-forecast_days:]

model2 = ARIMA(train, order=(5,1,0))
model2_fit = model2.fit()
pred_test = model2_fit.forecast(steps=forecast_days)

rmse = np.sqrt(mean_squared_error(test, pred_test))
mae = mean_absolute_error(test, pred_test)

col1, col2 = st.columns(2)
col1.metric("RMSE", round(rmse, 2))
col2.metric("MAE", round(mae, 2))

# =======================
# 📂 Raw Data
# =======================
if st.checkbox("Show Raw Data"):
    st.write(st_data.tail())

# =======================
# 📥 Download
# =======================
pred_df = pd.DataFrame({'Date': future_dates, 'Predicted': forecast}).set_index('Date')
csv = pred_df.to_csv().encode('utf-8')

st.download_button("📥 Download Predictions", csv, "predictions.csv", "text/csv")
