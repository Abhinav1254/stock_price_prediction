import streamlit as st
import yfinance as yf
from hybrid_model import run_hybrid_arima_model
import matplotlib.pyplot as plt

st.title("📈 Hybrid ARIMA-LSTM Stock Price Predictor")

stock_symbol = st.text_input("Enter stock symbol (e.g., AAPL, TCS.NS):")
n_days = st.slider("How many days to predict into the future?", 7, 60, 30)

if st.button("Predict"):
    st.info("Fetching stock data...")
    data = yf.download(stock_symbol, period="5y")

    if data.empty:
        st.error("No data found for this symbol.")
    else:
        st.success("Data fetched. Running prediction...")
        predicted_df, fig = run_hybrid_arima_model(data, n_days)
        st.pyplot(fig)
        st.subheader("Predicted Future Prices")
        st.dataframe(predicted_df.tail(n_days))
