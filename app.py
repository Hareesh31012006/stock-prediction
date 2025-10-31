import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
import os
import json
from datetime import datetime, timedelta
from textblob import TextBlob
from newsapi import NewsApiClient
import requests
import re

# Set page config
st.set_page_config(page_title="Advanced Stock Prediction with Sentiment", page_icon="🚀", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .prediction-section {
        background-color: #e8f4f8;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .sentiment-positive {
        color: #28a745;
        font-weight: bold;
    }
    .sentiment-negative {
        color: #dc3545;
        font-weight: bold;
    }
    .sentiment-neutral {
        color: #6c757d;
        font-weight: bold;
    }
    .recommendation-buy {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .recommendation-sell {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .recommendation-hold {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'sentiment' not in st.session_state:
    st.session_state.sentiment = None
if 'news_sentiment' not in st.session_state:
    st.session_state.news_sentiment = None

# --------------------------
# New Function for Recommendation
# --------------------------
def get_trade_recommendation(current_price, predicted_price):
    """Simple rule-based Buy/Sell/Hold recommendation."""
    price_diff = (predicted_price - current_price) / current_price * 100

    if price_diff > 2:
        return "Buy"
    elif price_diff < -2:
        return "Sell"
    else:
        return "Hold"

# --------------------------
# Rest of your code (unchanged)
# --------------------------

def download_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    with st.spinner(f"Downloading {ticker} data..."):
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if df.empty:
        st.error("No data downloaded. Check ticker symbol or date range.")
        return pd.DataFrame()
    df = df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].copy()
    df.index = pd.to_datetime(df.index)
    return df

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MA50'] = df['Close'].rolling(window=50, min_periods=1).mean()
    df['MA200'] = df['Close'].rolling(window=200, min_periods=1).mean()
    df['Momentum'] = df['Close'] - df['Close'].shift(10)
    df['ROC'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100
    df['Daily_Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Daily_Return'].rolling(window=20).std()
    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Upper'] = df['BB_Middle'] + 2 * df['Close'].rolling(window=20).std()
    df['BB_Lower'] = df['BB_Middle'] - 2 * df['Close'].rolling(window=20).std()
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    df.bfill(inplace=True)
    df.ffill(inplace=True)
    return df

def create_sequences(values: np.ndarray, seq_len: int) -> tuple:
    X, y = [], []
    for i in range(seq_len, len(values)):
        X.append(values[i-seq_len:i])
        y.append(values[i, 0])
    return np.array(X), np.array(y)

def build_enhanced_lstm(input_shape, units=100, dropout=0.3) -> tf.keras.Model:
    model = Sequential()
    model.add(Bidirectional(LSTM(units=units, return_sequences=True, input_shape=input_shape)))
    model.add(Dropout(dropout))
    model.add(Bidirectional(LSTM(units=units//2, return_sequences=True)))
    model.add(Dropout(dropout))
    model.add(LSTM(units=units//4, return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(units=50, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='huber', metrics=['mae', 'mse'])
    return model

def train_model(df: pd.DataFrame, seq_len=60, epochs=10, batch_size=32):
    feature_cols = ['Close', 'EMA12', 'EMA26', 'MA50', 'MA200', 'Momentum', 'Return']
    if 'Return' not in df.columns:
        df['Return'] = df['Close'].pct_change().fillna(0)
    data = df[feature_cols].values.astype('float32')

    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    test_data = data[train_size - seq_len:]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_data)
    train_scaled = scaler.transform(train_data)
    test_scaled = scaler.transform(test_data)

    X_train, y_train = create_sequences(train_scaled, seq_len)
    X_test, y_test = create_sequences(test_scaled, seq_len)

    model = build_enhanced_lstm(input_shape=(X_train.shape[1], X_train.shape[2]), units=128, dropout=0.3)

    progress_bar = st.progress(0)
    status_text = st.empty()

    class ProgressCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            progress = (epoch + 1) / epochs
            progress_bar.progress(progress)
            status_text.text(f"Epoch {epoch + 1}/{epochs} - Loss: {logs['loss']:.4f} - Val Loss: {logs.get('val_loss', 0):.4f}")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[ProgressCallback()],
        verbose=0
    )

    progress_bar.empty()
    status_text.empty()

    preds_scaled = model.predict(X_test, verbose=0)
    inv_preds, inv_true = [], []

    for i in range(len(X_test)):
        last_features = X_test[i, -1, :].copy()
        arr_pred = last_features.copy()
        arr_true = last_features.copy()
        arr_pred[0], arr_true[0] = preds_scaled[i, 0], y_test[i]
        inv_pred = scaler.inverse_transform(arr_pred.reshape(1, -1))[0, 0]
        inv_t = scaler.inverse_transform(arr_true.reshape(1, -1))[0, 0]
        inv_preds.append(inv_pred)
        inv_true.append(inv_t)

    inv_preds, inv_true = np.array(inv_preds), np.array(inv_true)
    rmse = np.sqrt(mean_squared_error(inv_true, inv_preds))
    test_dates = df.index[train_size:train_size + len(inv_preds)]
    return model, scaler, history, inv_preds, inv_true, test_dates, rmse

def plot_predictions(dates, true_values, pred_values, ticker):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates, true_values, label='Actual Close Price', color='blue', linewidth=2)
    ax.plot(dates, pred_values, label='Predicted Close Price', color='red', linewidth=2, linestyle='--')
    ax.set_title(f'{ticker} Stock Price Prediction', fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price ($)', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

# --- MAIN FUNCTION ---
def main():
    st.markdown('<div class="main-header">📈 Stock Price Prediction App</div>', unsafe_allow_html=True)
    st.sidebar.header("📊 Configuration")

    ticker = st.sidebar.text_input("Stock Ticker", value="AAPL")
    start_date = st.sidebar.date_input("Start Date", value=datetime.now() - timedelta(days=365*5))
    end_date = st.sidebar.date_input("End Date", value=datetime.now())
    seq_len = st.sidebar.slider("Sequence Length", 30, 120, 60)
    epochs = st.sidebar.slider("Training Epochs", 5, 50, 10)
    news_api_key = st.sidebar.text_input("NewsAPI Key (optional)", type="password")

    if st.sidebar.button("🚀 Train Model", type="primary"):
        df = download_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        if not df.empty:
            st.session_state.df = add_technical_indicators(df)
            with st.spinner("Training LSTM model..."):
                model, scaler, history, pred_values, true_values, dates, rmse = train_model(st.session_state.df, seq_len, epochs)
            st.session_state.model_trained = True
            st.session_state.predictions = {
                'dates': dates, 'true_values': true_values, 'pred_values': pred_values, 'rmse': rmse, 'ticker': ticker
            }
            st.success("✅ Model trained successfully!")

    if st.session_state.model_trained and st.session_state.predictions:
        pred_data = st.session_state.predictions
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Test RMSE", f"${pred_data['rmse']:.2f}")
        with col2: st.metric("Data Points", len(pred_data['true_values']))
        with col3: st.metric("Ticker", pred_data['ticker'])

        st.subheader("📈 Prediction Results")
        fig = plot_predictions(pred_data['dates'], pred_data['true_values'], pred_data['pred_values'], pred_data['ticker'])
        st.pyplot(fig)

        # --------------------------
        # NEW: Recommendation Section
        # --------------------------
        current_price = st.session_state.df['Close'].iloc[-1]
        predicted_price = pred_data['pred_values'][-1]
        recommendation = get_trade_recommendation(current_price, predicted_price)

        if recommendation == "Buy":
            st.markdown('<div class="recommendation-buy">📈 Recommendation: BUY</div>', unsafe_allow_html=True)
        elif recommendation == "Sell":
            st.markdown('<div class="recommendation-sell">📉 Recommendation: SELL</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="recommendation-hold">🤝 Recommendation: HOLD</div>', unsafe_allow_html=True)

        st.write(f"**Current Price:** ${current_price:.2f}")
        st.write(f"**Predicted Price:** ${predicted_price:.2f}")

if __name__ == "__main__":
    main()
