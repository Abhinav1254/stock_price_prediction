import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from joblib import Parallel, delayed
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ── Reproducibility ─────────────────────────────────────────────────────
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'


def build_and_train_lstm(X_train, y_train, X_val, y_val,
                         seq_len=60, epochs=5, batch_size=8):
    """
    Builds and trains a lightweight LSTM model.
    Reduced epochs and units for speed.
    """
    model = Sequential([
        LSTM(16, return_sequences=True, input_shape=(seq_len, X_train.shape[2])),
        Dropout(0.2),
        LSTM(16),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,
        verbose=0
    )
    return model


def create_sequences(X, y, seq_len=60):
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i-seq_len:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


def fit_fixed_arima(series, order):
    return ARIMA(series, order=order).fit()


def hybrid_forecast_with_exogs(
    close_arima, exog_arimas, lstm_model,
    df_hist, scaler_X, scaler_y, EXOG_COLS,
    look_back=60, horizon=30
):
    # Copy history and get last known price
    dfh = df_hist.copy()
    last_price = float(dfh['Close'].iloc[-1])

    # Precompute ARIMA forecasts in bulk
    exog_preds = {col: model.forecast(horizon) for col, model in exog_arimas.items()}
    return_preds = close_arima.forecast(horizon)

    # Prepare rolling window for exogenous features
    window = scaler_X.transform(dfh[EXOG_COLS].values[-look_back:])\
             .reshape(1, look_back, len(EXOG_COLS))

    preds = []
    idx = []

    for step in range(horizon):
        today = dfh.index[-1] + pd.Timedelta(days=step+1)
        idx.append(today)
        # apply precomputed exog forecast
        for col in EXOG_COLS:
            dfh.at[today, col] = exog_preds[col].iloc[step]
        # update window
        new_feats = scaler_X.transform(dfh.loc[[today], EXOG_COLS])
        window = np.concatenate([window[:,1:,:], new_feats.reshape(1,1,-1)], axis=1)
        # ARIMA log-return forecast
        lr = return_preds.iloc[step]
        # predict residual
        resid_s = lstm_model.predict(window, verbose=0)[0,0]
        resid = scaler_y.inverse_transform([[resid_s]])[0,0]
        hybrid_lr = lr + resid
        # compute forecast price
        price = (preds[-1] if preds else last_price) * np.exp(hybrid_lr)
        preds.append(price)

    return pd.DataFrame({'Forecast_Price': preds}, index=idx)


def run_hybrid_arima_model(df: pd.DataFrame, horizon: int = 30, symbol: str = None):
    # Preprocess
    df_proc = df.copy()
    df_proc['Log_Return'] = np.log(df_proc['Close']/df_proc['Close'].shift(1))
    df_proc.dropna(inplace=True)
    EXOG_COLS = ['Open','High','Low','Close','Volume']

    # Fit ARIMA on returns and exogs once
    close_arima = fit_fixed_arima(df_proc['Log_Return'], order=(5,0,2))
    exog_arimas = Parallel(n_jobs=-1)(
        delayed(fit_fixed_arima)(df_proc[col], order=(1,1,1))
        for col in EXOG_COLS
    )
    exog_arimas = dict(zip(EXOG_COLS, exog_arimas))

    # Scale features and compute residuals
    scaler_X = MinMaxScaler().fit(df_proc[EXOG_COLS])
    arima_pred = close_arima.predict(start=0, end=len(df_proc)-1)
    residuals = df_proc['Log_Return'].values - arima_pred.values
    scaler_y = MinMaxScaler().fit(residuals.reshape(-1,1))
    scaled_resid = scaler_y.transform(residuals.reshape(-1,1))

    # Prepare LSTM data with smaller dataset
    X, y = create_sequences(scaler_X.transform(df_proc[EXOG_COLS]), scaled_resid)
    split = max(int(0.8 * len(X)), 1)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Train LSTM once
    lstm_model = build_and_train_lstm(X_train, y_train, X_test, y_test)

    # Forecast
    forecast_df = hybrid_forecast_with_exogs(
        close_arima, exog_arimas, lstm_model,
        df_proc, scaler_X, scaler_y, EXOG_COLS,
        look_back=60, horizon=horizon
    )

    # Plot
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(df_proc.index, df_proc['Close'], label='Historical')
    ax.plot(forecast_df.index, forecast_df['Forecast_Price'], '--', label='Forecast')
    title = f"{symbol + ' ' if symbol else ''}{horizon}-Day Forecast"
    ax.set(title=title, xlabel='Date', ylabel='Price')
    ax.legend(); plt.tight_layout()

    return forecast_df, fig
