"""
BTC‑USDT 1‑Hour Prediction Bot • Bybit Edition (v3)
==================================================
Cambia la fuente de datos de **Binance → Bybit** para evitar bloqueos de IP en nubes
públicas y elimina las notificaciones locales de escritorio (`plyer`).

Puntos clave
------------
✔ Sustituido `binance.client` por **pybit** (API REST unificada de Bybit).
✔ Función `fetch_recent_klines` adaptada al formato de Bybit (timestamps en segundos).
✔ Dependencia nueva: `pybit` (añade en `requirements.txt`).
✔ Se quitan todas las llamadas a `plyer.notification` (solo quedan avisos vía ntfy).
✔ El resto de la lógica (LSTM, scaler, predicción, APScheduler) se mantiene idéntica.

Requisitos (requirements.txt)
-----------------------------
```
pybit
pandas
numpy
tensorflow-cpu
scikit-learn
apscheduler
requests
nest_asyncio
pytz
```
"""

# ───────────────────────────── 1. CONFIG & SETUP ─────────────────────────────
import os, asyncio, warnings
from datetime import datetime, timedelta, timezone
from typing import Tuple

import nest_asyncio
nest_asyncio.apply()

import numpy as np
import pandas as pd
import requests
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Bidirectional, Input, Attention, LayerNormalization, Flatten
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from pybit.unified_trading import HTTP  # Bybit SDK

# ─────────────── Parámetros ajustables ───────────────
TICKER               = "BTCUSDT"
INTERVAL             = 5
LOOK_BACK            = 60
PRED_HORIZON_MIN     = 60
LOOK_AHEAD_STEPS     = PRED_HORIZON_MIN // INTERVAL
MONITOR_FREQ_SEC     = 300
SIGNIFICANT_PCT      = 0.6
TRAIN_SPLIT_RATIO    = 0.8
EPOCHS               = 10
BATCH_SIZE           = 128
RETRAIN_EVERY_HRS    = 12
COOLDOWN_MIN_MONITOR = 10
MODEL_DIR            = "models"; os.makedirs(MODEL_DIR, exist_ok=True)

# ntfy.sh --------------------------------------------------------------------
NTFY_TOPIC = os.getenv("NTFY_TOPIC", "mente-sardina")
NTFY_URL   = f"https://ntfy.sh/{NTFY_TOPIC}".rstrip("/")

def ntfy_send(text: str, priority: str = "default"):
    try:
        requests.post(NTFY_URL, data=text.encode(), headers={"Priority": priority}, timeout=10)
    except Exception as e:
        print(f"[ntfy] error: {e}")

# ────────────────────── 2. DATA & FEATURES (Bybit) ──────────────────────
session = HTTP(testnet=False)

def fetch_recent_klines(lookback: int) -> pd.DataFrame:
    limit = min(lookback + 100, 1000)
    res = session.get_kline(category="linear", symbol=TICKER, interval=str(INTERVAL), limit=limit)
    data = res["result"]["list"]
    cols = ["Open time","Open","High","Low","Close","Volume","Turnover"]
    df = pd.DataFrame(data, columns=cols)
    df = df.astype({c: float for c in ["Open","High","Low","Close","Volume"]})
    df["Open time"] = pd.to_datetime(df["Open time"], unit="ms", utc=True)
    df.sort_values("Open time", inplace=True)
    df.set_index("Open time", inplace=True)
    return df

FEATURE_COLUMNS = ["Open","High","Low","Close","Volume","log_ret","range"]

def build_feature_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_ret"] = np.log(df["Close"].pct_change() + 1).fillna(0)
    df["range"]   = df["High"] - df["Low"]
    return df[FEATURE_COLUMNS]

# ────────────────────── 3. SEQUENCES ──────────────────────
scaler = StandardScaler()

def create_sequences(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, y_reg, y_cls = [], [], []
    idx_close = FEATURE_COLUMNS.index("Close")
    for i in range(LOOK_BACK, len(data) - LOOK_AHEAD_STEPS):
        window = data[i-LOOK_BACK:i]
        target = data[i+LOOK_AHEAD_STEPS-1, idx_close]
        X.append(window)
        y_reg.append(target)
        y_cls.append(int(target > data[i-1, idx_close]))
    return np.array(X), np.array(y_reg), np.array(y_cls)

# ────────────────────── 4. MODEL ──────────────────────
def build_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Bidirectional(LSTM(64, return_sequences=True))(inputs)
    x = Dropout(0.3)(x)
    x = Attention(use_scale=True)([x, x])
    x = LayerNormalization()(x)
    x = Flatten()(x)
    reg = Dense(1, name="reg_out")(Dense(32, activation="relu")(x))
    cls = Dense(1, activation="sigmoid", name="cls_out")(Dense(32, activation="relu")(x))
    model = Model(inputs, [reg, cls])
    model.compile(optimizer="adam", loss={"reg_out":"mse","cls_out":"binary_crossentropy"}, metrics={"cls_out":"accuracy"})
    return model

model = None

# ─────────────── 5. TRAIN / RETRAIN ───────────────
async def train_or_load():
    global model, scaler
    path = os.path.join(MODEL_DIR, f"btc_l{LOOK_BACK}_h{LOOK_AHEAD_STEPS}.h5")
    df_hist = fetch_recent_klines(LOOK_BACK + 12000)
    scaler.fit(build_feature_df(df_hist))
    if os.path.exists(path):
        mdl = load_model(path, compile=False)
        if mdl.input_shape[1] == LOOK_BACK:
            mdl.compile(optimizer="adam", loss={"reg_out":"mse","cls_out":"binary_crossentropy"}, metrics={"cls_out":"accuracy"})
            model = mdl
            print("✓ Modelo cargado de disco")
            return
        print("⚠ Forma incompatible, reentrenando…")
    data = scaler.transform(build_feature_df(df_hist))
    X, y_r, y_c = create_sequences(data)
    split = int(TRAIN_SPLIT_RATIO * len(X))
    X_tr, y_r_tr, y_c_tr = X[:split], y_r[:split], y_c[:split]
    X_val,y_r_val,y_c_val = X[split:],y_r[split:],y_c[split:]
    model = build_model((LOOK_BACK, len(FEATURE_COLUMNS)))
    cb = [EarlyStopping(patience=3, restore_best_weights=True), ModelCheckpoint(path, save_best_only=True)]
    print("⌛ Entrenando modelo (1h)…")
    model.fit(X_tr, {"reg_out":y_r_tr,"cls_out":y_c_tr}, validation_data=(X_val,{"reg_out":y_r_val,"cls_out":y_c_val}), epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=cb, verbose=2)
    print("✓ Entrenamiento completo")

# ────────────────────── 6. PRED / MONITOR ──────────────────────
current_pred = {"price":None,"direction":None,"target_time":None}
last_mon = datetime.min.replace(tzinfo=timezone.utc)

@tf.function(reduce_retracing=True)
def _infer(seq):
    return model(seq, training=False)

async def job_hourly_prediction():
    global current_pred
    df = fetch_recent_klines(LOOK_BACK+1)
    feat = build_feature_df(df)
    seq = scaler.transform(feat)[-LOOK_BACK:].reshape((1,LOOK_BACK,len(FEATURE_COLUMNS)))
    p_s, p_up = _infer(seq)
    p_s,p_up = float(p_s[0][0]), float(p_up[0][0])
    idx = FEATURE_COLUMNS.index("Close")
    last_close = feat.iloc[-1]["Close"]
    pred_price = scaler.mean_[idx] + scaler.scale_[idx]*p_s
    pct = (pred_price-last_close)/last_close*100
    now = datetime.now(timezone.utc)
    current_pred={"price":pred_price,"direction":pct>0,"target_time":now+timedelta(minutes=PRED_HORIZON_MIN)}
    arrow = "↑" if pct>0 else "↓"
    msg=f"BTC 1h ({now:%H:%M}) {arrow} ${pred_price:,.0f} ({pct:+.2f}%)"
    ntfy_send(msg, priority="high")
    print("[Pred]",msg)

async def job_monitor_deviation():
    global last_mon, current_pred
    if current_pred["price"] is None or datetime.now(timezone.utc)>=current_pred["target_time"]:
        return
    price = fetch_recent_klines(1).iloc[-1]["Close"]
    diff=(price-current_pred["price"]) / current_pred["price"]*100
    if abs(diff)<SIGNIFICANT_PCT or (datetime.now(timezone.utc)-last_mon).total_seconds()<COOLDOWN_MIN_MONITOR*60:
        return
    last_mon=datetime.now(timezone.utc)
    arr="▲" if diff>0 else "▼"
    msg=f"BTC desvío {arr} {diff:+.2f}% | ${price:,.0f} vs ${current_pred['price']:,.0f}"        
    ntfy_send(msg)
    print("[Dev]",msg)

# ────────────────────── 7. MAIN ──────────────────────
async def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    await train_or_load()
    sched=AsyncIOScheduler()
    await job_hourly_prediction()
    sched.add_job(job_hourly_prediction, "interval", hours=1, max_instances=1, coalesce=True)
    sched.add_job(job_monitor_deviation, "interval", seconds=MONITOR_FREQ_SEC, max_instances=1, coalesce=True)
    sched.add_job(train_or_load, "interval", hours=RETRAIN_EVERY_HRS)
    sched.start()
    print("✅ Bot activo. ntfy.sh/"+NTFY_TOPIC)
    while True:
        await asyncio.sleep(3600)

if __name__=="__main__":
    try:
        asyncio.run(main())
    except RuntimeError:
        loop=asyncio.get_event_loop()
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        print("Adiós")
