"""
BTC‑USD 1‑Hour Prediction Bot (v2.2)
===================================
Corrección de bug: el bucle infinito de `main()` intentaba `await asyncio`, lo que
provocaba `TypeError`. Ahora usa `await asyncio.sleep(3600)`.

---
Solo se ha cambiado el final de la función `main()`. El resto permanece igual.
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
from binance.client import Client
from plyer import notification

# ─────────────── Parámetros ajustables ───────────────
TICKER               = "BTCUSDT"
INTERVAL             = Client.KLINE_INTERVAL_5MINUTE
LOOK_BACK            = 60                  # largo de la ventana (nº velas)
PRED_HORIZON_MIN     = 60                  # horizonte objetivo (minutos)
LOOK_AHEAD_STEPS     = PRED_HORIZON_MIN // 5  # 12 velas de 5 min → 1 h
MONITOR_FREQ_SEC     = 300                # chequeo cada 5 min
SIGNIFICANT_PCT      = 0.6                # umbral de desviación (%)
TRAIN_SPLIT_RATIO    = 0.8
EPOCHS               = 10
BATCH_SIZE           = 128
RETRAIN_EVERY_HRS    = 12                # re‑entreno completo
COOLDOWN_MIN_MONITOR = 10                # anti‑spam monitor
MODEL_DIR            = "models"; os.makedirs(MODEL_DIR, exist_ok=True)

# ntfy.sh --------------------------------------------------------------------
NTFY_TOPIC = os.getenv("NTFY_TOPIC", "mente-sardina")
NTFY_URL   = f"https://ntfy.sh/{NTFY_TOPIC}".rstrip("/")

def ntfy_send(text: str, priority: str = "default"):
    try:
        requests.post(NTFY_URL, data=text.encode(), headers={"Priority": priority}, timeout=10)
    except Exception as e:
        print(f"[ntfy] error: {e}")

# ────────────────────── 2. DATA & FEATURES ──────────────────────
client = Client()

def fetch_recent_klines(lookback: int) -> pd.DataFrame:
    limit = lookback + 100
    klines = client.get_klines(symbol=TICKER, interval=INTERVAL, limit=limit)
    cols = [
        "Open time","Open","High","Low","Close","Volume","Close time",
        "Quote vol","Trades","Taker buy vol","Taker buy quote","Ignore"
    ]
    df = pd.DataFrame(klines, columns=cols)
    df = df.astype({c: float for c in ["Open","High","Low","Close","Volume"]})
    df["Open time"] = pd.to_datetime(df["Open time"], unit="ms", utc=True)
    df.set_index("Open time", inplace=True)
    return df

FEATURE_COLUMNS = ["Open","High","Low","Close","Volume","log_ret","range"]

def build_feature_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_ret"] = np.log(df["Close"].pct_change() + 1).fillna(0)
    df["range"] = df["High"] - df["Low"]
    return df[FEATURE_COLUMNS]

# ────────────────────── 3. SEQUENCES ──────────────────────
scaler = StandardScaler()

def create_sequences(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, y_reg, y_cls = [], [], []
    idx_close = FEATURE_COLUMNS.index("Close")
    for i in range(LOOK_BACK, len(data) - LOOK_AHEAD_STEPS):
        window = data[i - LOOK_BACK : i]
        target = data[i + LOOK_AHEAD_STEPS - 1, idx_close]
        X.append(window)
        y_reg.append(target)
        y_cls.append(int(target > data[i - 1, idx_close]))
    return np.array(X), np.array(y_reg), np.array(y_cls)

# ────────────────────── 4. MODEL ──────────────────────

def build_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Bidirectional(LSTM(64, return_sequences=True))(inputs)
    x = Dropout(0.3)(x)
    x = Attention(use_scale=True)([x, x])
    x = LayerNormalization()(x)
    x = Flatten()(x)
    reg_out = Dense(1, name="reg_out")(Dense(32, activation="relu")(x))
    cls_out = Dense(1, activation="sigmoid", name="cls_out")(Dense(32, activation="relu")(x))
    model = Model(inputs, [reg_out, cls_out])
    model.compile(
        optimizer="adam",
        loss={"reg_out": "mse", "cls_out": "binary_crossentropy"},
        metrics={"cls_out": "accuracy"},
    )
    return model

model = None

# ─────────────── 5. PERSISTENCIA / ENTRENAMIENTO ───────────────
async def train_or_load():
    global model, scaler
    path = os.path.join(MODEL_DIR, f"btc_l{LOOK_BACK}_h{LOOK_AHEAD_STEPS}.h5")
    df_hist = fetch_recent_klines(LOOK_BACK + 12000)
    scaler.fit(build_feature_df(df_hist))
    if os.path.exists(path):
        mdl = load_model(path, compile=False)
        if mdl.input_shape[1] == LOOK_BACK:
            mdl.compile(
                optimizer="adam",
                loss={"reg_out": "mse", "cls_out": "binary_crossentropy"},
                metrics={"cls_out": "accuracy"},
            )
            model = mdl
            print("✓ Modelo cargado de disco – forma compatible")
            return
        print("⚠ Modelo existente incompatible → re‑entrenando…")
    data_scaled = scaler.transform(build_feature_df(df_hist))
    X, y_r, y_c = create_sequences(data_scaled)
    split = int(TRAIN_SPLIT_RATIO * len(X))
    X_tr, y_r_tr, y_c_tr = X[:split], y_r[:split], y_c[:split]
    X_val, y_r_val, y_c_val = X[split:], y_r[split:], y_c[split:]
    model = build_model((LOOK_BACK, len(FEATURE_COLUMNS)))
    callbacks = [EarlyStopping(patience=3, restore_best_weights=True), ModelCheckpoint(path, save_best_only=True)]
    print("⌛ Entrenando modelo (1 h)…")
    model.fit(X_tr, {"reg_out": y_r_tr, "cls_out": y_c_tr}, validation_data=(X_val, {"reg_out": y_r_val, "cls_out": y_c_val}), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2, callbacks=callbacks)
    print("✓ Entrenamiento completo")

# ─────────────── 6. PREDICCIÓN & MONITOREO ───────────────
current_pred = {"price": None, "direction": None, "made_at": None, "target_time": None, "confidence": None}
last_monitor_alert = datetime.min.replace(tzinfo=timezone.utc)

@tf.function(reduce_retracing=True)
def _infer(seq):
    return model(seq, training=False)

async def job_hourly_prediction():
    global current_pred
    df = fetch_recent_klines(LOOK_BACK + 1)
    feat = build_feature_df(df)
    seq = scaler.transform(feat)[-LOOK_BACK:].reshape((1, LOOK_BACK, len(FEATURE_COLUMNS)))
    pred_scaled, prob_up = _infer(seq)
    pred_scaled, prob_up = float(pred_scaled[0][0]), float(prob_up[0][0])
    idx_close = FEATURE_COLUMNS.index("Close")
    last_close = feat.iloc[-1]["Close"]
    pred_price = scaler.mean_[idx_close] + scaler.scale_[idx_close] * pred_scaled
    change = pred_price - last_close
    pct = change / last_close * 100
    up = change > 0
    conf = prob_up * 100 if up else (100 - prob_up * 100)
    now_utc = datetime.now(timezone.utc)
    current_pred = {"price": pred_price, "direction": up, "made_at": now_utc, "target_time": now_utc + timedelta(minutes=PRED_HORIZON_MIN), "confidence": conf}
    dir_symbol = "↑" if up else "↓"
    msg = f"BTC predicción base 1 h (desde {now_utc:%H:%M UTC})\n{dir_symbol} ${pred_price:,.0f} ({pct:+.2f}%) | Conf: {conf:.1f}%"
    notification.notify(title="BTC 1 h", message=msg, timeout=5)
    ntfy_send(msg, priority="high" if conf > 70 else "default")
    print("[Pred]", msg.replace("\n", " | "))

async def job_monitor_deviation():
    global last_monitor_alert, current_pred
    if current_pred["price"] is None or datetime.now(timezone.utc) >= current_pred["target_time"]:
        return
    current_price = fetch_recent_klines(1).iloc[-1]["Close"]
    diff_pct = (current_price - current_pred["price"]) / current_pred["price"] * 100
    if abs(diff_pct) < SIGNIFICANT_PCT:
        return
    if (datetime.now(timezone.utc) - last_monitor_alert).total_seconds() < COOLDOWN_MIN_MONITOR * 60:
        return
    last_monitor_alert = datetime.now(timezone.utc)
    dir_symbol = "▲" if diff_pct > 0 else "▼"
    msg = f"BTC desvío {dir_symbol} {diff_pct:+.2f}% respecto a predicción 1 h\nPrecio actual: ${current_price:,.0f} | Target: ${current_pred['price']:,.0f}"
    notification.notify(title="BTC desvío 1 h", message=msg, timeout=5)
    ntfy_send(msg)
    print("[Dev]", msg.replace("\n", " | "))

# ─────────────── 7. MAIN ───────────────
async def main():
    warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
    await train_or_load()
    scheduler = AsyncIOScheduler()
    await job_hourly_prediction()
    scheduler.add_job(job_hourly_prediction, "interval", hours=1, max_instances=1, coalesce=True)
    scheduler.add_job(job_monitor_deviation, "interval", seconds=MONITOR_FREQ_SEC, max_instances=1, coalesce=True)
    scheduler.add_job(train_or_load, "interval", hours=RETRAIN_EVERY_HRS)
    scheduler.start()
    print(f"✅ Bot 1 h activo. ntfy.sh/{NTFY_TOPIC}\n")
    while True:
        await asyncio.sleep(3600)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except RuntimeError:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())
    except (KeyboardInterrupt, SystemExit):
        print("Adiós 👋")
