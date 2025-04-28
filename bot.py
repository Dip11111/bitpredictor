import os, sys, asyncio, warnings, signal, math, time
from datetime import datetime, timedelta, timezone
from typing import Tuple

import uvloop
import numpy as np
import pandas as pd
import requests
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Bidirectional, Input, Attention,
    LayerNormalization, Flatten
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from pybit.unified_trading import HTTP  # Bybit SDK

# ─────────────── Configuración básica ───────────────
TICKER               = os.getenv("TICKER",              "BTCUSDT")
INTERVAL             = int(os.getenv("INTERVAL",               5))
LOOK_BACK            = int(os.getenv("LOOK_BACK",             60))
PRED_HORIZON_MIN     = int(os.getenv("PRED_MIN",              60))
LOOK_AHEAD_STEPS     = PRED_HORIZON_MIN // INTERVAL
MONITOR_FREQ_SEC     = int(os.getenv("MONITOR_SEC",          300))
SIGNIFICANT_PCT      = float(os.getenv("SIGNIF_PCT",         0.6))
TRAIN_SPLIT_RATIO    = float(os.getenv("TRAIN_SPLIT",        0.8))
EPOCHS               = int(os.getenv("EPOCHS",               10))
BATCH_SIZE           = int(os.getenv("BATCH_SIZE",          128))
RETRAIN_EVERY_HRS    = int(os.getenv("RETRAIN_HRS",          12))
COOLDOWN_MIN_MONITOR = int(os.getenv("COOLDOWN_MIN",         10))
DO_TRAIN             = os.getenv("TRAIN", "1") != "0"
MODEL_DIR            = "/workspace/models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ntfy.sh notifications
NTFY_TOPIC = os.getenv("NTFY_TOPIC", "mente-sardina")
NTFY_URL   = f"https://ntfy.sh/{NTFY_TOPIC}".rstrip("/")

def ntfy_send(text: str, priority: str = "default"):
    try:
        requests.post(NTFY_URL, data=text.encode(), headers={"Priority": priority}, timeout=10)
    except Exception as e:
        print(f"[ntfy] {e}", file=sys.stderr)

# Use uvloop for better performance
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# ────────────────────── 1. DATA (Bybit) ──────────────────────
session = HTTP(testnet=False, timeout=10)

def fetch_recent_klines(needed: int) -> pd.DataFrame:
    frames, remaining = [], needed
    cursor = None
    while remaining > 0:
        limit = min(remaining, 1000)
        try:
            res = session.get_kline(
                category="linear", symbol=TICKER,
                interval=str(INTERVAL), limit=limit,
                cursor=cursor
            )
        except Exception as e:
            print(f"[Bybit] {e} – retrying in 2s", file=sys.stderr)
            time.sleep(2)
            continue
        data = res["result"]["list"]
        frames.append(pd.DataFrame(data))
        remaining -= limit
        cursor = res["result"].get("nextPageCursor")
        if not cursor:
            break
    if not frames:
        raise RuntimeError("Bybit returned no data")
    df = pd.concat(frames, ignore_index=True)
    df.columns = ["Open time","Open","High","Low","Close","Volume","Turnover"]
    df = df.astype({c: float for c in ["Open","High","Low","Close","Volume"]})
    df["Open time"] = pd.to_datetime(pd.to_numeric(df["Open time"]), unit="ms", utc=True)
    df.set_index("Open time", inplace=True)
    return df

# ────────────────────── 2. FEATURES ──────────────────────
FEATURE_COLUMNS = ["Open","High","Low","Close","Volume","log_ret","range"]

def build_feature_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_ret"] = np.log(df["Close"].pct_change() + 1).fillna(0)
    df["range"]   = df["High"] - df["Low"]
    return df[FEATURE_COLUMNS]

# ────────────────────── 3. SEQUENCES ──────────────────────
scaler = StandardScaler()

def create_sequences(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, y_reg, y_cls = [], [], []
    idx_close = FEATURE_COLUMNS.index("Close")
    for i in range(LOOK_BACK, len(arr) - LOOK_AHEAD_STEPS):
        window = arr[i-LOOK_BACK:i]
        target = arr[i+LOOK_AHEAD_STEPS-1, idx_close]
        X.append(window)
        y_reg.append(target)
        y_cls.append(int(target > arr[i-1, idx_close]))
    return np.asarray(X), np.asarray(y_reg), np.asarray(y_cls)

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
    model.compile(optimizer="adam", loss={"reg_out":"mse", "cls_out":"binary_crossentropy"}, metrics={"cls_out":"accuracy"})
    return model

model = None

# ────────────────────── 5. TRAIN / LOAD ──────────────────────
async def train_or_load():
    global model, scaler
    path = os.path.join(MODEL_DIR, f"btc_l{LOOK_BACK}_h{LOOK_AHEAD_STEPS}.h5")
    df_hist = fetch_recent_klines(LOOK_BACK + 12000)
    scaler.fit(build_feature_df(df_hist))
    if os.path.exists(path):
        try:
            mdl = load_model(path, compile=False)
            if mdl.input_shape[1] == LOOK_BACK:
                mdl.compile(optimizer="adam", loss={"reg_out":"mse","cls_out":"binary_crossentropy"}, metrics={"cls_out":"accuracy"})
                model = mdl
                print("✓ Loaded model from disk")
                return
        except Exception:
            print("⚠ Failed to load model, retraining…")
    if not DO_TRAIN:
        raise RuntimeError("Model missing and TRAIN=0")
    data = scaler.transform(build_feature_df(df_hist))
    X, y_r, y_c = create_sequences(data)
    split = math.floor(TRAIN_SPLIT_RATIO * len(X))
    X_tr, y_r_tr, y_c_tr = X[:split], y_r[:split], y_c[:split]
    X_val, y_r_val, y_c_val = X[split:], y_r[split:], y_c[split:]
    model = build_model((LOOK_BACK, len(FEATURE_COLUMNS)))
    callbacks = [EarlyStopping(patience=3, restore_best_weights=True), ModelCheckpoint(path, save_best_only=True)]
    print(f"⌛ Training on {len(X_tr):,} samples…")
    model.fit(X_tr, {"reg_out":y_r_tr, "cls_out":y_c_tr}, validation_data=(X_val,{"reg_out":y_r_val,"cls_out":y_c_val}), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2, callbacks=callbacks)
    print("✓ Training complete")

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
    p_s, p_up = map(float, _infer(seq))
    idx = FEATURE_COLUMNS.index("Close")
    last_close = feat.iloc[-1]["Close"]
    pred_price = scaler.mean_[idx] + scaler.scale_[idx] * p_s
    pct = (pred_price - last_close) / last_close * 100
    now = datetime.now(timezone.utc)
    current_pred = {"price": pred_price, "direction": pct>0, "target_time": now + timedelta(minutes=PRED_HORIZON_MIN)}
    arrow = "↑" if pct>0 else "↓"
    msg = f"BTC 1h ({now:%H:%M UTC}) {arrow} ${pred_price:,.0f} ({pct:+.2f}%)"
    ntfy_send(msg, priority="high")
    print("[Pred]", msg, flush=True)

async def job_monitor_deviation():
    global last_mon, current_pred
    if current_pred["price"] is None or datetime.now(timezone.utc) >= current_pred["target_time"]:
        return
    price = fetch_recent_klines(1).iloc[-1]["Close"]
    diff = (price - current_pred["price"]) / current_pred["price"] * 100
    if abs(diff) < SIGNIFICANT_PCT or (datetime.now(timezone.utc) - last_mon).total_seconds() < COOLDOWN_MIN_MONITOR*60:
        return
    last_mon = datetime.now(timezone.utc)
    arr = "▲" if diff>0 else "▼"
    msg = f"BTC deviation {arr} {diff:+.2f}% | ${price:,.0f} vs ${current_pred['price']:,.0f}"
    ntfy_send(msg)
    print("[Dev]", msg, flush=True)

# ────────────────────── 7. MAIN ──────────────────────
async def main():
    warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
    print("🚀 Starting BTC-Bot (Railway)…", flush=True)

    await train_or_load()

    scheduler = AsyncIOScheduler()
    await job_hourly_prediction()
    scheduler.add_job(job_hourly_prediction, 'interval', hours=1, max_instances=1, coalesce=True)
    scheduler.add_job(job_monitor_deviation, 'interval', seconds=MONITOR_FREQ_SEC, max_instances=1, coalesce=True)
    if DO_TRAIN:
        scheduler.add_job(train_or_load, 'interval', hours=RETRAIN_EVERY_HRS)
    scheduler.start()
    print(f"✅ Bot running. NTFY: {NTFY_URL}", flush=True)

    stop = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        asyncio.get_event_loop().add_signal_handler(sig, stop.set)
    await stop.wait()
    print("🔌 Shutting down…", flush=True)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"❌ Fatal error: {e}", file=sys.stderr)
        sys.exit(1)
