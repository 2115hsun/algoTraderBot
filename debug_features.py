import pandas as pd, numpy as np, json, joblib
import config
from strategies.base import ffm_block, adx_pair, FFM_COLS

print(f"FFM_COLS len: {len(FFM_COLS)}")

n = 500
bars = pd.DataFrame({
    "time":   pd.date_range("2024-01-01", periods=n, freq="2min", tz="UTC"),
    "open":  np.random.uniform(20000, 21000, n),
    "high":  np.random.uniform(20000, 21000, n),
    "low":   np.random.uniform(20000, 21000, n),
    "close": np.random.uniform(20000, 21000, n),
    "volume":np.random.uniform(100, 1000, n),
})

i = n - 1
ffm = ffm_block(bars, i)
print(f"ffm_block returned shape: {ffm.shape}")
print(f"  non-NaN count: {(~np.isnan(ffm)).sum()}")
print(f"  NaN count:    {np.isnan(ffm).sum()}")

adx_i, adx_slope = adx_pair(bars, i)
print(f"adx_pair: ({adx_i}, {adx_slope})")

hand = np.concatenate([ffm, [adx_i, adx_slope]]).astype(np.float32)
print(f"hand vector shape: {hand.shape}")

embed = np.random.randn(263).astype(np.float32)
X = np.concatenate([embed, hand]).reshape(1, -1)
print(f"Full X shape: {X.shape}")

bundle = joblib.load("models/supertrend_chronos.joblib")
print(f"Bundle keys: {list(bundle.keys())}")

try:
    proba = bundle["signal_head"].predict_proba(X)
    print(f"SUCCESS: proba shape {proba.shape}")
except Exception as e:
    print(f"FAILED: {e}")
