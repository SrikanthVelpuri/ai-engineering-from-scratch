"""
Near-parallel constraint/feature vectors — a runnable demo on a REAL dataset.

Story: American Airlines' scheduler went wobbly when two flights produced
near-parallel constraint vectors. The same math (ill-conditioning) bites any
regression with near-duplicate features. We prove it on the Longley dataset —
the textbook example of multicollinearity (GNP, Population, Year all rise
together over time, so their column vectors nearly point the same way).

Run:  python near_parallel_demo.py
Needs: numpy, pandas, requests, scikit-learn
"""

import sys
import numpy as np
import pandas as pd
import requests
from io import StringIO
from sklearn.linear_model import LinearRegression

# Windows consoles default to cp1252; force UTF-8 so symbols print.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

np.set_printoptions(precision=4, suppress=True)
RULE = "=" * 70


def banner(title):
    print("\n" + RULE)
    print(title)
    print(RULE)


# ----------------------------------------------------------------------
# 0. Pull a real dataset from the internet
# ----------------------------------------------------------------------
banner("0. FETCH REAL DATA  (Longley economic dataset, 1947-1962)")

URL = "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/longley.csv"
text = requests.get(URL, timeout=30).text
df = pd.read_csv(StringIO(text)).drop(columns=["rownames"])
print(df.head())
print(f"\nshape: {df.shape}  (only 16 rows — small data makes the bug LOUD)")

FEATURES = ["GNP.deflator", "GNP", "Unemployed", "Armed.Forces", "Population", "Year"]
TARGET = "Employed"
X = df[FEATURES].to_numpy(dtype=float)
y = df[TARGET].to_numpy(dtype=float)


# ----------------------------------------------------------------------
# 1. A "feature" is a vector. Are any two nearly parallel?
# ----------------------------------------------------------------------
banner("1. ANGLE BETWEEN FEATURE VECTORS  (cos≈1  =>  nearly parallel)")


def cosine(a, b):
    return (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))


# center each column so we measure SHAPE, not the fact they're all positive
Xc = X - X.mean(axis=0)
print(f"{'pair':<28}{'cos(theta)':>12}{'angle(deg)':>12}")
for i in range(len(FEATURES)):
    for j in range(i + 1, len(FEATURES)):
        c = cosine(Xc[:, i], Xc[:, j])
        ang = np.degrees(np.arccos(np.clip(c, -1, 1)))
        flag = "  <-- NEARLY PARALLEL" if abs(c) > 0.99 else ""
        if abs(c) > 0.9:
            print(f"{FEATURES[i]+' vs '+FEATURES[j]:<28}{c:>12.4f}{ang:>12.2f}{flag}")


# ----------------------------------------------------------------------
# 2. Condition number: HOW ill-conditioned is the whole matrix?
# ----------------------------------------------------------------------
banner("2. CONDITION NUMBER  (rule of thumb: >1e3 shaky, >1e6 dangerous)")

# add the intercept column the way a regression actually builds the matrix
Xdesign = np.column_stack([np.ones(len(X)), X])
cond = np.linalg.cond(Xdesign)
print(f"cond(X) = {cond:,.0f}")
lost_digits = np.log10(cond)
print(f"~ you lose about {lost_digits:.0f} of your ~16 decimal digits of precision")


# ----------------------------------------------------------------------
# 3. QR pre-check: WHICH columns are the culprits?
# ----------------------------------------------------------------------
banner("3. QR PRE-CHECK  (tiny |diag(R)| = this column adds almost nothing new)")

Q, R = np.linalg.qr(Xdesign)
diag = np.abs(np.diag(R))
names = ["intercept"] + FEATURES
for n, d in sorted(zip(names, diag), key=lambda t: t[1]):
    flag = "  <-- redundant direction" if d < 1e-1 * diag.max() else ""
    print(f"{n:<16}|R_ii| = {d:>12.4f}{flag}")


# ----------------------------------------------------------------------
# 4. The payoff: tiny data change -> huge coefficient swing
# ----------------------------------------------------------------------
banner("4. INSTABILITY  (perturb data by 0.1% -> coefficients LEAP)")

# Standardize so every coefficient is on the same scale (z-scores).
# This isolates COLLINEARITY from mere unit-of-measure differences.
Xz = (X - X.mean(axis=0)) / X.std(axis=0)
rng = np.random.default_rng(0)


def fit_coefs(Xmat, jitter=0.0):
    if jitter:
        Xmat = Xmat + rng.normal(0, jitter, Xmat.shape)
    return LinearRegression().fit(Xmat, y).coef_


coef_before = fit_coefs(Xz)
coef_after = fit_coefs(Xz, jitter=0.001)  # 0.1% rounding-error-sized nudge
swing = np.linalg.norm(coef_after - coef_before) / np.linalg.norm(coef_before)

print(f"{'feature':<16}{'coef (clean)':>16}{'coef (nudged)':>16}{'abs change':>12}")
for n, a, b in zip(FEATURES, coef_before, coef_after):
    print(f"{n:<16}{a:>16.3f}{b:>16.3f}{abs(b-a):>12.3f}")
print(f"\noverall coefficient vector moved {swing:.0%} from a 0.1% data nudge.")
print("The model 'fits' both times, but the story it tells is unstable.")


# ----------------------------------------------------------------------
# 5. The fix: drop the near-duplicate columns, re-check
# ----------------------------------------------------------------------
banner("5. THE FIX  (drop near-parallel columns -> stability returns)")

KEEP = ["Unemployed", "Armed.Forces", "Year"]  # 1 trend kept; GNP/Pop/defl dropped
ki = [FEATURES.index(k) for k in KEEP]
Xkz = Xz[:, ki]

cond_before = np.linalg.cond(np.column_stack([np.ones(len(Xz)), Xz]))
cond_after = np.linalg.cond(np.column_stack([np.ones(len(Xkz)), Xkz]))
print(f"kept features        : {KEEP}")
print(f"cond (standardized)  BEFORE : {cond_before:,.0f}")
print(f"cond (standardized)  AFTER  : {cond_after:,.1f}")

c_clean = fit_coefs(Xkz)
c_nudge = fit_coefs(Xkz, jitter=0.001)
swing_k = np.linalg.norm(c_nudge - c_clean) / np.linalg.norm(c_clean)
print(f"\n{'feature':<16}{'coef (clean)':>16}{'coef (nudged)':>16}{'abs change':>12}")
for n, a, b in zip(KEEP, c_clean, c_nudge):
    print(f"{n:<16}{a:>16.3f}{b:>16.3f}{abs(b-a):>12.3f}")
print(f"\noverall coefficient vector moved only {swing_k:.1%} now. Stable + trustworthy.")

banner("DONE")
