"""
The airline solver, in miniature.

Two flights/constraints that are nearly the same produce two nearly-parallel
rows. We solve A x = b (find the 'corner' where the rules meet), then nudge b
by a rounding-error-sized amount and watch the answer LEAP. Finally we show the
QR / condition-number pre-check that catches it before solving.

Run:  python airline_solver_demo.py
Needs: numpy
"""

import sys
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

np.set_printoptions(precision=6, suppress=True)
RULE = "=" * 64


def banner(t):
    print("\n" + RULE + "\n" + t + "\n" + RULE)


def report(name, A, b):
    """Solve A x = b and probe its stability."""
    print(f"\n--- {name} ---")
    c1, c2 = A[0], A[1]
    cos = (c1 @ c2) / (np.linalg.norm(c1) * np.linalg.norm(c2))
    ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    print(f"angle between the two constraint rows : {ang:6.3f} deg")
    print(f"det(A)                                : {np.linalg.det(A):.6e}")
    print(f"condition number cond(A)              : {np.linalg.cond(A):,.1f}")

    x = np.linalg.solve(A, b)
    b2 = b + np.array([0.0, 1e-4])          # nudge one entry by 0.0001
    x2 = np.linalg.solve(A, b2)
    print(f"solution x                            : {x}")
    print(f"solution after nudging b by 1e-4      : {x2}")
    print(f"=> answer moved by                    : {np.linalg.norm(x2 - x):.4f}")


# ----------------------------------------------------------------------
banner("CASE A: two SENSIBLE flights (rows point different ways)")
# Constraint rows are far from parallel -> sharp, stable corner.
A_good = np.array([[1.0, 0.0],
                   [0.0, 1.0]])
b_good = np.array([2.0, 2.0])
report("well-conditioned", A_good, b_good)

# ----------------------------------------------------------------------
banner("CASE B: two LOOK-ALIKE layovers (rows nearly parallel)")
# Rows differ only in the 4th decimal -> near-duplicate wall.
A_bad = np.array([[1.0, 1.0],
                  [1.0, 1.0001]])
b_bad = np.array([2.0, 2.0001])
report("ill-conditioned", A_bad, b_bad)

# ----------------------------------------------------------------------
banner("THE PRE-CHECK: catch CASE B before the solver runs")
for name, A in [("CASE A", A_good), ("CASE B", A_bad)]:
    Q, R = np.linalg.qr(A)
    diag = np.abs(np.diag(R))
    smallest = diag.min()
    verdict = "DUPLICATE ROW -- drop/merge!" if smallest < 1e-3 else "ok"
    print(f"{name}: min |diag(R)| = {smallest:.6f}  ->  {verdict}")

banner("DONE")
print("Lesson: cos(theta)->1, det->0, cond->huge are three names for the")
print("same danger. The QR diagonal names WHICH row to blame, cheaply.")
