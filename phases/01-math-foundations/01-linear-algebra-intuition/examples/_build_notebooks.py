"""
Build + execute the two demo notebooks so outputs are baked into the cells.
Run once:  python _build_notebooks.py
"""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from nbconvert.preprocessors import ExecutePreprocessor

HERE = __import__("pathlib").Path(__file__).parent


def build(filename, cells):
    nb = new_notebook()
    nb.cells = [
        new_markdown_cell(c[1]) if c[0] == "md" else new_code_cell(c[1])
        for c in cells
    ]
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3", "language": "python", "name": "python3"
    }
    ep = ExecutePreprocessor(timeout=120, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": str(HERE)}})
    out = HERE / filename
    nbf.write(nb, out)
    print("wrote", out)


# ======================================================================
# NOTEBOOK 1 — airline_solver_demo.ipynb
# ======================================================================
airline = [
    ("md", "# The Airline Solver, in Miniature\n\n"
           "Two flight constraints. When they point different ways the schedule "
           "solver is stable; when they're **near-parallel look-alike layovers** "
           "a rounding-error-sized change makes the answer *leap*.\n\n"
           "Companion to "
           "[`american-airlines-near-parallel-constraints.md`]"
           "(../docs/american-airlines-near-parallel-constraints.md)."),
    ("code", "import numpy as np\n"
             "np.set_printoptions(precision=6, suppress=True)\n\n"
             "def report(name, A, b):\n"
             "    c1, c2 = A[0], A[1]\n"
             "    cos = (c1 @ c2) / (np.linalg.norm(c1) * np.linalg.norm(c2))\n"
             "    ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))\n"
             "    print(f'--- {name} ---')\n"
             "    print(f'angle between rows : {ang:8.3f} deg')\n"
             "    print(f'det(A)             : {np.linalg.det(A):.6e}')\n"
             "    print(f'cond(A)            : {np.linalg.cond(A):,.1f}')\n"
             "    x  = np.linalg.solve(A, b)\n"
             "    x2 = np.linalg.solve(A, b + np.array([0.0, 1e-4]))\n"
             "    print(f'x                  : {x}')\n"
             "    print(f'x after 1e-4 nudge : {x2}')\n"
             "    print(f'answer moved by    : {np.linalg.norm(x2 - x):.4f}')"),
    ("md", "## Case A — two sensible flights (rows perpendicular)\n\n"
           "Rows point different ways → one sharp, stable corner."),
    ("code", "A_good = np.array([[1., 0.],\n"
             "                   [0., 1.]])\n"
             "b_good = np.array([2., 2.])\n"
             "report('well-conditioned', A_good, b_good)"),
    ("md", "## Case B — two look-alike layovers (rows nearly parallel)\n\n"
           "Rows differ only in the 4th decimal → a near-duplicate wall. Watch "
           "the answer move **14,000×** the size of the input nudge."),
    ("code", "A_bad = np.array([[1., 1.],\n"
             "                  [1., 1.0001]])\n"
             "b_bad = np.array([2., 2.0001])\n"
             "report('ill-conditioned', A_bad, b_bad)"),
    ("md", "## The pre-check: catch Case B *before* solving\n\n"
           "A near-zero entry on the diagonal of **R** (from `A = QR`) means that "
           "row adds almost no new direction — a duplicate."),
    ("code", "for name, A in [('CASE A', A_good), ('CASE B', A_bad)]:\n"
             "    Q, R = np.linalg.qr(A)\n"
             "    smallest = np.abs(np.diag(R)).min()\n"
             "    verdict = 'DUPLICATE ROW -- drop/merge!' if smallest < 1e-3 else 'ok'\n"
             "    print(f'{name}: min |diag(R)| = {smallest:.6f}  ->  {verdict}')"),
    ("md", "### Read it\n\n"
           "`angle → 0°`, `det → 0`, `cond → huge` are **three names for the same "
           "danger**. A `0.0001` input error became a `1.4` output error. That is "
           "exactly what *'the solver went wobbly'* feels like."),
]

# ======================================================================
# NOTEBOOK 2 — near_parallel_demo.ipynb
# ======================================================================
longley = [
    ("md", "# Near-Parallel Features on Real Data (Longley, 1947-1962)\n\n"
           "The same instability from the airline solver, but inside an ordinary "
           "**linear regression**. We pull the Longley economic dataset live from "
           "the internet — the textbook multicollinearity case where `GNP`, "
           "`Population`, and `Year` all rise together."),
    ("code", "import numpy as np, pandas as pd, requests\n"
             "from io import StringIO\n"
             "from sklearn.linear_model import LinearRegression\n"
             "np.set_printoptions(precision=4, suppress=True)\n\n"
             "URL = ('https://vincentarelbundock.github.io/'\n"
             "       'Rdatasets/csv/datasets/longley.csv')\n"
             "df = pd.read_csv(StringIO(requests.get(URL, timeout=30).text))\n"
             "df = df.drop(columns=['rownames'])\n"
             "df.head()"),
    ("code", "FEATURES = ['GNP.deflator','GNP','Unemployed',\n"
             "            'Armed.Forces','Population','Year']\n"
             "TARGET = 'Employed'\n"
             "X = df[FEATURES].to_numpy(float)\n"
             "y = df[TARGET].to_numpy(float)\n"
             "print('X shape:', X.shape, '(only 16 rows -> the bug is LOUD)')"),
    ("md", "## 1. A feature is a vector — are any two nearly parallel?\n\n"
           "Center each column (compare *shape*, not the fact they're all "
           "positive), then measure the angle. `cos → 1` means parallel."),
    ("code", "Xc = X - X.mean(axis=0)\n"
             "def cosine(a, b):\n"
             "    return (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))\n\n"
             "print(f\"{'pair':<28}{'cos':>10}{'angle(deg)':>12}\")\n"
             "for i in range(len(FEATURES)):\n"
             "    for j in range(i+1, len(FEATURES)):\n"
             "        c = cosine(Xc[:, i], Xc[:, j])\n"
             "        if abs(c) > 0.99:\n"
             "            ang = np.degrees(np.arccos(np.clip(c, -1, 1)))\n"
             "            print(f'{FEATURES[i]+\" vs \"+FEATURES[j]:<28}{c:>10.4f}{ang:>12.2f}')"),
    ("md", "## 2. Condition number — how ill-conditioned is the whole matrix?\n\n"
           "Rule of thumb: `>1e3` shaky, `>1e6` dangerous."),
    ("code", "Xdesign = np.column_stack([np.ones(len(X)), X])\n"
             "cond = np.linalg.cond(Xdesign)\n"
             "print(f'cond(X) = {cond:,.0f}')\n"
             "print(f'~ you lose about {np.log10(cond):.0f} of ~16 decimal digits')"),
    ("md", "## 3. QR pre-check — *which* columns are the culprits?\n\n"
           "Small `|diag(R)|` = that column adds almost no new direction."),
    ("code", "Q, R = np.linalg.qr(Xdesign)\n"
             "diag = np.abs(np.diag(R))\n"
             "names = ['intercept'] + FEATURES\n"
             "for n, d in sorted(zip(names, diag), key=lambda t: t[1]):\n"
             "    flag = '  <-- redundant direction' if d < 0.1*diag.max() else ''\n"
             "    print(f'{n:<16}|R_ii| = {d:>12.4f}{flag}')"),
    ("md", "## 4. The payoff — nudge the data 0.1%, watch coefficients move\n\n"
           "Standardize first so every coefficient is on the same scale. The "
           "condition number *is* the amplification factor."),
    ("code", "Xz = (X - X.mean(0)) / X.std(0)\n"
             "rng = np.random.default_rng(0)\n\n"
             "def fit_coefs(Xmat, jitter=0.0):\n"
             "    if jitter:\n"
             "        Xmat = Xmat + rng.normal(0, jitter, Xmat.shape)\n"
             "    return LinearRegression().fit(Xmat, y).coef_\n\n"
             "before = fit_coefs(Xz)\n"
             "after  = fit_coefs(Xz, jitter=0.001)\n"
             "swing = np.linalg.norm(after-before)/np.linalg.norm(before)\n"
             "print(f\"{'feature':<16}{'clean':>12}{'nudged':>12}{'abs change':>12}\")\n"
             "for n,a,b in zip(FEATURES, before, after):\n"
             "    print(f'{n:<16}{a:>12.3f}{b:>12.3f}{abs(b-a):>12.3f}')\n"
             "print(f'\\noverall coefficient vector moved {swing:.0%} from a 0.1% nudge')"),
    ("md", "## 5. The fix — drop the near-parallel columns\n\n"
           "Keep one trend (`Year`), drop the rest (`GNP`, `Population`, "
           "`GNP.deflator`). The condition number collapses and the model "
           "becomes stable."),
    ("code", "KEEP = ['Unemployed', 'Armed.Forces', 'Year']\n"
             "ki = [FEATURES.index(k) for k in KEEP]\n"
             "Xkz = Xz[:, ki]\n\n"
             "cond_b = np.linalg.cond(np.column_stack([np.ones(len(Xz)), Xz]))\n"
             "cond_a = np.linalg.cond(np.column_stack([np.ones(len(Xkz)), Xkz]))\n"
             "print(f'cond (standardized) BEFORE : {cond_b:,.0f}')\n"
             "print(f'cond (standardized) AFTER  : {cond_a:,.1f}')\n\n"
             "c0 = fit_coefs(Xkz)\n"
             "c1 = fit_coefs(Xkz, jitter=0.001)\n"
             "sw = np.linalg.norm(c1-c0)/np.linalg.norm(c0)\n"
             "print(f\"\\n{'feature':<16}{'clean':>12}{'nudged':>12}{'abs change':>12}\")\n"
             "for n,a,b in zip(KEEP, c0, c1):\n"
             "    print(f'{n:<16}{a:>12.3f}{b:>12.3f}{abs(b-a):>12.3f}')\n"
             "print(f'\\noverall coefficient vector moved only {sw:.1%} now')"),
    ("md", "### Read it\n\n"
           "Same data, same algorithm. The only change is **removing the "
           "duplicate directions you can now see**. `cond` falls from ~111 to "
           "~3.7 and the model stops lying."),
]

build("airline_solver_demo.ipynb", airline)
build("near_parallel_demo.ipynb", longley)
print("\nAll notebooks built and executed.")
