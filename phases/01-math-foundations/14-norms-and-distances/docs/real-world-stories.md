# Norms & Distances — Real-World Stories

> "Similar" is meaningless until you pick a distance. Picking the wrong one is a bug, not a style choice.

## The Mental Model

Different norms answer different questions:

| Norm | Measures | When to use |
|---|---|---|
| L1 (Manhattan) | Sum of \|differences\| | Robust to outliers; sparse solutions |
| L2 (Euclidean) | Root-sum-squares | Smooth optimization; assume Gaussian noise |
| L∞ | Max difference | Worst-case bounds |
| Cosine | Angle | Direction matters, magnitude doesn't |
| Mahalanobis | Scaled by covariance | Features on different scales / correlated |

```mermaid
flowchart LR
    A["Compare two items"] --> B{What property?}
    B -->|"Direction matters"| C[Cosine]
    B -->|"Magnitude matters"| D[L2]
    B -->|"Features on different scales"| E[Mahalanobis]
    B -->|"Outlier-robust"| F[L1]
    style E fill:#ffd966
```

## Code: Five Distances on the Same Pair

```python
import numpy as np

a = np.array([100.0, 0.5, 3])    # e.g., price $, duration_hrs, stops
b = np.array([110.0, 0.6, 4])

# Raw L1, L2, L∞
print("L1 :", np.sum(np.abs(a - b)))
print("L2 :", np.linalg.norm(a - b))
print("L∞ :", np.max(np.abs(a - b)))

# Cosine
print("cos:", a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

# Mahalanobis with covariance from data
data = np.random.randn(1000, 3) * np.array([50, 0.3, 1.5])
S_inv = np.linalg.inv(np.cov(data, rowvar=False))
diff = a - b
print("mhl:", np.sqrt(diff @ S_inv @ diff))
```

## Code: Why L2 on Raw Features is Wrong

```python
# Without normalization, the $ axis dwarfs everything else
a = np.array([100.0, 0.5, 3])
b = np.array([100.0, 5.0, 3])   # huge difference in hours

c = np.array([110.0, 0.5, 3])   # $10 difference, same hours

print("L2(a,b) =", np.linalg.norm(a - b))   # 4.5
print("L2(a,c) =", np.linalg.norm(a - c))   # 10 — but should feel "closer" to a
```

## Amazon — Visual Search

"Show me products that look like this picture" runs on embedding similarity. The fashion team A/B-tested cosine vs L2-on-normalized vectors. L2 won — because magnitude encoded *popularity* in the embedding, and customers preferred popular look-alikes. A distance choice that looked like a style decision moved conversion measurably.

## American Airlines — "Find Me a Similar Flight"

L1 distance on `(price_$, duration_min, stops)` treats $50 the same as 50 minutes — nonsense. AA.com uses Mahalanobis with the covariance learned from actual booking history. The result respects how customers *actually* trade off these features. A user who values cheap-and-short gets a different "similar" set than one who values direct flights at any price.

## Takeaways

- Distance is a modeling choice. State it explicitly.
- Normalize *before* using L2 — or use Mahalanobis to bake the scaling in.
- Cosine ignores magnitude. Sometimes that's exactly right; sometimes it's a bug.
