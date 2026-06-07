# Real-World Stories — Math Foundations Cheatsheet

> One page to remember all 18 stories. Each topic anchors to two real production bugs (or wins) — one at Amazon, one at American Airlines — so the math sticks.

## How to use this sheet

1. **First pass:** read top to bottom. Don't worry about the math — just remember the *story*.
2. **When studying a phase:** come back here first and ask "what bug does this math prevent?"
3. **Before an interview / review:** scan the **Hook** column. Each hook is a single sentence that should snap the whole story back into your head.

---

## The 18 stories at a glance

| # | Topic | Hook (one sentence to remember it by) |
| --- | --- | --- |
| 01 | Linear algebra intuition | "Modifier perpendicular to base → signal lost." |
| 02 | Vectors & matrices ops | "Same matmul, different memory pattern, 100x speed gap." |
| 03 | Matrix transformations | "Wrong inverse → drones inspect the wrong panel." |
| 04 | Calculus for ML | "Gradients rank leverage, not loudness." |
| 05 | Chain rule & autodiff | "A misplaced .detach() makes the policy learn nothing." |
| 06 | Probability & distributions | "Heavy-tailed data + t-test = ship noise as wins." |
| 07 | Bayes' theorem | "The prior moves more than the model does." |
| 08 | Optimization | "Wrong optimizer doesn't crash — it converges to a worse answer." |
| 09 | Information theory | "Recommendation entropy drops → the homepage feels broken." |
| 10 | Dimensionality reduction | "Compress to see; the residual is where anomalies hide." |
| 11 | SVD | "The elbow in the singular values is where to cut." |
| 12 | Tensor operations | "Wrong axis split = an all-reduce you didn't budget for." |
| 13 | Numerical stability | "Subtracting two near-equal big numbers gives noise, not the answer." |
| 14 | Norms & distances | "Distance is a modeling choice. Cosine vs L2 changed conversion." |
| 15 | Statistics for ML | "1,000 tests at 0.05 = ~50 false 'wins'." |
| 16 | Sampling methods | "Importance sampling lets you A/B test without running an A/B test." |
| 17 | Linear systems | "Always check the condition number before trusting Ax = b." |
| 18 | Convex optimization | "Convex = certifiable. Non-convex = hope." |

---

## The full deck — one card per phase

Each card has the same five parts:

- **What it is:** the math, in one breath
- **Amazon story:** one production scene
- **AA story:** one production scene
- **Why it sticks:** the underlying principle
- **Anchor phrase:** what to mutter to yourself on the way back to the concept

---

### 01 — Linear Algebra Intuition

- **What it is:** Vectors are arrows in space. Dot product = alignment. Orthogonality = independence.
- **Amazon:** Search for "running shoes for flat feet" returned generic running shoes. The modifier vector was nearly perpendicular to the base — naive addition diluted the signal. Cross-attention re-ranker fixed it.
- **AA:** Crew-pairing LP solver became unstable on near-parallel constraint vectors (two look-alike layovers a few hours apart). QR pre-check spots the duplicates before the solver runs.
- **Why it sticks:** When you can *see* the geometry, the bug names itself.
- **Anchor phrase:** *"Modifier perpendicular → signal lost."*

### 02 — Vectors & Matrices Operations

- **What it is:** Matmul is a memory access pattern as much as it is multiplication. BLAS, broadcasting, and algebraic identities are how you go fast.
- **Amazon:** Alexa wake-word detector — rewrite `A @ B` as `(B.T @ A.T).T` so chunks fit in L2 cache. 15% latency drop on every Echo.
- **AA:** Storm grounds DFW. Looping flight-by-flight = 4-hour re-plan. One batched matmul across all flights = seconds.
- **Why it sticks:** Same answer, different access pattern, orders-of-magnitude difference.
- **Anchor phrase:** *"Cache-friendly matmul saves a million devices."*

### 03 — Matrix Transformations

- **What it is:** A matrix is a function from one space to another. Chain them in order. Inverses undo them.
- **Amazon:** Kiva warehouse robots reported "phantom collisions." Root cause: missing inverse meant the shelf offset was applied twice — phantom shelf 60 cm from the real one.
- **AA:** Drone tail inspection. One sign error in the z-axis of the drone pose mirrored every reported crack location. Crews inspected undamaged panels; real cracks hid.
- **Why it sticks:** Order and direction matter. Sanity-check by pushing a known point through the chain.
- **Anchor phrase:** *"One missing inverse, every crack is on the wrong panel."*

### 04 — Calculus for ML

- **What it is:** A gradient ranks how much each input moves the output. Universal sensitivity tool.
- **Amazon:** Prime Day forecasting — best-sellers dominated the gradient, long-tail SKUs were silently underweighted. Grouped gradient clipping rescued the rare-item accuracy.
- **AA:** When ORD melts down, the highest-leverage hub to fix may be a *small upstream* one feeding ORD's connections, not ORD itself. Ops dashboard ranks fires by leverage, not size.
- **Why it sticks:** "Where should I push?" is calculus, not intuition.
- **Anchor phrase:** *"Push where the partial derivative is biggest, not where the noise is loudest."*

### 05 — Chain Rule & Autodiff

- **What it is:** Backprop is the chain rule executed backwards across a graph of ops. PyTorch/JAX automate it; you still need to picture it.
- **Amazon:** Rekognition custom similarity layer — autograd couldn't do it efficiently. Engineer derived the gradient, wrote `backward()`, and ran `gradcheck` to make sure it matched numerical.
- **AA:** RL gate-assignment agent picked nonsensical gates. Bug: a `.detach()` deep in the simulator killed the reward gradient. Walking the autograd graph node-by-node found it.
- **Why it sticks:** Autograd is mechanical chain rule — magic only if you don't know what it's doing.
- **Anchor phrase:** *"`.detach()` is the usual suspect when the model 'isn't learning.'"*

### 06 — Probability & Distributions

- **What it is:** The shape of your uncertainty decides whether differences are signal or noise.
- **Amazon:** Buy Box experiments measured revenue per session (log-normal). Naive t-tests turned a single $50k order into a "significant" win. Estimated 8-figure cost before the team enforced log-transform or rank tests.
- **AA:** Global no-show rate over-books Miami too little, JFK-LHR too much. Per-route Beta-Binomial with hierarchical pooling → ~$50M/year improvement.
- **Why it sticks:** Pick the distribution that matches the data-generating process, not the default.
- **Anchor phrase:** *"Anywhere money is involved, suspect log-normal."*

### 07 — Bayes' Theorem

- **What it is:** Bayes flips a conditional: from P(evidence | hypothesis) to P(hypothesis | evidence). The prior matters — sometimes more than the model.
- **Amazon:** "99% accurate" fraud model with a uniform prior blocks real customers at 100x the right rate. Fix: region- and time-specific priors baked into scoring.
- **AA:** Engine sensor anomaly fires. P(failure | anomaly) ≠ P(anomaly | failure). Bayesian updating across every safe flight lowers the posterior; every anomaly raises it. Grounding decisions come from the posterior, not from raw alerts.
- **Why it sticks:** Without the right prior, even a great model is dangerous.
- **Anchor phrase:** *"Same model, different prior, different action."*

### 08 — Optimization

- **What it is:** Each optimizer assumes a different objective shape. Match them or lose silently.
- **Amazon:** Ads auction bidder. SGD oscillated, Adam over-fit recent hour. Cosine warm restarts keyed to traffic patterns (low LR overnight, restart at morning surge) won.
- **AA:** Route-network "where should we fly next?" — non-convex with integer constraints. Pure gradient methods are wrong. MIP + convex relaxation + Lagrangian decomposition is the right tool.
- **Why it sticks:** Pick by objective shape. The LR schedule is part of the algorithm.
- **Anchor phrase:** *"Wrong optimizer doesn't crash — it converges to a worse answer."*

### 09 — Information Theory

- **What it is:** Entropy = uncertainty. KL = how far one distribution is from another. Mutual info = how much one variable tells you about another.
- **Amazon:** Recommender collapses to the same five items. Track per-customer entropy; when it drops below a threshold, a diversity-aware re-ranker kicks in.
- **AA:** Two similar flights — which difference moves the booking? Mutual information between feature deltas and historical bookings ranks the tiebreakers.
- **Why it sticks:** KL is the right monitor for drift. Mutual info selects features without assuming linearity.
- **Anchor phrase:** *"Cross-entropy is the loss. KL is the alarm. MI is the feature picker."*

### 10 — Dimensionality Reduction

- **What it is:** Find the directions your data actually varies along. Discard the rest.
- **Amazon:** Thousands of behavioral signals per customer → PCA to ~50 components keeps 95% of variance. First 3 components are interpretable (spend, browsing, breadth). The strategy doc writes itself.
- **AA:** ~2,000 telemetry signals per flight → reduce to 20 + one-class SVM. Catches "outside the envelope" before incidents. Also track the residual — anomalies sometimes live in the discarded directions.
- **Why it sticks:** Compression reveals structure that's invisible in raw data.
- **Anchor phrase:** *"Project AND watch the residual."*

### 11 — Singular Value Decomposition

- **What it is:** Any matrix = U Σ Vᵀ. Singular values rank the "importance" of each direction. Truncate at the elbow → best rank-k approximation.
- **Amazon:** Product embeddings, 1024-dim × 600M items. SVD-truncate to rank 256 → ~75% storage saved, recall almost untouched.
- **AA:** Crew-scheduling matrix is huge but ~99% of feasible schedules live in a low-rank subspace. SVD reveals it; optimizer works in compressed coordinates. Overnight instead of days.
- **Why it sticks:** Real-world matrices are almost never full-rank.
- **Anchor phrase:** *"The elbow in the singular values is where to cut."*

### 12 — Tensor Operations

- **What it is:** Tensors are N-D arrays. Operations are contractions along chosen axes. `einsum` makes the axes explicit.
- **Amazon:** Trainium tensor parallelism — splitting the wrong axis of `[batch, seq, hidden]` forces an all-reduce that dominates wall time. Splitting the right axis keeps comms cheap.
- **AA:** Demand modeled as a 5D tensor (origin × dest × DoW × days-out × fare-class). Pricing queries that took minutes as pandas groupbys → milliseconds as tensor contractions on a GPU.
- **Why it sticks:** `einsum` is a load-balancing language, not just notation.
- **Anchor phrase:** *"If you can't write the shape signature, you don't understand the op."*

### 13 — Numerical Stability

- **What it is:** Floats are not real numbers. Subtracting near-equal big numbers, taking `exp` of large values, `log(0)` — all silent killers.
- **Amazon:** Training a big LM in fp16 saves memory but tiny gradients underflow to zero. Loss scaling: multiply the loss by a big factor, unscale before stepping. Fix is one wrapper.
- **AA:** Revenue solver subtracted two ~$1B numbers in fp32; the difference (in cents) was pure noise. Years of forecasts off by tens of millions. Fix: do that one subtraction in fp64.
- **Why it sticks:** "The model just stopped learning" is usually numerical, not architectural.
- **Anchor phrase:** *"Catastrophic cancellation: never subtract two near-equal numbers."*

### 14 — Norms & Distances

- **What it is:** Different norms answer different "close-to-what" questions. Distance is a modeling choice.
- **Amazon:** Visual search A/B-tested cosine vs L2 on normalized vectors. L2 won — magnitude in the embedding encoded popularity, and customers liked popular look-alikes.
- **AA:** "Similar flights" with L1 on (price_$, duration_min, stops) treats $50 the same as 50 minutes — nonsense. Mahalanobis with covariance from booking history respects how customers actually trade off features.
- **Why it sticks:** State your distance choice explicitly. It's a model decision.
- **Anchor phrase:** *"Cosine throws away magnitude. Sometimes that's right. Sometimes it's a bug."*

### 15 — Statistics for ML

- **What it is:** Tools to tell signal from noise: CIs, hypothesis tests, multiple-testing correction, hierarchical pooling.
- **Amazon:** ~10,000 experiments running at once. No multiple-testing correction → ~500 false "wins" per day shipped. Internal platform enforces sequential testing + FDR by default.
- **AA:** Did the new simulator scenario reduce go-arounds? With 14,000 pilots and rare events, naive per-pilot stats drown in noise. Hierarchical models pool across pilots while preserving individual variance.
- **Why it sticks:** Confidence intervals beat p-values for decision-making.
- **Anchor phrase:** *"More tests → more false wins. Always correct."*

### 16 — Sampling Methods

- **What it is:** When you can't compute an expectation exactly, sample. Importance sampling for "wrong distribution," MCMC for "unnormalized," rejection for "bounded ratio."
- **Amazon:** Counterfactual ranking evaluation — "if we had shown B instead of A?" Inverse propensity scoring on logged data answers it without running an A/B test. Filters most candidates offline.
- **AA:** Hurricane stress test — "if a Cat-3 hits MIA?" Monte Carlo over weather × crew × mechanical distributions tells planning the 95th-percentile delay before hurricane season.
- **Why it sticks:** Sampling converts "what if?" into a number with an interval.
- **Anchor phrase:** *"Replay history with weights instead of running the experiment."*

### 17 — Linear Systems

- **What it is:** Solving Ax = b is everywhere. Direct (LU/QR) for small dense; iterative (CG/GMRES) for big sparse. Always check the condition number.
- **Amazon:** Last-mile ETA — Manhattan (sparse) wants CG; rural Texas (smaller, denser) wants direct LU. Switching by geography keeps ETAs accurate and the pipeline cheap.
- **AA:** Every takeoff solves a weight-and-balance system. Irregular cargo can make the matrix near-singular. Dispatch software refuses takeoff if the condition number is too high. Without that check, "legal" takeoffs become unstable in edge cases.
- **Why it sticks:** Don't trust the answer without checking conditioning.
- **Anchor phrase:** *"Condition number first, solve second."*

### 18 — Convex Optimization

- **What it is:** Convex problems have one minimum, and you can prove it. Spotting (or building) convex structure unlocks certifiable answers.
- **Amazon:** Inventory placement across ~175 fulfillment centers is integer in reality, but the LP relaxation is convex. Solve, round, bound the gap → "within 2% of optimal." Leadership can defend the placement.
- **AA:** Treasury hedges jet fuel with futures. Allocating across instruments to minimize variance under coverage constraints is a convex QP. `cvxpy` solves in milliseconds with a proof of optimality.
- **Why it sticks:** Convex solvers give certificates, not just answers.
- **Anchor phrase:** *"Relax → solve → round → bound the gap."*

---

## Cross-cutting patterns (the meta-lessons)

Once you've read all 18, four themes repeat:

1. **The bug isn't where the symptom is.**
   - Search bug → vector geometry (01). Model not learning → numerical underflow (13). Phantom collisions → missing matrix inverse (03). Weird gates → autograd graph cut by `.detach()` (05).

2. **Choose the right tool for the *shape* of the problem.**
   - Distribution shape (06), objective shape (08), distance shape (14), matrix shape (17), problem-convexity shape (18).

3. **Compression is the path to insight, not just to fewer bytes.**
   - PCA reveals clusters (10), SVD reveals low-rank subspaces (11), tensor reshape reveals communication cost (12).

4. **Without certificates you're guessing.**
   - Convex optimality bound (18), `gradcheck` for custom backwards (05), condition number before solving (17), FDR before shipping (15), Bayesian posterior before grounding (07).

---

## How to drill this

- **30-second drill:** look at the Hook column, name the company and the bug.
- **3-minute drill:** for one row, recreate the Amazon story, the AA story, and the anchor phrase from memory. Check against the card.
- **Spaced repetition:** treat each row as a flashcard. Front = topic. Back = the two stories + the anchor.

## Where the full stories live

Each phase folder has the full version at `docs/real-world-stories.md`:

- [01 — Linear algebra intuition](01-linear-algebra-intuition/docs/real-world-stories.md)
- [02 — Vectors & matrices ops](02-vectors-matrices-operations/docs/real-world-stories.md)
- [03 — Matrix transformations](03-matrix-transformations/docs/real-world-stories.md)
- [04 — Calculus for ML](04-calculus-for-ml/docs/real-world-stories.md)
- [05 — Chain rule & autodiff](05-chain-rule-and-autodiff/docs/real-world-stories.md)
- [06 — Probability & distributions](06-probability-and-distributions/docs/real-world-stories.md)
- [07 — Bayes' theorem](07-bayes-theorem/docs/real-world-stories.md)
- [08 — Optimization](08-optimization/docs/real-world-stories.md)
- [09 — Information theory](09-information-theory/docs/real-world-stories.md)
- [10 — Dimensionality reduction](10-dimensionality-reduction/docs/real-world-stories.md)
- [11 — Singular value decomposition](11-singular-value-decomposition/docs/real-world-stories.md)
- [12 — Tensor operations](12-tensor-operations/docs/real-world-stories.md)
- [13 — Numerical stability](13-numerical-stability/docs/real-world-stories.md)
- [14 — Norms & distances](14-norms-and-distances/docs/real-world-stories.md)
- [15 — Statistics for ML](15-statistics-for-ml/docs/real-world-stories.md)
- [16 — Sampling methods](16-sampling-methods/docs/real-world-stories.md)
- [17 — Linear systems](17-linear-systems/docs/real-world-stories.md)
- [18 — Convex optimization](18-convex-optimization/docs/real-world-stories.md)
