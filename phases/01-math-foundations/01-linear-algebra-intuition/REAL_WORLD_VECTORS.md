# Linear Algebra in Real-World AI Projects

## Where These Concepts Are Used

| Concept | Real-World Project | How It's Used |
|---|---|---|
| **Dot product** | OpenAI `text-embedding-ada-002` | Score similarity between a query and stored document chunks in RAG pipelines |
| **Cosine similarity** | Spotify Discover Weekly | Compare user taste vectors to find similar listeners and recommend songs |
| **Vector addition** | Word2Vec / GloVe | `king - man + woman ≈ queen` — arithmetic on word meaning |
| **Magnitude / normalize** | FAISS (Meta) | Normalize embeddings before indexing so cosine search becomes fast dot product search |
| **Projection** | Scikit-learn PCA | Projects high-dim data (e.g. 512-dim face embeddings) onto top principal components for compression |
| **Gram-Schmidt** | PyTorch `torch.linalg.qr` | Stabilizes weight matrices during training; used in spectral normalization for GANs |
| **Linear independence** | Feature selection pipelines | Detect and drop redundant/collinear features before training (e.g. in pandas + sklearn) |
| **Matrix × vector** | Every `nn.Linear` in PyTorch/TF | GPT's attention projections (Q, K, V matrices) are all matrix-vector multiplications |
| **Angle between vectors** | Google Search semantic ranking | Measures angular distance between query embedding and document embedding |
| **Scalar multiplication** | Gradient descent | `weights = weights - lr * gradient` — the `lr *` is scalar multiplication on a weight vector |
| **Transpose** | Transformer self-attention | `QKᵀ` — transpose of K multiplied with Q to compute attention scores |
| **Rank** | Stable Diffusion (LoRA fine-tuning) | Low-rank decomposition (`rank=4`) reduces trainable parameters from millions to thousands |

---

## Terminology & Formulas Explained

### 1. Vector
A vector is an ordered list of numbers representing a point or direction in space.

$$\vec{a} = [a_1, a_2, \ldots, a_n]$$

In AI, a vector is how data is represented — a word, sentence, image, or user preference is encoded as a list of floats called an **embedding**.

---

### 2. Dot Product
Measures how much two vectors point in the same direction. The result is a single number (scalar).

$$\vec{a} \cdot \vec{b} = \sum_{i=1}^{n} a_i b_i$$

**Example:** `[1,2,3] · [4,5,6] = 1×4 + 2×5 + 3×6 = 32`

- Positive → vectors lean the same way
- Zero → vectors are perpendicular
- Negative → vectors lean opposite ways

Used in: attention score computation, retrieval ranking.

---

### 3. Magnitude (Norm)
The length of a vector — how "far" it stretches from the origin.

$$|\vec{a}| = \sqrt{\sum_{i=1}^{n} a_i^2}$$

**Example:** `|[3,4]| = √(9+16) = 5`

Used in: normalizing embeddings before cosine search, gradient clipping.

---

### 4. Normalization
Scales a vector to length 1 (a unit vector) without changing its direction.

$$\hat{a} = \frac{\vec{a}}{|\vec{a}|}$$

Why it matters: normalized embeddings make cosine similarity equivalent to a dot product, which is much faster to compute in large-scale search (FAISS, Pinecone).

---

### 5. Cosine Similarity
Measures the angle between two vectors, ignoring their lengths. Returns a value in `[-1, 1]`.

$$\cos(\theta) = \frac{\vec{a} \cdot \vec{b}}{|\vec{a}|\ |\vec{b}|}$$

- `1.0` → identical direction (same meaning)
- `0.0` → completely unrelated
- `-1.0` → opposite meaning

Used in: semantic search, recommendation systems, RAG pipelines, plagiarism detection.

---

### 6. Angle Between Vectors
The actual geometric angle derived from cosine similarity.

$$\theta = \arccos\!\left(\frac{\vec{a} \cdot \vec{b}}{|\vec{a}|\ |\vec{b}|}\right)$$

**Example:** Perpendicular vectors → `θ = 90°`, parallel vectors → `θ = 0°`

Used in: evaluating how semantically different two embeddings are.

---

### 7. Vector Addition & Subtraction
Component-wise addition/subtraction — moves an arrow in space.

$$\vec{a} + \vec{b} = [a_1+b_1,\ a_2+b_2,\ \ldots]$$

Famous AI example (Word2Vec):

$$\text{king} - \text{man} + \text{woman} \approx \text{queen}$$

This works because word embeddings encode semantic relationships as directions in vector space.

---

### 8. Scalar Multiplication
Scales all components of a vector by a constant.

$$c \cdot \vec{a} = [c \cdot a_1,\ c \cdot a_2,\ \ldots]$$

Used in gradient descent at every training step:

$$\theta \leftarrow \theta - \alpha \cdot \nabla_\theta \mathcal{L}$$

where `α` (learning rate) is the scalar and `∇L` is the gradient vector.

---

### 9. Projection
The "shadow" of vector `a` cast onto vector `b` — how much of `a` lies along the direction of `b`.

$$\text{proj}_{\vec{b}}(\vec{a}) = \frac{\vec{a} \cdot \vec{b}}{\vec{b} \cdot \vec{b}} \cdot \vec{b}$$

The residual `a - proj` is always perpendicular to `b`. This is the geometric foundation of:
- **Least squares regression** (minimizing the residual)
- **PCA** (projecting onto principal directions)
- **Gram-Schmidt** (removing overlap between basis vectors)

---

### 10. Linear Independence
A set of vectors is linearly independent if no vector can be written as a combination of the others.

$$c_1\vec{v_1} + c_2\vec{v_2} + \ldots + c_n\vec{v_n} = \vec{0} \implies \text{all } c_i = 0$$

Why it matters: redundant features in a dataset are linearly dependent. Detecting and removing them improves model training (avoids multicollinearity in linear/logistic regression).

---

### 11. Gram-Schmidt Orthogonalization
A process that takes any set of vectors and produces an equivalent set that is:
- **Orthogonal** — every pair is perpendicular (dot product = 0)
- **Normal** — every vector has length 1

**Algorithm (simplified):**
1. Take vector `v₁`, normalize it → `u₁`
2. Take `v₂`, subtract its projection onto `u₁` → normalize → `u₂`
3. Repeat for each subsequent vector

Used in:
- **QR decomposition** (`torch.linalg.qr`) for solving linear systems
- **Spectral normalization** in GANs to stabilize discriminator training

---

### 12. Matrix × Vector (Linear Transformation)
Applying a matrix to a vector transforms (rotates, scales, projects) it into a new space.

$$\vec{y} = W\vec{x}, \quad y_i = \sum_j W_{ij} x_j$$

This is the core operation in every neural network layer:

```
Input (3D) → Linear layer W (2×3 matrix) → Output (2D)
```

In transformers, the Query, Key, and Value projections are all matrix-vector multiplications:

$$Q = W_Q x, \quad K = W_K x, \quad V = W_V x$$

---

### 13. Transpose
Flips a matrix over its diagonal — rows become columns.

$$A^T_{ij} = A_{ji}$$

Critical in transformer self-attention:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

`QKᵀ` produces a score matrix where entry `(i,j)` is how much token `i` attends to token `j`.

---

### 14. Matrix Rank
The number of linearly independent rows (or columns) in a matrix — a measure of how much "useful information" it encodes.

- A full-rank `n×n` matrix → invertible, no information loss
- A rank-deficient matrix → redundant rows/columns, information is compressed

Used in **LoRA (Low-Rank Adaptation)** for fine-tuning LLMs:

$$W' = W + \Delta W = W + BA, \quad B \in \mathbb{R}^{d \times r},\ A \in \mathbb{R}^{r \times k},\ r \ll d$$

Instead of updating all `d×k` parameters, only `r×(d+k)` parameters are trained where `r` (rank) is small (e.g. 4 or 8). This is how Stable Diffusion and LLaMA are fine-tuned cheaply.

---

## What Goes Wrong When You Get These Wrong

### 1. Skipping Normalization before Cosine Similarity
**What you think you're measuring:** semantic closeness between embeddings  
**What you're actually measuring:** a mix of direction AND vector length

A document repeated 10 times has a 10× longer embedding vector. Without normalization it scores higher than a genuinely more relevant document — just because it's longer. This causes **retrieval bias** in RAG pipelines and semantic search.

**Real failure:** Early vector databases that skipped L2-normalization returned longer documents as "more similar" regardless of content.

---

### 2. Using Dot Product Instead of Cosine Similarity on Un-normalized Embeddings
**Symptom:** Your similarity scores are dominated by frequency, not meaning.  
**Example:** In a product search engine, popular items with high-frequency embeddings crowd out niche but relevant results.

**Fix:** Either normalize first, or explicitly use cosine similarity. Most embedding APIs (OpenAI, Cohere) return pre-normalized vectors for this reason.

---

### 3. Ignoring Linear Dependence (Collinear Features)
**What happens:** Your model trains fine but generalizes poorly.

If you feed a linear/logistic regression model two features that are perfectly correlated (e.g. "price in USD" and "price in EUR"), the weight matrix is **rank-deficient** — infinitely many weight combinations produce the same loss. Gradient descent wanders without converging cleanly.

- Coefficients become unstable and flip signs between runs
- Confidence intervals blow up
- The model appears to overfit

**Real failure:** Clinical ML models trained on correlated lab values (e.g. BUN and creatinine both measuring kidney function) notoriously produce unreliable coefficients.

---

### 4. Not Understanding Projection (in PCA / Regression)
**What you miss:** why information is lost when you reduce dimensions.

If you project 512-dim face embeddings down to 2 dimensions without understanding projection, you don't know:
- Which components you dropped (and whether they encoded identity vs. lighting)
- Why two faces that looked different in 512D now appear identical in 2D

**Real failure:** Teams that blindly apply PCA to reduce embedding dimensions for speed end up degrading retrieval quality — the dropped dimensions often encoded the most discriminative features.

---

### 5. Getting the Dot Product Sign Wrong in Attention
The transformer attention score is:

$$\text{score}(q, k) = q \cdot k^T$$

If you accidentally compute $k \cdot q^T$ with wrong broadcasting shapes, attention weights become **transposed** — every token attends to the wrong other tokens. The model still trains (loss decreases) but learns a corrupted inductive bias.

This class of silent shape bug is one of the most common sources of subtle performance degradation in custom transformer implementations.

---

### 6. Forgetting to Scale the Dot Product in Attention
The full formula is:

$$\text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)$$

Without dividing by $\sqrt{d_k}$, dot products grow large as embedding dimension increases. Large values push softmax into **saturation** — gradients vanish and the model stops learning. The original "Attention Is All You Need" paper explicitly notes this was a discovered failure during training.

---

### 7. Wrong Rank in LoRA Fine-tuning
**Too low rank (e.g. rank=1):** The weight update $\Delta W = BA$ has too little capacity — the model can't adapt to the new task. Fine-tuning produces near-zero improvement.

**Too high rank (e.g. rank=256 on a small dataset):** You're essentially fine-tuning the full matrix. The low-rank assumption breaks down, you overfit to the fine-tuning data, and lose the base model's generalization.

**Real failure:** Early LoRA experiments on small domain datasets with high rank resulted in catastrophic forgetting — the model aced the new task but forgot its general language ability.

---

### 8. Not Applying Gram-Schmidt (Unstable Bases)
In gradient descent, if weight matrices develop nearly-linearly-dependent rows over training (a form of rank collapse), gradients become **numerically unstable** — tiny floating-point errors get amplified.

**Symptoms:**
- Loss suddenly spikes to NaN mid-training
- Gradient norms explode without obvious cause
- Training is non-reproducible across hardware

**Fix:** Spectral normalization (based on Gram-Schmidt / QR decomposition) constrains the singular values of weight matrices, keeping training stable. Widely used in GAN discriminators.

---

### 9. Misunderstanding Vector Addition in Embedding Arithmetic
Word2Vec's `king - man + woman ≈ queen` only works because the embedding space was trained such that **gender is encoded as a consistent direction**.

If you apply this reasoning to embeddings from a model that wasn't trained with such structure (e.g. raw BERT contextual embeddings), the arithmetic doesn't hold — you get garbage outputs.

**Real failure:** Teams that tried to do analogy reasoning with BERT embeddings (instead of static Word2Vec/GloVe) found the results were incoherent, because BERT embeddings are context-dependent, not fixed geometric directions.

---

### 10. Using Euclidean Distance Instead of Cosine Similarity for Embeddings
Euclidean distance measures absolute separation in space. Cosine similarity measures angular separation.

For embeddings, two sentences can be far apart in Euclidean space but express the same idea — just encoded at different magnitudes. Using L2 distance without normalization causes a semantic search system to:
- Miss paraphrases that are geometrically distant
- Return near-duplicates that happen to be close in raw coordinates

**Real failure:** Early Elasticsearch dense vector search used L2 distance by default. Teams that didn't switch to cosine similarity reported their semantic search was "barely better than keyword search."

---

### Summary: The Cost of Getting It Wrong

| Mistake | Consequence |
|---|---|
| No normalization before cosine similarity | Retrieval biased toward long documents, not relevant ones |
| Collinear features in regression | Unstable weights, poor generalization, flipped coefficients |
| Blind PCA without understanding projection | Silent quality degradation in downstream tasks |
| Wrong attention dot product shape | Model trains but learns corrupted attention patterns |
| Missing $1/\sqrt{d_k}$ scaling | Vanishing gradients, model stops learning |
| Wrong LoRA rank | Either no adaptation or catastrophic forgetting |
| Skipping spectral normalization | NaN loss, exploding gradients, non-reproducible runs |
| Embedding arithmetic on wrong model type | Incoherent analogy results |
| L2 distance instead of cosine on embeddings | Semantic search barely better than keyword matching |
