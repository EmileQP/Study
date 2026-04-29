# AIML 231 – Techniques in Machine Learning: Study Guide

> **Exam context (2026 Trimester 1 Mid-term — source: `AIML231_Midterm_Test_Frontpage.pdf`)**  
> Date: 30 April 2026 | Time: 50 minutes | Closed book | 50 marks total  
> Permitted materials: silent non-programmable calculator; non-electronic foreign language dictionary

| Question | Topic | Marks |
|---|---|---|
| 1 | Machine Learning Basics | 7 |
| 2 | Classification | 18 |
| 3 | ML Pipeline and Data Pre-processing | 15 |
| 4 | Regression | 5 |
| 5 | Clustering | 5 |

**Style reference (`2024_1_AIML231.pdf`):** The 2024 final exam (100 marks, 2 hours) covered the same five topics with the same relative weightings, plus Neural Networks, Search/Evolutionary Computation, and Advanced Topics. Question types include short-answer definitions, worked numerical calculations, and multi-part structured responses. Marks are printed per sub-part.

---

## Topic 1 – Machine Learning Basics (7 marks)

*Source files: `AIML231_Week1_ML_Overview.pdf`, `AIML231_Week1_ML_Tasks.pdf`, `AIML231_Week2_Classification.pdf`*

### 1.0 Week 1 ML Overview: AI, Data, and the ML Workflow

*(Source: `AIML231_Week1_ML_Overview.pdf`)*

#### What is Artificial Intelligence vs Machine Learning?

- **Artificial Intelligence (AI):** A broad field of computer science focused on creating systems capable of carrying out tasks in a way we would consider "smart."
- **Machine Learning (ML):** A current application of AI based around the idea of giving machines access to data and letting them learn for themselves — without being explicitly programmed.

#### The Four Elements of Every ML Problem

| Element | Description | Example |
|---|---|---|
| **Data** | The raw examples used for learning | Banknote measurements CSV |
| **Task** | What we want the system to do | Classify note as genuine or forged |
| **Model** | The learned function used to make predictions | Trained decision tree |
| **Algorithm** | The procedure that produces the model from data | ID3, gradient descent |

#### Data Representation

- Datasets are typically structured as a **table of instances** (rows) and **features** (columns).
- Each row is one **instance** (or example, observation, data point).
- **Features (X):** The input variables — the independent variables used for prediction. Also called attributes or predictors. Represented as a **vector**: X = (x₀, x₁, x₂, …, xd).
- **Labels (y):** The output variable (the thing we want to predict). In supervised learning each instance has a label; in unsupervised learning there are no labels.
- Data can be viewed as **points in a d-dimensional vector space**, where d = number of features.

#### Training, Validation, and Testing

A supervised ML workflow always splits data into (at least) three non-overlapping subsets:

| Subset | Role |
|---|---|
| **Training set** | Used to fit (train) the model — the algorithm sees these examples. |
| **Validation set** | Used during development to tune hyperparameters and compare models. The model does NOT train on this. |
| **Test set** | Held out until the very end; provides an unbiased estimate of real-world performance. Never touched during training or tuning. |

> **Exam tip:** Never fit any preprocessing transformer (scaler, encoder) on the validation or test set — fit on train, transform all sets. Doing otherwise causes **data leakage**.

#### General ML Workflow

1. **Define the problem** — what task, what data, what success metric?
2. **Collect and explore data** — understand distributions, spot missing values, outliers.
3. **Preprocess data** — clean, encode, scale, engineer features; *split into train/val/test first*.
4. **Select and train a model** — choose algorithm family; fit on training data.
5. **Evaluate and tune** — measure performance on validation set; adjust hyperparameters.
6. **Final evaluation** — assess on held-out test set to estimate real-world performance.
7. **Deploy** — integrate model into production; monitor for drift.

#### Features and Labels in Practice

- **Numerical features** (continuous or discrete numbers) — e.g., height, price, sensor readings.
- **Categorical features** (finite set of categories) — e.g., colour, postcode, true/false.
- **Binary label** (two classes) → binary classification.
- **Multi-class label** (>2 classes) → multi-class classification.
- **Continuous label** (real number) → regression.
- **No label** → unsupervised (clustering, dimensionality reduction).

#### Overfitting, Underfitting, and Generalisation

| Concept | Cause | Symptom |
|---|---|---|
| **Underfitting** | Model too simple; high bias | Poor performance on *both* train and test sets |
| **Good fit** | Model complexity matches data | Good performance on train and test sets |
| **Overfitting** | Model too complex; high variance; memorises noise | High train accuracy, *much lower* test accuracy |

- **Generalisation:** A model's ability to perform well on *unseen* data.
- The goal of ML is good generalisation, not simply low training error.

#### Python Ecosystem for ML

Common tools introduced in Week 1:
- **NumPy** — array/matrix operations
- **Pandas** — data frames and CSV loading
- **Matplotlib** — visualisation
- **Scikit-learn (sklearn)** — ML algorithms, pipelines, preprocessing, metrics
- **Jupyter Notebooks** — interactive development environment

---

### 1.1 Key Concepts and Definitions

**Machine Learning (ML)** is a sub-field of AI where a system *learns* from data rather than following hand-coded rules. Every ML problem involves four elements: **data**, **task**, **model**, and **algorithm**.

| Term | Definition |
|---|---|
| **ML model** | The final output of a training process — can make predictions on new inputs (e.g., a trained decision tree). |
| **ML algorithm** | The procedure used to train a model from data (e.g., gradient descent, ID3). |

**Three learning paradigms** (source: `AIML231_Week1_ML_Tasks.pdf` p. 8):

| Paradigm | Supervision | Description |
|---|---|---|
| **Supervised learning** | Full (labelled data) | Environment provides correct answer for each input; goal is "do what the teacher does." |
| **Unsupervised learning** | None | No teacher; system has internally defined goals (e.g., find structure, compress). |
| **Reinforcement learning** | Reward/punishment | No per-step label; environment gives reward/punishment to guide actions. |

**Main task taxonomy** (`AIML231_Week1_ML_Tasks.pdf` p. 9):
- Supervised → *Classification* (discrete output), *Regression* (continuous output)
- Unsupervised → *Clustering*, *Dimensionality Reduction*
- Semi-supervised, Self-supervised also exist

### 1.2 Core Concepts

**Curse of Dimensionality** (`AIML231_Week1_ML_Tasks.pdf` p. 17; also `AIML231_Week2_Classification.pdf` p. 24):  
As the number of features (dimensions) grows, data becomes increasingly sparse. A fixed training set covers an exponentially smaller fraction of the feature space. This degrades distance-based algorithms (like KNN) and makes learning harder. Also stated in the 2024 exam Q1(b).

**Bias–Variance Trade-off** (`AIML231_Week2_Classification.pdf` p. 24):

> Expected generalisation error = **Bias** + **Variance** + Irreducible error

| Term | Meaning | Symptom |
|---|---|---|
| **Bias** | Error from wrong assumptions (model too simple) | Underfitting — bad on *both* train & test |
| **Variance** | Error from sensitivity to small data changes (model too complex) | Overfitting — great on train, bad on test |

**Overfitting vs. Underfitting** (`AIML231_Week2_Classification.pdf` p. 23):
- *Overfitting*: model memorises training noise; high train accuracy, poor test accuracy.
- *Underfitting*: model too simple; poor on both train and test.
- *Generalisation*: model's ability to perform well on unseen data.

### 1.3 Three Key Differences: Supervised vs. Reinforcement Learning (from 2024 Q1c)

| | Supervised Learning | Reinforcement Learning |
|---|---|---|
| Feedback | Immediate, per-sample label | Delayed reward/punishment signal |
| Dataset | Fixed labelled dataset | Interaction with environment; no fixed dataset |
| Goal | Match teacher's output | Maximise cumulative reward over time |

### 1.4 Typical Pitfalls

- Confusing *model* and *algorithm* — be precise.
- Saying "clustering" when you mean "classification" (clustering is *unsupervised*).
- Thinking reinforcement learning "has a teacher" — it has rewards, not labels.

### 1.5 Example Questions

**Q:** Briefly explain unsupervised machine learning and give two examples. (3 marks, from 2024 Q1a)  
**A:** Unsupervised ML learns patterns from data without labelled outputs. No teacher provides correct answers. Two examples: (1) *Clustering* — grouping customers by purchase behaviour; (2) *Dimensionality reduction (PCA)* — compressing image features.

**Q:** What is the curse of dimensionality? (2 marks, from 2024 Q1b)  
**A:** As the number of features increases, data becomes increasingly sparse in high-dimensional space. A fixed number of training examples covers a shrinking fraction of the feature space, making it harder for models to generalise (distance-based methods suffer most).

---

## Topic 2 – Classification (18 marks)

*Source files: `AIML231_Week2_Classification.pdf`, `AIML231_Week3_DT.pdf`, `AIML231_Week3_PerformanceMetrics.pdf`*

### 2.1 Classification vs. Regression

| | Classification | Regression |
|---|---|---|
| Output type | Discrete/categorical (class label) | Continuous numerical value |
| Example | Email → spam / not spam | House → price in $ |
| Metrics | Accuracy, F1, AUC | MSE, RMSE, R² |

Source: `AIML231_Week6_Regression (1).pdf` p. 2; 2024 exam Q2(a).

### 2.2 Training / Validation / Test Sets

(`AIML231_Week2_Classification.pdf` pp. 8, 25; 2024 Q2b)

| Set | Role |
|---|---|
| **Training set** | Used to train (fit) the model — model *sees* these labels. |
| **Validation set** | Used to tune hyperparameters (e.g., choose K in KNN); model does NOT see test labels. |
| **Test set** | Held out completely; used only at the very end to estimate real-world performance. |

> **Critical rule:** Always split *before* preprocessing to avoid **data leakage** (see Topic 3).

### 2.3 K-Fold Cross Validation

(`AIML231_Week2_Classification.pdf` pp. 26–30; 2024 Q2c)

**Key idea:** Partition dataset into K equal folds. In each of K experiments, use K−1 folds for training and the remaining fold for testing. Average the K error estimates → more reliable than a single train/test split.

**When to use:**
- When the dataset is small and a single holdout may give misleading results.
- When comparing models or tuning hyperparameters.
- Common choice: K = 10.

**Variants:**
- *Random subsampling* – K random splits; some instances may appear in both train and test.
- *Leave-one-out (LOO)* – K = n; used for very small datasets.

### 2.4 K-Nearest Neighbour (KNN) Classifier

(`AIML231_Week2_Classification.pdf` pp. 14–22)

**Algorithm:**
1. Choose K.
2. Compute distance (Euclidean or Manhattan) from query point to all training points.
3. Find K nearest neighbours.
4. Assign the majority class label among the K neighbours.

**Euclidean distance** between (x₁, x₂, …, xₙ) and (a₁, a₂, …, aₙ):

```
D = sqrt( (x₁-a₁)² + (x₂-a₂)² + … + (xₙ-aₙ)² )
```

**Effect of K** (`AIML231_Week2_Classification.pdf` p. 22):
- **K too small (e.g., K=1):** Overfitting — model is too sensitive to noise/outliers.
- **K too large:** Underfitting — model over-smooths; boundaries become too coarse; ignores local structure; computationally expensive on large datasets.

*2024 Q2(d) asks for two problems with large K: (1) underfitting/over-smoothing; (2) slow prediction (must examine many neighbours).*

**Strengths/Weaknesses:**
- Strengths: simple, no training phase, adapts to non-linear boundaries.
- Weaknesses: slow at prediction for large data, poor with high dimensions (curse of dimensionality), sensitive to irrelevant features.

### 2.5 Decision Trees

(`AIML231_Week3_DT.pdf`)

**Structure:** nodes = decision points (features), branches = outcomes, leaves = class labels.

**Building a decision tree (ID3 algorithm):**
1. Compute entropy of the full dataset.
2. For each feature, compute information gain.
3. Split on the feature with highest information gain.
4. Recurse on each subset.
5. Stop when a node is pure or stopping condition is met.

**Entropy** (uncertainty measure):

```
H(Y) = −Σ P(Y=yᵢ) × log₂(P(Y=yᵢ))
```

- H = 0 → perfectly pure node (one class).
- H = 1 → maximum uncertainty for binary classes (equal split).

**Conditional Entropy:**

```
H(Y|X) = −Σⱼ P(X=xⱼ) × Σᵢ P(Y=yᵢ|X=xⱼ) × log₂(P(Y=yᵢ|X=xⱼ))
```

**Information Gain:**

```
IG(Y, X) = H(Y) − H(Y|X)
```

Higher IG → better split (feature reduces uncertainty more).

**Overfitting in decision trees** (`AIML231_Week3_DT.pdf` p. 14–15):
- Full trees memorise training data.
- Solutions: (1) set max tree depth; (2) set minimum partition size; (3) **pre-pruning** (stop early when gain is too small); (4) **post-pruning** (grow full tree then remove branches).

**Handling real-valued features:** Binary threshold split; choose threshold that maximises IG (only consider class boundaries, not every possible value).

### 2.6 Random Forests

(`AIML231_Week3_DT.pdf` p. 20)

- Ensemble of decision trees, each trained on a **bootstrap sample** (sampled with replacement).
- Each split considers only a **random subset** of features.
- Final prediction: **majority vote** over all trees.
- Effect: reduces overfitting, improves accuracy relative to single tree.

### 2.7 Performance Metrics

(`AIML231_Week3_PerformanceMetrics.pdf`)

**Confusion Matrix (binary):**

|  | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actual Positive** | TP (True Positive) | FN (False Negative) |
| **Actual Negative** | FP (False Positive) | TN (True Negative) |

**Core metrics:**

| Metric | Formula | Intuition |
|---|---|---|
| Accuracy | (TP+TN)/(TP+TN+FP+FN) | Overall correct fraction |
| Error Rate | 1 − Accuracy | Overall wrong fraction |
| TPR / Sensitivity / Recall | TP/(TP+FN) | Of actual positives, how many detected? |
| TNR / Specificity | TN/(TN+FP) | Of actual negatives, how many correctly rejected? |
| FPR | FP/(FP+TN) = 1 − TNR | Of actual negatives, how many falsely flagged? |
| FNR | FN/(FN+TP) = 1 − TPR | Of actual positives, how many missed? |
| Precision | TP/(TP+FP) | Of predicted positives, how many correct? |
| F1-Score | 2×Precision×Recall / (Precision+Recall) | Harmonic mean; balances precision & recall |

**Key identities:** FNR + TPR = 1; FPR + TNR = 1.

**ROC Curve:** Plot TPR (y-axis) vs FPR (x-axis) across all decision thresholds. Ideal point = (0, 1).

**AUC (Area Under ROC Curve):** Single number summary.
- AUC = 1.0 → perfect classifier.
- AUC = 0.5 → random guess (worst case).
- 0.5 < AUC < 1.0 → typical range.

### 2.8 Worked Example: Information Gain Calculation

(From 2024 exam Q2f — restaurant menu dataset)

**Dataset:** 10 items, 5 Popular, 5 Unpopular. Spice feature: 7 spicy, 3 mild.

**Step 1 — H(Y):**  
P(Popular) = 5/10 = 0.5, P(Unpopular) = 5/10 = 0.5  
H(Y) = −(0.5 log₂ 0.5 + 0.5 log₂ 0.5) = −(0.5 × −1 + 0.5 × −1) = **1.0 bit**

**Step 2 — H(Y|Spice):**  
- Spicy (7 items): 4 Popular, 3 Unpopular → H = −(4/7 log₂(4/7) + 3/7 log₂(3/7)) ≈ 0.985 bits  
- Mild (3 items): 1 Popular, 2 Unpopular → H = −(1/3 log₂(1/3) + 2/3 log₂(2/3)) ≈ 0.918 bits  
H(Y|Spice) = 7/10 × 0.985 + 3/10 × 0.918 ≈ 0.690 + 0.276 = **0.965 bits**

**Step 3 — IG:**  
IG = 1.0 − 0.965 = **0.035 bits**

### 2.9 Worked Example: Confusion Matrix Calculations

(From 2024 exam Q2e — medical test)

Data: 200 patients (40 with disease, 160 without). Test correctly identifies 28 positives (TP=28); falsely flags 20 negatives (FP=20).  
FN = 40 − 28 = 12; TN = 160 − 20 = 140.

- **FNR** = FN/(TP+FN) = 12/(28+12) = 12/40 = **0.30 (30%)**
- **TNR** = TN/(FP+TN) = 140/(20+140) = 140/160 = **0.875 (87.5%)**

### 2.10 Typical Pitfalls

- Forgetting FNR + TPR = 1 and FPR + TNR = 1.
- Using accuracy alone for imbalanced datasets — use F1 or AUC.
- Not showing working when asked for a calculation — show every step.
- Confusing precision (predicted positives that are correct) with recall/TPR (actual positives detected).
- Large K in KNN → underfitting, NOT overfitting.
- Pre-pruning vs. post-pruning: pre = stop early; post = prune after full growth.

---

## Topic 3 – ML Pipeline and Data Pre-processing (15 marks)

*Source files: `Week4.1_Pipelines.pdf`, `Week4.2_EDA.pdf`, `AIML231_Week5_DataPreprocessing (1).pdf`, `AIML231_Week5_FeatureSelection.pdf`, `AIML231_Week6_FeatureConstruction (1).pdf`*

### 3.1 CRISP-DM

(`Week4.1_Pipelines.pdf` pp. 12–18; 2024 Q3a)

**CRoss Industry Standard Process for Data Mining** — six phases (cyclical, not linear):

| Phase | Key activities |
|---|---|
| 1. **Business Understanding** | Define objectives, success criteria, project plan |
| 2. **Data Understanding** | Collect data, describe it, explore it, check quality |
| 3. **Data Preparation** | Select, clean, construct, integrate, format data (~80% of effort) |
| 4. **Modelling** | Select model type, design test, build and tune model |
| 5. **Evaluation** | Assess against business criteria, review process, decide next steps |
| 6. **Deployment** | Deploy model, plan monitoring/maintenance, produce final report |

### 3.2 Data Leakage

(`AIML231_Week5_DataPreprocessing (1).pdf` p. 5; 2024 Q3c)

**Definition:** When information from the test set "leaks" into training (e.g., fitting a scaler on the full dataset before splitting).

**Correct order:** Split first, then preprocess (fit scaler/encoder on train only, transform test).

**Incorrect (Nick's approach in 2024 Q3c):** Normalise the *whole* dataset → then split → data leakage. The test set statistics contaminate the training normalisation, giving an overly optimistic estimate of performance.

### 3.3 Exploratory Data Analysis (EDA)

(`Week4.2_EDA.pdf`)

**Purpose:** Understand data structure and quality, discover patterns, identify outliers, check assumptions before modelling.

**EDA step-by-step:**
1. Inspect data — variable types, number of samples, formats.
2. Handle missing values.
3. Explore data characteristics (univariate and multivariate).
4. Perform data transformations.

**Missing data types:**
| Type | Description | Example |
|---|---|---|
| MCAR | Missingness unrelated to data | Survey lost in mail |
| MAR | Depends on observed variables | Young people skip income question |
| MNAR | Depends on missing value itself | Very low income people don't report income |

**Handling missing values:**
- MCAR: sample deletion or variable deletion (only if < ~5% missing).
- MAR/MNAR: mean/mode imputation, or KNN imputation (considers feature relationships).

### 3.4 Data Preprocessing

(`AIML231_Week5_DataPreprocessing (1).pdf`)

#### Encoding Categorical Features

| Method | When | How |
|---|---|---|
| **One-hot encoding** | Nominal features (no ordering) | Each category → binary column; k categories → k columns |
| **Ordinal encoding** | Ordinal features (ordering exists) | Map categories to integers preserving order (e.g., XS=0, S=1, M=2, L=3, XL=4) |

#### Scaling Numerical Features

(`AIML231_Week5_DataPreprocessing (1).pdf` pp. 12–15; `Week4.2_EDA.pdf` p. 32)

Must scale because: features with larger ranges dominate distance-based algorithms.

| Method | Formula | When to use |
|---|---|---|
| **Min-Max scaling** | x' = (x − xₘᵢₙ) / (xₘₐₓ − xₘᵢₙ) → [0, 1] | Non-normal features; sensitive to outliers |
| **Z-score standardisation** | z = (x − μ) / σ → mean 0, std 1 | Normally distributed features |

Rule of thumb: if features are a mix of normal and non-normal, use min-max on all.

#### Discretisation

(`AIML231_Week5_DataPreprocessing (1).pdf` pp. 17–20; 2024 Q3b)

Converting a continuous feature into discrete bins.

| Method | How | Pros / Cons |
|---|---|---|
| **Equal-Width (Uniform)** | Divide range [min, max] into N bins of equal width W = (max−min)/N | Preserves scale; uneven distribution with skewed data |
| **Equal-Depth (Quantile)** | Sort values; put same number of samples in each bin | Better for skewed data; bin widths vary |

#### Missing Value Imputation

(`AIML231_Week5_DataPreprocessing (1).pdf` pp. 22–27)

| Method | Description |
|---|---|
| **Sample/Variable deletion** | Remove rows or columns; only when < 5% missing |
| **Mean imputation** | Replace missing numeric values with feature mean |
| **Mode imputation** | Replace missing categorical values with most frequent value |
| **KNN imputation** | Use K nearest neighbours' values; more accurate but slower |

### 3.5 Feature Selection

(`AIML231_Week5_FeatureSelection.pdf`)

**Purpose:** Identify a minimal subset of relevant features to improve performance, reduce overfitting, reduce computation.

**Feature Selection vs. Feature Construction:**
- **Feature Selection (FS):** Choose a subset F' ⊆ F of original features (m < n features).
- **Feature Construction (FC):** Build *new* features from original ones, possibly all of them.

#### Univariate Feature Selection (Feature Ranking)

Evaluates each feature independently using a score function. Top-ranked features are selected.

Score functions:
- `r_regression()` / Pearson correlation — linear relationships for regression.
- `mutual_info_regression()` — linear and non-linear for regression.
- `mutual_info_classif()` — for classification tasks.
- `chi2()` — for categorical features and discrete targets.

#### Multivariate Feature Selection (Feature Subset Selection)

Considers feature interactions. Exponential search space (2ⁿ subsets for n features) → use greedy search.

**Sequential Forward Feature Selection (SFFS)** (`AIML231_Week5_FeatureSelection.pdf` pp. 17–21; 2024 Q3e):
1. Start with empty set {}.
2. At each step: add the feature that, when added to current subset, gives the best performance.
3. Stop when desired number of features is selected.
- Best when optimal subset is small.
- Cannot remove previously added features (*nesting effect*).

**Sequential Backward Feature Selection (SBFS):**
1. Start with full feature set.
2. At each step: remove the feature whose removal gives the best performance.
3. Stop when desired number of features is reached.
- Best when optimal subset is large.

### 3.6 Feature Construction / Dimensionality Reduction

(`AIML231_Week6_FeatureConstruction (1).pdf`; 2024 Q3d–f)

**Three categories of feature construction approaches** (also applies to feature selection):
- **Filter:** Uses statistical properties of data alone (independent of any learning algorithm). E.g., PCA.
- **Wrapper:** Uses a learning algorithm's performance to evaluate feature subsets. E.g., SFFS with KNN.
- **Embedded:** Feature selection built into the model training itself. E.g., Lasso regression.

#### Principal Component Analysis (PCA)

(`AIML231_Week6_FeatureConstruction (1).pdf` pp. 7–15; 2024 Q3f)

**Definition:** Linearly transforms (possibly correlated) features into uncorrelated *principal components* (PCs) that capture maximum variance.

**Algorithm:**
1. Centralise data: subtract mean of each feature (xᵢ ← xᵢ − μᵢ).
2. (Optional) Scale data so all features have similar range.
3. Compute covariance matrix C (d × d).
4. Eigen-decompose C: find eigenvalues λₗ and eigenvectors aₗ.
5. Rank PCs by eigenvalue (higher = more variance captured).
6. Select top p PCs (by cumulative explained variance ≥ 95%, or elbow of scree plot).

**Key facts:**
- Maximum number of PCs = number of original features d.
- PCs are orthogonal (uncorrelated) to each other.
- PCA is an **unsupervised, filter** approach — it does not use class labels.

**Limitations of PCA:**
1. Assumes *linear* relationships; fails on non-linear data (→ use Kernel PCA).
2. Must choose number of components p (not automatic).
3. Interpretability: constructed features are linear combinations and may lack meaning.
4. Uncorrelated features are not always sufficient (non-linear separability).

**Kernel PCA:** Maps data to higher-dimensional space via kernel trick K(xᵢ, xⱼ) = cov(φ(xᵢ), φ(xⱼ)); handles non-linear data.

**Polynomial Features:** Generate polynomial combinations of existing features (e.g., degree-2 of [x₁, x₂] → [1, x₁, x₂, x₁², x₁x₂, x₂²]).

### 3.7 Worked Example: SFFS

(From 2024 exam Q3e — 4 features, select 3)

**Iteration 1 (start with {}):** Evaluate all single features.  
Best: {f₃} with accuracy 0.7 → add f₃.

**Iteration 2 (current = {f₃}):** Evaluate {f₁,f₃}=0.55, {f₂,f₃}=0.6, {f₃,f₄}=0.76.  
Best: {f₃,f₄} = 0.76 → add f₄.

**Iteration 3 (current = {f₃,f₄}):** Evaluate {f₁,f₃,f₄}=0.8, {f₂,f₃,f₄}=0.5.  
Best: {f₁,f₃,f₄} = 0.8 → add f₁.

**Result:** Selected features = {f₁, f₃, f₄}. Algorithm terminates (3 features selected as required).

### 3.8 Typical Pitfalls

- **Data leakage:** Scaling/encoding before splitting is a very common exam pitfall.
- Confusing feature *selection* (subset of original) with feature *construction* (new features built from originals).
- Stating PCA is "supervised" — it is unsupervised (no class labels used).
- Forgetting that with d original features, PCA can produce *at most d* PCs.
- Confusing equal-width and equal-depth discretisation — know which handles skew better.
- In SFFS, at each step you evaluate adding each *remaining* feature to the *current* subset, not all possible subsets.

---

## Topic 4 – Regression (5 marks)

*Source file: `AIML231_Week6_Regression (1).pdf`*

### 4.1 Key Concepts and Definitions

**Regression** is a supervised ML task that predicts a **continuous** numerical output from input features.

**Simple Linear Regression** (1 feature):

```
f(x) = w₀ + w₁ × x
```
- w₀ = intercept, w₁ = slope.

**Multiple Linear Regression** (d features):

```
f(xᵢ) = w₀ + w₁xᵢ₁ + w₂xᵢ₂ + … + wdxᵢd
```
- **d+1 weights total** (d slopes + 1 intercept). ← Key fact for exams.

### 4.2 Core Formulas

**Residual Error** for instance i:  
```
eᵢ = yᵢ − f(xᵢ)
```

**Residual Sum of Squares (RSS)** — minimisation objective:
```
RSS = Σᵢ (yᵢ − f(xᵢ))²
```

Optimisation: **Least Squares Estimation (LSE)** (closed form) or **Gradient Descent** (iterative).

### 4.3 Regularised Regression

(`AIML231_Week6_Regression (1).pdf` pp. 10–12; 2024 Q4b)

| Method | Penalty term | Effect |
|---|---|---|
| **Ridge regression** (L2) | RSS + λ Σ wₖ² | Shrinks weights toward zero; never exactly zero |
| **Lasso regression** (L1) | RSS + λ Σ |wₖ| | Can drive weights to *exactly* zero → embedded feature selection |

**Which is embedded FS?** Lasso — because it zeroes out irrelevant feature weights, effectively removing them.

λ controls regularisation strength: λ=0 → plain linear regression; large λ → heavy penalty.

### 4.4 Regression Performance Metrics

(`AIML231_Week6_Regression (1).pdf` pp. 15–18)

| Metric | Formula | Notes |
|---|---|---|
| **MSE** | (1/N) Σ(yᵢ − ŷᵢ)² | Differentiable; penalises large errors; sensitive to outliers; squared units |
| **MAE** | (1/N) Σ|yᵢ − ŷᵢ| | Less sensitive to outliers; same units as output; not differentiable at 0 |
| **RMSE** | √MSE | Same units as output; sensitive to outliers |
| **R²** | 1 − RSS / Σ(yᵢ − ȳ)² | ≤ 1; higher=better; R²=1 perfect, R²=0 no better than mean predictor, R²<0 very poor |

**Recommendation:** Use R² for comparing models across different scales.

### 4.5 Number of Weights (Exam Favourite)

*2024 Q4a:* "Training set has 10 features, how many weights?"  
Answer: **11** — one weight per feature (w₁ to w₁₀) plus one intercept (w₀).

### 4.6 Typical Pitfalls

- Forgetting the intercept weight (w₀) — always d+1 weights for d features.
- Confusing Ridge (shrinks toward zero but not to zero) with Lasso (can reach zero → feature selection).
- Saying MSE and MAE have the same sensitivity to outliers — MSE is *more* sensitive due to squaring.

---

## Topic 5 – Clustering (5 marks)

*Source file: `AIML231_Week7-Clustering (1).pdf`*

### 5.1 Key Concepts and Definitions

**Clustering** is an *unsupervised* task: group objects so that within-group similarity is high and between-group similarity is low. Class labels are **unknown** — clustering discovers hidden structure in data.

| | Clustering | Classification |
|---|---|---|
| Labels | Unknown (unsupervised) | Known (supervised) |
| Training data | Not required | Required |
| Number of classes | Usually not given in advance | Known |
| Aim | Find natural groups in data | Classify future instances |

#### Real-World Motivation: Customer Segmentation

**Customer segmentation** divides customers into groups sharing common characteristics:
- Historical purchases, geographical location, product/service preferences, socio-economic factors (income, education).
- Each group can be targeted with tailored marketing strategies.
- Leads to better customer support, more effective communication, and increased revenue.

**Netflix example:** Netflix clusters users and content together. Based on a user's viewing history and what similar users tend to watch, the system quickly recommends relevant content — a direct application of clustering without any explicit content labels.

---

### 5.2 Distance and Similarity Measures

The core challenge in clustering is defining **similarity**. This is typically formalised as a **distance** D(o₁, o₂): a real number between two data points. Different distance measures lead to different cluster shapes.

#### For Numerical Features

| Measure | Formula | Intuition |
|---|---|---|
| **Euclidean distance** | √(Σᵢ (oᵢ₁ − oᵢ₂)²) | Straight-line ("as the crow flies") distance |
| **Manhattan distance** | Σᵢ \|oᵢ₁ − oᵢ₂\| | "Taxicab" distance — travel only up/down then left/right |
| **Cosine distance** | 1 − (o₁·o₂) / (\|o₁\| \|o₂\|) | Angle between the two vectors; ignores magnitude |

- **Euclidean:** most common; good for compact, spherical clusters.
- **Manhattan:** less sensitive to outliers than Euclidean; useful when features have very different scales.
- **Cosine:** commonly used for text/document data (e.g., term-frequency vectors) where only direction matters.

#### For Categorical Features

| Measure | Description |
|---|---|
| **Hamming distance** | Count of positions where the two values differ (e.g., "cat" vs "car" → 1 difference) |

> **Exam tip:** Choosing the right distance measure is part of the clustering design. Euclidean on categorical data is meaningless.

---

### 5.3 Clustering Methods Overview

Two major families of clustering algorithms:

| Family | Approach | Example algorithms |
|---|---|---|
| **Hierarchical** | Build a tree of nested clusters | Agglomerative, Divisive |
| **Partitional** | Divide data into K non-overlapping clusters | K-Means, K-Medoids |
| **Density-based** | Define clusters as dense regions | DBSCAN, Mean Shift |

---

### 5.4 Hierarchical Clustering

Produces a **tree-like hierarchy** of clusters, represented as a **dendrogram**. Useful when you want to explore multiple levels of grouping without re-running the algorithm.

#### Two Directions

| Type | Direction | Description |
|---|---|---|
| **Agglomerative (bottom-up)** | Leaves → root | Start with n individual clusters; repeatedly merge the two most similar until one cluster remains. **More common and more efficient.** |
| **Divisive (top-down)** | Root → leaves | Start with one cluster; recursively split until every instance is its own cluster. Rarely used in practice. |

#### Agglomerative Clustering Algorithm (detailed)

1. Start: treat every observation as its own cluster → n clusters (leaves of the tree).
2. Compute all pairwise distances — **n(n−1)/2** pairs.
3. Find the two clusters with minimum dissimilarity; merge them.
4. Recompute distances from the new cluster to all others (using chosen linkage method).
5. Repeat steps 3–4 until only one cluster remains.

**Key property:** No random initialisation — the algorithm is **fully deterministic**. Same data always produces the same dendrogram.

**Computational cost:** Must recalculate distances at each step → **O(n²)** to O(n² log n). Expensive for large datasets.

#### Linkage Methods

How we measure the distance *between two clusters* (not individual points):

| Linkage | Distance between cluster A and cluster B | Tendency |
|---|---|---|
| **Single linkage** | Minimum pairwise distance (closest pair) | Can create long, "chained" clusters (trailing effect) |
| **Complete linkage** | Maximum pairwise distance (farthest pair) | Tends to produce compact, more balanced clusters |
| **Average linkage** | Mean of all pairwise distances | Compromise; generally produces balanced dendrograms |

> **Exam tip:** Complete and average linkage tend to give more balanced, meaningful dendrograms. Single linkage can cause the "chaining" problem.

#### Dendrograms

A **dendrogram** is the tree diagram produced by hierarchical clustering:
- Each **leaf** = one original data point.
- Each **internal node** = a merge (or split) event.
- The **height** of a merge = the dissimilarity between the merged clusters at that step.
- To obtain k clusters: cut the dendrogram horizontally at an appropriate height → k subtrees = k clusters. **No need to re-run** — one dendrogram gives all levels of clustering.

---

### 5.5 Partitional Clustering: K-Means

*(Source: `AIML231_Week7-Clustering (1).pdf` pp. 10–15; 2024 Q4c)*

**K-Means** is the most widely used partitional clustering algorithm. It partitions n instances into exactly K clusters.

#### Algorithm Steps

1. Choose K (the number of clusters).
2. **Initialise:** Place K cluster centroids randomly (in feature space).
3. **Assign:** For each instance, compute its distance to all K centroids; assign the instance to the nearest centroid.
4. **Update:** Recompute each centroid as the **mean** (average position) of all instances currently assigned to that cluster.
5. **Repeat** steps 3–4 until the centroids stop changing (convergence).

#### Strengths

- **Simple and interpretable** — intuitive algorithm with clear steps.
- **Scalable** — O(n·K·I·d) complexity (n=instances, K=clusters, I=iterations, d=features); handles large datasets well.
- **Flexible** — works with any numeric features and any distance metric.

#### Limitations

| Limitation | Why it matters |
|---|---|
| Must specify K in advance | No built-in mechanism to determine the right K |
| Stochastic (random initialisation) | Different runs may give different results; may converge to local optima |
| Requires mean to be defined | Cannot be directly applied to categorical-only data |
| Sensitive to outliers | The mean is pulled toward extreme values |
| Re-run needed for different K | Unlike hierarchical, you cannot extract different K from a single run |

---

### 5.6 Density-Based Clustering

Density-based methods define **clusters as dense regions of data points**, separated by sparse regions. They can find clusters of **arbitrary shape** and identify **outliers** automatically.

#### Mean Shift

- Places a **sliding window** (with a fixed bandwidth h) centred at each data point.
- Computes the **mean** of all points within the window.
- Shifts the window toward that mean — repeating until convergence.
- Points whose windows converge to the **same mode** (peak) are grouped into the same cluster.

**Strengths:**
- Automatically determines number of clusters (no K needed).
- Handles arbitrary cluster shapes.

**Limitations:**
- Sensitive to **bandwidth choice** (h) — acts like K in K-Means.
- Computationally expensive — O(n² per iteration).

#### DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

DBSCAN uses two parameters:
- **eps (ε):** The radius of the neighbourhood around each point.
- **minPts:** The minimum number of points required within eps to form a dense region.

**Point types:**
- **Core point:** Has at least minPts points within distance eps (including itself) → in a dense region.
- **Border point:** Within eps of a core point but has fewer than minPts neighbours itself.
- **Outlier/Noise point:** Neither a core point nor a border point → not assigned to any cluster.

**Algorithm:** Start from an unvisited core point; expand the cluster by recursively adding all density-reachable points (core points within eps, and their neighbourhood). Move to unvisited points until done.

**Strengths:**
- Handles **arbitrary shapes** — not limited to spherical clusters.
- Automatically identifies **outliers** as noise.
- No need to specify K in advance.

**Limitations:**
- Struggles with **varying density** — one eps/minPts pair cannot handle clusters of different densities.
- Sensitive to parameter selection (eps, minPts).
- Poor performance in **high-dimensional** data (distances become uniform → curse of dimensionality).

| | K-Means | Agglomerative | DBSCAN | Mean Shift |
|---|---|---|---|---|
| Specify K? | Yes | No | No | No |
| Deterministic? | No | Yes | Yes | Yes |
| Handles outliers? | No | No | Yes | No |
| Arbitrary shapes? | No | Partially | Yes | Yes |
| Large datasets? | Yes | No (O(n²)) | Yes (with index) | No |

---

### 5.7 Clustering Performance Metrics

Since there are no ground-truth labels in unsupervised clustering, we evaluate using **internal metrics** that measure compactness and separability.

#### Core Principle

Good clusters should be:
- **Compact** (low intra-cluster distance) — instances within a cluster are close to each other.
- **Well-separated** (high inter-cluster distance) — clusters are far from each other.

| Metric | Description |
|---|---|
| **Intra-cluster distance** | Average within-cluster distances — *minimise* |
| **Inter-cluster distance** | Average between-cluster distances — *maximise* |

#### Silhouette Score

For each instance i, the silhouette score s(i) measures how well it fits its assigned cluster compared to the nearest alternative cluster:

```
s(i) = (b(i) − a(i)) / max(a(i), b(i))
```

Where:
- **a(i)** = average distance from i to all other instances in the **same** cluster (intra-cluster cohesion).
- **b(i)** = minimum average distance from i to instances in any **other** cluster (nearest-cluster separation).

| Score | Interpretation |
|---|---|
| s(i) = 1 | Instance is perfectly placed in its cluster. |
| s(i) = 0 | Instance is on the boundary between two clusters. |
| s(i) = −1 | Instance is misclassified — closer to a neighbouring cluster. |

The overall **silhouette coefficient** = average s(i) across all instances. Closer to 1 = better clustering.

#### Other Internal Indices (mentioned for completeness)

- **Davies-Bouldin Index** — ratio of intra-cluster scatter to inter-cluster separation; *lower* is better.
- **Dunn Index** — ratio of minimum inter-cluster distance to maximum intra-cluster distance; *higher* is better.
- **Calinski-Harabasz Index** — ratio of inter-cluster dispersion to intra-cluster dispersion; *higher* is better.

---

### 5.8 Typical Pitfalls

- Saying clustering "requires training labels" — it does **NOT** (unsupervised).
- Confusing K-Means (**stochastic**) with agglomerative (**deterministic**).
- Forgetting the Silhouette score range is [−1, 1], not [0, 1].
- For large datasets, K-Means scales far better than agglomerative (pairwise distances are O(n²)).
- Forgetting that K-Means requires **re-running** for each different K, whereas a single dendrogram can be cut at any height.
- Saying DBSCAN needs K — it does not; K is automatic from density.
- Confusing **eps** and **minPts** in DBSCAN — eps is the radius, minPts is the point count threshold.

---

## Quick-Reference Summary Table

| Topic | Key facts to memorise |
|---|---|
| AI vs ML | AI = broad field; ML = AI subfield that learns from data without explicit programming |
| Four ML elements | Data, Task, Model, Algorithm |
| Data representation | Instances = rows; Features (X) = input columns; Labels (y) = output column |
| Train/Val/Test split | Train = fit model; Validation = tune hyperparameters; Test = final unbiased evaluation |
| ML workflow | Define → Collect → Preprocess → Train → Evaluate → Deploy |
| ML Types | Supervised (labels), Unsupervised (no labels), Reinforcement (rewards) |
| KNN large K | Underfitting, over-smoothing, slow prediction |
| Entropy | H=0 pure; H=1 max uncertainty (binary) |
| Cross-val | K−1 folds train, 1 fold test; average K estimates |
| CRISP-DM | 6 phases: Business Understand → Data Understand → Data Prep → Model → Evaluate → Deploy |
| Data leakage | Always split first, then fit preprocessor on train only |
| Min-max | x'=(x−min)/(max−min) → [0,1] |
| Z-score | z=(x−μ)/σ → mean 0, std 1 |
| SFFS | Start empty; add best feature; stop at target size |
| PCA | Unsupervised filter; max d PCs; orthogonal components |
| Linear regression weights | d features → d+1 weights (d slopes + 1 intercept) |
| Lasso vs Ridge | Lasso zeroes weights (feature selection); Ridge shrinks but keeps all |
| Clustering | Unsupervised; no labels; finds natural groups |
| Customer segmentation | Group users by behaviour; Netflix uses clustering for recommendations |
| Distance measures | Euclidean (straight line); Manhattan (taxicab); Cosine (angle, text); Hamming (categorical) |
| K-Means limits | Need K; stochastic; mean-based only; sensitive to outliers |
| K-Means strengths | Simple; scalable O(n·K·I·d); flexible |
| Agglomerative | Deterministic; no K needed; O(n²); dendrogram; bottom-up |
| Divisive | Top-down hierarchical; less common than agglomerative |
| Linkage methods | Single=min; Complete=max (balanced); Average=mean (balanced) |
| Dendrogram | Tree of merges; cut at height h → k clusters; no re-run needed |
| DBSCAN params | eps (radius) + minPts (threshold); core/border/noise points |
| DBSCAN strengths | No K; arbitrary shapes; finds outliers |
| DBSCAN limits | Struggles with varying density; high-dimensional data |
| Mean Shift | No K needed; finds modes; sensitive to bandwidth; expensive |
| Silhouette | Range [−1,1]; higher = better match to cluster; formula uses a(i) and b(i) |
| Other cluster metrics | Davies-Bouldin (lower better); Dunn (higher better); Calinski-Harabasz (higher better) |
| FNR + TPR = 1 | FPR + TNR = 1 |
| F1 | Harmonic mean of Precision and Recall |
| R² | ≤1; 1=perfect; 0=no better than mean; recommended for regression |
