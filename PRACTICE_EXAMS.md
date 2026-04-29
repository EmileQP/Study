# AIML 231 – Practice Exams and Question Bank

> Based on: `AIML231_Midterm_Test_Frontpage.pdf` (2026 exam structure) and `2024_1_AIML231.pdf` (2024 exam style reference), plus lecture slides (Weeks 1–7).

---

## Practice Exam 1

**AIML 231 – Techniques in Machine Learning**  
Time: 50 minutes | Total: 50 marks | Closed book  
Permitted: silent non-programmable calculator; non-electronic foreign language dictionary  
*Attempt all questions. Show all working where numerical answers are required.*

---

### Question 1 – Machine Learning Basics [7 marks]

**(a) [2 marks]**  
Briefly explain the difference between **supervised** and **unsupervised** machine learning. Give one example task for each.

**(b) [2 marks]**  
Explain the concept of **overfitting** in machine learning. What symptom would you observe in a model's training accuracy vs. test accuracy if it is overfitting?

**(c) [3 marks]**  
Identify **three key differences** between supervised machine learning and reinforcement learning.

---

### Question 2 – Classification [18 marks]

**(a) [2 marks]**  
Briefly state **one key difference** between classification and regression.

**(b) [3 marks]**  
Explain the role of each of the three data subsets used in supervised machine learning: **training set**, **validation set**, and **test set**.

**(c) [3 marks]**  
Describe the **K-Fold Cross Validation** method. State **one advantage** it has over a simple holdout (single train/test split).

**(d) [2 marks]**  
When using KNN and the hyperparameter K is set to a **very small value** (e.g., K = 1), what two problems might occur?

**(e) [3 marks]**  
A fraud detection system was tested on 500 transactions. Of these, 50 are actual fraud cases and 450 are legitimate. The system correctly identifies 40 fraud cases but also incorrectly flags 30 legitimate transactions as fraud.

Calculate the following. Show your working.
1. True Positive Rate (TPR / Sensitivity)
2. False Positive Rate (FPR)
3. Precision

**(f) [5 marks]**  
Consider the following dataset of 8 loan applications, each labelled Approved or Rejected, and described by two features: Credit Score (Good/Bad) and Employment Status (Employed/Unemployed).

| # | Credit Score | Employment | Outcome |
|---|---|---|---|
| 1 | Good | Employed | Approved |
| 2 | Good | Employed | Approved |
| 3 | Good | Unemployed | Approved |
| 4 | Bad | Employed | Approved |
| 5 | Good | Unemployed | Rejected |
| 6 | Bad | Employed | Rejected |
| 7 | Bad | Unemployed | Rejected |
| 8 | Bad | Unemployed | Rejected |

Given the entropy and information gain formulas:

```
H(Y) = −Σ P(Y=yᵢ) log₂ P(Y=yᵢ)
H(Y|X) = −Σⱼ P(X=xⱼ) Σᵢ P(Y=yᵢ|X=xⱼ) log₂ P(Y=yᵢ|X=xⱼ)
IG(Y,X) = H(Y) − H(Y|X)
```

Calculate the information gain of using the **Credit Score** feature to split this dataset. Show all steps.

---

### Question 3 – ML Pipeline and Data Pre-processing [15 marks]

**(a) [3 marks]**  
List the **six phases** of the CRISP-DM process model. Give a one-sentence description of each phase.

**(b) [2 marks]**  
What is **data leakage** in the context of a machine learning pipeline? Give one concrete example of how it can occur during data preprocessing.

**(c) [2 marks]**  
Explain the difference between **min-max scaling** and **z-score standardisation**. When would you prefer each?

**(d) [3 marks]**  
**Feature construction** and **feature selection** are two approaches to improving the quality of features used in machine learning.
1. State the main difference between feature construction and feature selection.
2. State which approach PCA belongs to, and justify your answer.

**(e) [5 marks]**  
You are applying **Sequential Forward Feature Selection (SFFS)** to choose 2 features from the set {f₁, f₂, f₃, f₄}. The classification accuracies for all relevant subsets are:

| Subset | Accuracy |
|---|---|
| {f₁} | 0.55 |
| {f₂} | 0.70 |
| {f₃} | 0.60 |
| {f₄} | 0.50 |
| {f₁, f₂} | 0.80 |
| {f₁, f₃} | 0.72 |
| {f₁, f₄} | 0.65 |
| {f₂, f₃} | 0.85 |
| {f₂, f₄} | 0.78 |
| {f₃, f₄} | 0.68 |

Show which feature is added at each step. Justify your answer and state the final selected feature subset.

---

### Question 4 – Regression [5 marks]

**(a) [2 marks]**  
A linear regression model is trained on a dataset with **8 features**. How many **weights** does the weight vector contain? Briefly justify your answer.

**(b) [3 marks]**  
Ridge regression and Lasso regression extend linear regression by adding a regularisation term.
1. Write the objective function that Lasso regression minimises.
2. Which of Ridge or Lasso can be used as an embedded feature selection method? Justify your answer.

---

### Question 5 – Clustering [5 marks]

**(a) [3 marks]**  
Describe the main steps of the **K-Means** clustering algorithm. List **two limitations** of K-Means.

**(b) [2 marks]**  
**Agglomerative clustering** and K-Means are both clustering algorithms. Suppose you have a dataset of 500,000 instances and you need to cluster them. Which algorithm is more computationally efficient for this task? Briefly justify your answer.

---

## Practice Exam 1 — Answer Key and Marking Guide

---

### Q1 Answers [7 marks]

**(a) [2 marks]**
- **Supervised:** Training data has labels; model learns to map inputs to outputs. Example: email classification (spam/not spam). *(1 mark)*
- **Unsupervised:** No labels; model discovers structure/patterns. Example: customer clustering. *(1 mark)*

**(b) [2 marks]**
- Overfitting: model learns the training data too well, including noise, so it fails to generalise. *(1 mark)*
- Symptom: very high training accuracy (e.g., ≈100%) but significantly lower test accuracy. *(1 mark)*

**(c) [3 marks]** (1 mark each for any three)
1. Supervised learning uses per-sample labels; RL uses delayed reward/punishment signals.
2. Supervised learning has a fixed labelled dataset; RL interacts with an environment dynamically.
3. Goal in supervised = match teacher's output; goal in RL = maximise cumulative reward.

---

### Q2 Answers [18 marks]

**(a) [2 marks]**
- Classification predicts a **discrete/categorical** output (class label); regression predicts a **continuous** numerical value. *(2 marks for a correct distinction)*

**(b) [3 marks]** (1 mark each)
- Training set: used to fit/train the model.
- Validation set: used to tune hyperparameters and select the best model; not used in final evaluation.
- Test set: held out until the very end; used only to estimate real-world performance.

**(c) [3 marks]**
- Partition dataset into K equal folds. *(1 mark)*
- Train on K−1 folds, test on the remaining fold; repeat K times. Average the K test results. *(1 mark)*
- Advantage: all data used for both training and testing; more reliable estimate than a single holdout. *(1 mark)*

**(d) [2 marks]**
- K=1: overfitting — model is too sensitive to noise and individual outliers. *(1 mark)*
- K=1: highly variable predictions — small changes in data → large changes in output. *(1 mark)*
*(Accept: high variance, non-smooth decision boundaries, memorises training data.)*

**(e) [3 marks]**
- TP = 40, FN = 10, FP = 30, TN = 420.
- **TPR = TP/(TP+FN) = 40/50 = 0.80 (80%)** *(1 mark including working)*
- **FPR = FP/(FP+TN) = 30/450 ≈ 0.067 (6.7%)** *(1 mark including working)*
- **Precision = TP/(TP+FP) = 40/70 ≈ 0.571 (57.1%)** *(1 mark including working)*

**(f) [5 marks]**

Step 1 — H(Y):  
P(Approved) = 4/8 = 0.5, P(Rejected) = 4/8 = 0.5  
H(Y) = −(0.5 log₂ 0.5 + 0.5 log₂ 0.5) = −(−0.5 − 0.5) = **1.0 bit** *(1 mark)*

Step 2 — H(Y | Credit Score):
- Good (4 items: rows 1,2,3,5): 3 Approved, 1 Rejected → H = −(3/4 log₂(3/4) + 1/4 log₂(1/4)) = −(−0.311 − 0.5) = **0.811 bits** *(1 mark)*
- Bad (4 items: rows 4,6,7,8): 1 Approved, 3 Rejected → H = same = **0.811 bits** *(by symmetry)* *(1 mark)*
- H(Y | Credit Score) = 4/8 × 0.811 + 4/8 × 0.811 = **0.811 bits** *(1 mark)*

Step 3 — IG:  
IG = 1.0 − 0.811 = **0.189 bits** *(1 mark)*

*(Award partial marks for correct method with arithmetic errors.)*

---

### Q3 Answers [15 marks]

**(a) [3 marks]** (0.5 mark per phase name, 0.5 mark per description — total 6 items × 0.5 = 3)
1. Business Understanding — define objectives and success criteria with the customer.
2. Data Understanding — collect, describe, explore, and check quality of data.
3. Data Preparation — clean, select, construct, and format data (~80% of effort).
4. Modelling — select and train ML models, tune hyperparameters.
5. Evaluation — assess whether model meets business objectives; review process.
6. Deployment — deploy model; plan monitoring, maintenance, and retraining.

**(b) [2 marks]**
- Definition: Information from the test/holdout set leaks into the training process, giving an overly optimistic performance estimate. *(1 mark)*
- Example: Fitting a normalisation scaler on the entire dataset before splitting into train/test — the training process has "seen" test set statistics. *(1 mark)*

**(c) [2 marks]**
- Min-max: scales to [0,1] using x' = (x−min)/(max−min); very sensitive to outliers; use when features are not normally distributed. *(1 mark)*
- Z-score: scales to mean=0, std=1 using z=(x−μ)/σ; use when features are normally distributed. *(1 mark)*

**(d) [3 marks]**
1. Feature selection chooses a *subset* of existing features (no new features created); feature construction builds *new* features from original ones. *(1 mark)*
2. PCA belongs to the **filter** approach. *(1 mark)* Justification: PCA uses only the statistical properties of the data (covariance matrix) to construct principal components, without using a learning algorithm or class labels. *(1 mark)*

**(e) [5 marks]**

**Iteration 1 (start = {}):** Evaluate {f₁}=0.55, {f₂}=0.70, {f₃}=0.60, {f₄}=0.50.  
Best: **f₂** (0.70) → add f₂. Current subset = {f₂}. *(2 marks: correct identification + justification)*

**Iteration 2 (current = {f₂}):** Evaluate {f₂,f₁}=0.80, {f₂,f₃}=0.85, {f₂,f₄}=0.78.  
Best: **f₃** with {f₂,f₃}=0.85 → add f₃. Current subset = {f₂,f₃}. *(2 marks: correct identification + justification)*

**Termination:** 2 features selected as required. Final subset = **{f₂, f₃}**. *(1 mark)*

---

### Q4 Answers [5 marks]

**(a) [2 marks]**
- **9 weights** *(1 mark for correct number)*
- Justification: one weight (slope) per feature = 8 weights, plus one intercept w₀ = 9 total. A linear regression with d features has d+1 weights. *(1 mark)*

**(b) [3 marks]**
1. Lasso objective: **RSS + λ Σᵢ |wᵢ|** (sum of absolute values of slopes, excluding intercept). *(1 mark)*
2. **Lasso** can be used as embedded feature selection. *(1 mark)*  
   Justification: the L1 penalty can drive some weights to exactly zero, effectively removing those features from the model. Ridge's L2 penalty only shrinks weights toward zero but never to exactly zero. *(1 mark)*

---

### Q5 Answers [5 marks]

**(a) [3 marks]**

Steps of K-Means *(2 marks — award 1 mark for partial description of 2+ steps)*:
1. Initialise K cluster centroids randomly.
2. Assign each instance to the nearest centroid (by Euclidean distance).
3. Recompute each centroid as the mean of assigned instances.
4. Repeat steps 2–3 until centroids no longer change.

Two limitations *(1 mark for any two)*:
- Must specify K in advance.
- Stochastic: different random initialisations may produce different results (local optima).
- Only works when the mean is defined (not suitable for purely categorical data).
- Sensitive to outliers.

**(b) [2 marks]**
- **K-Means** is more computationally efficient. *(1 mark)*
- Justification: Agglomerative clustering requires computing all pairwise distances (O(n²)), which is prohibitive for 500,000 instances. K-Means scales approximately linearly with the number of instances. *(1 mark)*

---

---

## Practice Exam 2

**AIML 231 – Techniques in Machine Learning**  
Time: 50 minutes | Total: 50 marks | Closed book  
Permitted: silent non-programmable calculator; non-electronic foreign language dictionary  
*Attempt all questions. Show all working where numerical answers are required.*

---

### Question 1 – Machine Learning Basics [7 marks]

**(a) [2 marks]**  
Explain what the **bias–variance trade-off** is in machine learning. Name one model characteristic associated with high bias and one associated with high variance.

**(b) [2 marks]**  
Explain the **curse of dimensionality** in machine learning. State one algorithm that is particularly affected by it.

**(c) [3 marks]**  
Identify **three key differences** between supervised machine learning and unsupervised machine learning.

---

### Question 2 – Classification [18 marks]

**(a) [2 marks]**  
Briefly describe the **Random Forest** classifier. State one advantage Random Forests have over a single decision tree.

**(b) [3 marks]**  
Describe the concept of **information gain** in decision tree learning. Why is a feature with high information gain preferred for splitting?

**(c) [4 marks]**  
Briefly describe the **K-Fold Cross Validation** method. Explain when Leave-One-Out Cross Validation should be preferred over 10-Fold Cross Validation.

**(d) [2 marks]**  
Define **precision** and **recall** (also called TPR/Sensitivity). In which type of problem is recall especially important?

**(e) [3 marks]**  
A disease diagnostic system was tested on 300 patients: 60 have the disease (positive) and 240 do not (negative). The system correctly identifies 48 patients with the disease and incorrectly classifies 24 healthy patients as having the disease.

Calculate:
1. Accuracy
2. F1-Score

Show all working.

**(f) [4 marks]**  
Consider the following 6 instances and the feature **Weather** (Sunny/Rainy):

| # | Weather | Outcome |
|---|---|---|
| 1 | Sunny | Play |
| 2 | Sunny | Play |
| 3 | Sunny | No Play |
| 4 | Rainy | Play |
| 5 | Rainy | No Play |
| 6 | Rainy | No Play |

1. (2 marks) Calculate the entropy of the full dataset H(Outcome).
2. (2 marks) Calculate the information gain of using **Weather** to split the dataset. Show all working.

---

### Question 3 – ML Pipeline and Data Pre-processing [15 marks]

**(a) [2 marks]**  
Distinguish between **equal-width (uniform)** and **equal-depth (quantile)** discretisation. Give one advantage and one disadvantage of each.

**(b) [3 marks]**  
A student proposes the following preprocessing workflow for a classification task with a dataset of 1000 instances:

> "I will normalise all 1000 instances using min-max scaling. Then I will split into 800 training and 200 test instances. Then I will train a KNN classifier."

Is this an appropriate workflow? Identify any problems and describe the correct approach.

**(c) [2 marks]**  
Describe **one-hot encoding** and **ordinal encoding**. When should each be used?

**(d) [4 marks]**  
Principal Component Analysis (PCA) is a feature construction technique.

1. (1 mark) Is PCA a supervised or unsupervised algorithm? Justify your answer.
2. (1 mark) A dataset has 15 original features. What is the maximum number of principal components that can be built?
3. (2 marks) List **two limitations** of PCA.

**(e) [4 marks]**  
You are applying **Sequential Backward Feature Selection (SBFS)** to select 2 features from the set {f₁, f₂, f₃}. The classification accuracies for all relevant subsets are:

| Subset | Accuracy |
|---|---|
| {f₁, f₂, f₃} | 0.82 |
| {f₂, f₃} (remove f₁) | 0.88 |
| {f₁, f₃} (remove f₂) | 0.75 |
| {f₁, f₂} (remove f₃) | 0.70 |
| {f₂} (remove f₁ then f₃) | 0.65 |
| {f₃} (remove f₁ then f₂) | 0.79 |
| {f₁} (remove f₂ then f₃) | 0.55 |

Show which feature is removed at each step. State the final selected feature subset.

---

### Question 4 – Regression [5 marks]

**(a) [2 marks]**  
Describe the **Residual Sum of Squares (RSS)** used in linear regression. Why do we square the residuals rather than sum them directly?

**(b) [1 mark]**  
Define R² (R-squared) and state the range of values it can take.

**(c) [2 marks]**  
Explain how **Ridge regression** differs from standard linear regression. What effect does increasing the regularisation parameter λ have?

---

### Question 5 – Clustering [5 marks]

**(a) [2 marks]**  
Describe the **Silhouette Score**. What does a score close to 1, close to 0, and close to −1 indicate?

**(b) [3 marks]**  
Compare **K-Means** and **Agglomerative Clustering** on the following criteria:
1. Whether K needs to be specified in advance.
2. Whether the algorithm is deterministic.
3. Computational complexity relative to number of instances n.

---

## Practice Exam 2 — Answer Key and Marking Guide

---

### Q1 Answers [7 marks]

**(a) [2 marks]**
- Bias–variance trade-off: Increasing model complexity reduces bias but increases variance; there is a trade-off between under- and overfitting. Total expected error = bias + variance + irreducible noise. *(1 mark)*
- High bias → overly simple model (e.g., linear model on non-linear data) = underfitting. High variance → overly complex model that memorises training noise = overfitting. *(1 mark)*

**(b) [2 marks]**
- As the number of features increases, data becomes exponentially sparse in the feature space; a fixed number of training points covers a shrinking fraction of the space, making learning harder. *(1 mark)*
- KNN is particularly affected — distances become meaningless in high dimensions. *(1 mark)*

**(c) [3 marks]** (1 mark each for any three)
1. Supervised requires labelled data; unsupervised does not.
2. Supervised learns to predict a target output (classification/regression); unsupervised discovers hidden structure (clustering/dimensionality reduction).
3. Supervised can be evaluated against ground-truth labels; unsupervised evaluation is more subjective.

---

### Q2 Answers [18 marks]

**(a) [2 marks]**
- Random Forest: ensemble of decision trees, each trained on a bootstrap sample; each split uses a random feature subset; final prediction = majority vote. *(1 mark)*
- Advantage: reduces overfitting (variance) compared to a single tree due to averaging / diversity of trees. *(1 mark)*

**(b) [3 marks]**
- Information gain = reduction in entropy after splitting on a feature: IG(Y,X) = H(Y) − H(Y|X). *(1 mark)*
- A feature with high IG creates more "pure" subsets, meaning the class distribution is more certain after the split. *(1 mark)*
- This helps build smaller, more accurate trees — we prefer the feature that most reduces uncertainty. *(1 mark)*

**(c) [4 marks]**
- K-Fold: divide data into K folds; in each experiment use K−1 folds to train, 1 fold to test; repeat K times and average estimates. *(2 marks)*
- LOO preferred when: dataset is very small (e.g., n < 50), where losing any data for testing is costly; LOO is effectively n-fold CV (k=n), maximising training data at the cost of high computational expense. *(2 marks)*

**(d) [2 marks]**
- **Precision** = TP/(TP+FP): of instances predicted positive, how many are actually positive. *(0.5 mark)*
- **Recall (TPR)** = TP/(TP+FN): of actual positive instances, how many are detected. *(0.5 mark)*
- Recall is especially important in problems where missing a positive case has serious consequences — e.g., disease diagnosis (missing a diseased patient is dangerous); fraud detection. *(1 mark)*

**(e) [3 marks]**

TP=48, FN=12, FP=24, TN=216.
Total = 300.

**Accuracy** = (TP+TN)/Total = (48+216)/300 = 264/300 = **0.88 (88%)** *(1 mark with working)*

Precision = TP/(TP+FP) = 48/(48+24) = 48/72 = 0.667  
Recall = TP/(TP+FN) = 48/(48+12) = 48/60 = 0.8  
**F1 = 2 × (0.667 × 0.8) / (0.667 + 0.8) = 2 × 0.533 / 1.467 ≈ 0.727** *(2 marks with working; 1 mark for method, 1 for correct answer)*

**(f) [4 marks]**

Step 1 — H(Outcome):  
P(Play) = 3/6 = 0.5, P(No Play) = 3/6 = 0.5  
H(Outcome) = −(0.5 log₂ 0.5 + 0.5 log₂ 0.5) = **1.0 bit** *(1 mark)*

Step 2 — H(Outcome | Weather):
- Sunny (3 items): 2 Play, 1 No Play  
  H = −(2/3 log₂(2/3) + 1/3 log₂(1/3)) = −(−0.390 − 0.528) = **0.918 bits** *(0.5 mark)*
- Rainy (3 items): 1 Play, 2 No Play  
  H = −(1/3 log₂(1/3) + 2/3 log₂(2/3)) = **0.918 bits** *(0.5 mark)*
- H(Y|Weather) = 3/6 × 0.918 + 3/6 × 0.918 = **0.918 bits** *(0.5 mark)*

Step 3 — IG = 1.0 − 0.918 = **0.082 bits** *(0.5 mark)*

---

### Q3 Answers [15 marks]

**(a) [2 marks]** (0.5 mark per advantage/disadvantage, 0.5 per method name)
- **Equal-width:** Divides range [min, max] into N bins of width (max−min)/N. Advantage: preserves numeric scale/spacing. Disadvantage: uneven sample distribution with skewed data (some bins may be empty). *(1 mark)*
- **Equal-depth:** Puts same number of samples in each bin (sort then divide). Advantage: handles skewed distributions well. Disadvantage: varying bin widths make interpretation harder. *(1 mark)*

**(b) [3 marks]**
- Problem: **data leakage**. The student normalised using min-max scaling on the entire 1000-instance dataset before splitting. The test set statistics (min, max) influenced the normalisation, so the test performance estimate is optimistic. *(2 marks — 1 for identifying data leakage, 1 for correct explanation)*
- Correct approach: split into train/test first; then fit the scaler on the training set only; apply that fitted scaler to both training and test sets. *(1 mark)*

**(c) [2 marks]**
- **One-hot encoding:** Each category becomes its own binary column. Use for *nominal* features (no inherent ordering, e.g., colour, city). *(1 mark)*
- **Ordinal encoding:** Map categories to integers preserving order. Use for *ordinal* features (natural ordering, e.g., XS < S < M < L < XL; Low < Medium < High). *(1 mark)*

**(d) [4 marks]**
1. **Unsupervised** — PCA uses only the covariance structure of the data (feature variances and co-variances); it does not use class labels. *(1 mark)*
2. **Maximum 15 principal components** — with d original features, at most d PCs can be built (one per eigenvector of the d×d covariance matrix). *(1 mark)*
3. Two limitations *(1 mark each, any two)*:
   - Assumes a *linear* mapping — cannot capture non-linear relationships (Kernel PCA needed).
   - Must choose the number of components p — not fully automatic.
   - Constructed components lose interpretability (linear combinations with no direct meaning).
   - Uncorrelated features may not be sufficient for non-linearly separable data.

**(e) [4 marks]**

**Iteration 1 (start = {f₁, f₂, f₃} = 0.82):** Evaluate removing each feature:
- Remove f₁ → {f₂, f₃} = 0.88  
- Remove f₂ → {f₁, f₃} = 0.75  
- Remove f₃ → {f₁, f₂} = 0.70  
Best: remove **f₁** → {f₂, f₃} gives highest accuracy 0.88. *(2 marks: correct removal + justification)*

**Iteration 2 (current = {f₂, f₃}):** Evaluate removing each remaining feature:
- Remove f₂ → {f₃} = 0.79  
- Remove f₃ → {f₂} = 0.65  
Best: remove **f₃** → {f₂} = 0.79. BUT we want 2 features — target reached already at end of Iteration 1.

*(Award full marks if student correctly terminates after Iteration 1 with {f₂, f₃}, since 2 features is the target.)*

Final selected features: **{f₂, f₃}** with accuracy 0.88. *(2 marks: correct stopping + correct final subset)*

---

### Q4 Answers [5 marks]

**(a) [2 marks]**
- RSS = Σᵢ(yᵢ − f(xᵢ))² — sum of squared differences between actual and predicted values for all training instances. *(1 mark)*
- Residuals are squared because: (1) squared values are always non-negative (positive and negative errors don't cancel); (2) larger errors are penalised more heavily; (3) RSS is differentiable, enabling gradient-based optimisation. *(1 mark — any one valid reason)*

**(b) [1 mark]**
- R² = 1 − RSS / Σ(yᵢ − ȳ)²; range: R² ≤ 1 (can be negative for very poor models). R²=1 perfect, R²=0 no better than predicting the mean, R²<0 worse than mean predictor. *(1 mark)*

**(c) [2 marks]**
- Ridge regression adds a penalty λ Σwₖ² to the RSS objective, penalising large weights. *(1 mark)*
- Increasing λ: the regularisation penalty grows stronger, shrinking weights further toward zero → reduces overfitting (higher bias, lower variance); at very large λ the model approaches the zero-weight solution. *(1 mark)*

---

### Q5 Answers [5 marks]

**(a) [2 marks]**
- The silhouette score for instance i: s(i) = (b(i) − a(i)) / max(a(i), b(i)), where a(i) = avg distance to instances in the same cluster, b(i) = min avg distance to instances in any other cluster. Scores averaged over all instances. *(1 mark)*
- Score ≈ 1: instance is well-matched to its cluster (much closer to own cluster than others). *(0.5 mark)*
- Score ≈ 0: instance is on the border between clusters. *(0.25 mark)*
- Score ≈ −1: instance is better matched to a neighbouring cluster (likely misclustered). *(0.25 mark)*

**(b) [3 marks]** (1 mark per criterion)

| Criterion | K-Means | Agglomerative |
|---|---|---|
| K specified in advance | Yes — must choose K | No — K determined by cutting the dendrogram |
| Deterministic | No — random centroid initialisation | Yes — no random initialisation |
| Complexity with n | O(n·K·I·d) — approx linear in n | O(n² log n) — quadratic; problematic for large n |

---

---

## Question Bank (Short Practice Questions per Topic)

### Topic 1: Machine Learning Basics

**Q1-S1.** Define machine learning in one sentence and name the four elements of any ML problem.  
*Answer: ML is a sub-field of AI where systems learn from data to improve performance without being explicitly programmed. Elements: data, task, model, algorithm.*

**Q1-S2.** What is the difference between an ML model and an ML algorithm? Give one example of each.  
*Answer: Model = final trained output that makes predictions (e.g., trained decision tree). Algorithm = procedure used to train it (e.g., ID3, gradient descent).*

**Q1-S3.** Name and briefly describe the three main types of ML supervision.  
*Answer: Supervised (labelled data, teacher); Unsupervised (no labels, finds structure); Reinforcement (reward/punishment, no per-sample labels).*

**Q1-S4.** Explain the difference between bias and variance in ML.  
*Answer: Bias = error from overly simple model assumptions (underfitting). Variance = error from sensitivity to training data fluctuations (overfitting).*

**Q1-S5.** A model achieves 99% training accuracy and 55% test accuracy. Is this overfitting or underfitting? Explain.  
*Answer: Overfitting. The model has memorised the training data including noise and fails to generalise to unseen data.*

**Q1-S6.** What is the difference between a **feature** and a **label** in a supervised ML dataset?  
*Answer: A feature (X) is an input variable describing an instance (e.g., age, pixel value). A label (y) is the output variable the model is trained to predict (e.g., spam/not spam, house price). Features are used as inputs; labels are the targets.*

**Q1-S7.** What are the three roles of the training set, validation set, and test set? Why must they be kept separate?  
*Answer: Training set — used to fit model parameters; validation set — used to tune hyperparameters and select among models (not seen during training); test set — used once at the very end to estimate real-world performance. They must be separate to avoid data leakage and ensure honest performance estimates.*

**Q1-S8.** List the general steps of a typical machine learning workflow in order.  
*Answer: (1) Define the problem; (2) Collect and explore data; (3) Preprocess data (split first, then scale/encode); (4) Train model; (5) Evaluate and tune on validation set; (6) Final evaluation on test set; (7) Deploy.*

**Q1-S9.** Name three numerical and one categorical feature type. What distance measure is appropriate for categorical features?  
*Answer: Numerical: height, temperature, pixel intensity. Categorical example: colour, gender, postcode. Categorical → Hamming distance.*

**Q1-S10.** A model scores 60% on both training and test sets. Is this overfitting, underfitting, or a good fit? What would you do to improve it?  
*Answer: Underfitting — the model performs poorly on both sets, suggesting it is too simple. To improve: increase model complexity, add more informative features, reduce regularisation, or try a more powerful algorithm.*

---

### Topic 2: Classification

**Q2-S1.** What is a decision boundary in classification?  
*Answer: The surface/line that separates different class regions in feature space. Simple classifiers use linear boundaries; complex classifiers can produce curved/non-linear boundaries.*

**Q2-S2.** Calculate the Euclidean distance between points (2, 4) and (5, 8).  
*Answer: √((5−2)² + (8−4)²) = √(9+16) = √25 = 5.*

**Q2-S3.** A classifier achieves TP=80, TN=170, FP=30, FN=20. Calculate accuracy, TPR, and FPR.  
*Answer: Total=300. Accuracy=(80+170)/300=0.833. TPR=80/(80+20)=0.8. FPR=30/(30+170)=0.15.*

**Q2-S4.** What is the AUC of an ROC curve? What values indicate a good classifier vs. random guessing?  
*Answer: AUC = area under the ROC curve; range (0.5, 1.0]. AUC=1.0 is perfect; AUC=0.5 is random guessing (worst case).*

**Q2-S5.** Why is K=10 a common choice for K-Fold Cross Validation?  
*Answer: It balances between computational cost and accuracy of the error estimate. Large K gives accurate estimates but is slow; small K is fast but less reliable. K=10 is an empirical sweet spot.*

**Q2-S6.** Explain pre-pruning vs. post-pruning in decision trees.  
*Answer: Pre-pruning stops tree growth early (when gain < threshold or node too small). Post-pruning grows a full tree then removes branches that improve generalisation.*

**Q2-S7.** What is the entropy of a node with 50% positive and 50% negative cases?  
*Answer: H = −(0.5 log₂ 0.5 + 0.5 log₂ 0.5) = −(−0.5 − 0.5) = 1.0 bit.*

**Q2-S8.** What is the entropy of a fully pure node (all instances in one class)?  
*Answer: H = −(1 × log₂ 1) = 0. A pure node has zero entropy.*

**Q2-S9.** What distinguishes KNN from other classifiers?  
*Answer: KNN is a "lazy learner" — it stores all training data and makes predictions at query time without building an explicit model. No training phase; prediction searches training data.*

**Q2-S10.** Calculate F1-Score given Precision=0.75 and Recall=0.60.  
*Answer: F1 = 2 × (0.75 × 0.60) / (0.75 + 0.60) = 2 × 0.45 / 1.35 = 0.667.*

---

### Topic 3: ML Pipeline and Data Pre-processing

**Q3-S1.** Name the six phases of CRISP-DM in order.  
*Answer: 1. Business Understanding 2. Data Understanding 3. Data Preparation 4. Modelling 5. Evaluation 6. Deployment.*

**Q3-S2.** What is model staleness? Name two types of drift.  
*Answer: Model staleness = when a deployed model no longer reflects current system behaviour. Data drift (P(x) shifts) and Concept drift (P(y|x) changes).*

**Q3-S3.** A numerical feature has values [10, 20, 30, 100]. Apply min-max scaling to the value 30.  
*Answer: x' = (30−10)/(100−10) = 20/90 ≈ 0.222.*

**Q3-S4.** A feature has mean=50 and std=10. Apply z-score standardisation to the value 65.  
*Answer: z = (65−50)/10 = 1.5.*

**Q3-S5.** You have a dataset with feature values [2, 4, 6, 8, 10, 12] and want 3 equal-width bins. What are the bins?  
*Answer: Width = (12−2)/3 = 3.33. Bins: [2–5.33], [5.33–8.67], [8.67–12].*

**Q3-S6.** Explain the nesting effect in sequential feature selection.  
*Answer: In SFFS, once a feature is added it cannot be removed later (nested forward search). In SBFS, once a feature is removed it cannot be added back. This can lead to suboptimal subsets.*

**Q3-S7.** What are the three categories of feature construction approaches? Give one example of each.  
*Answer: Filter (e.g., PCA — uses data statistics only); Wrapper (e.g., SFFS with a classifier — evaluates via model performance); Embedded (e.g., Lasso regression — feature selection built into training).*

**Q3-S8.** What is the "explained variance ratio" of a principal component and how is it used?  
*Answer: ExpVar(zₗ) = λₗ / Σλₖ — the fraction of total data variance captured by PC zₗ. Used to select the number of PCs (e.g., choose p such that cumulative ratio ≥ 0.95).*

**Q3-S9.** Differentiate between MCAR, MAR, and MNAR missing data.  
*Answer: MCAR = random, unrelated to data; MAR = depends on observed variables; MNAR = depends on the missing value itself.*

**Q3-S10.** A dataset has features measured in metres (range: 1–10) and kilograms (range: 500–2000). Why should you scale before applying KNN?  
*Answer: KNN uses distance; the kilogram feature dominates due to its larger range. Scaling (min-max or z-score) ensures all features contribute equally.*

---

### Topic 4: Regression

**Q4-S1.** A simple linear regression model is: Price = 50,000 + 3,000 × FloorSpace. If FloorSpace = 80m², what is the predicted price?  
*Answer: Price = 50,000 + 3,000 × 80 = 50,000 + 240,000 = $290,000.*

**Q4-S2.** A model trained on 5 houses has residuals [10000, −5000, 8000, −12000, 4000]. Calculate the RSS.  
*Answer: RSS = 10000² + 5000² + 8000² + 12000² + 4000² = 100M + 25M + 64M + 144M + 16M = 349,000,000.*

**Q4-S3.** What is MSE? How does it differ from MAE?  
*Answer: MSE = (1/N) Σ(yᵢ−ŷᵢ)². MAE = (1/N) Σ|yᵢ−ŷᵢ|. MSE penalises large errors more (squaring), is differentiable, but sensitive to outliers. MAE is more robust to outliers but not differentiable at zero.*

**Q4-S4.** A regression model has R²=0.85. What does this mean?  
*Answer: The model explains 85% of the variance in the target variable; it is substantially better than simply predicting the mean.*

**Q4-S5.** Why is ridge regression not suitable as an embedded feature selection method?  
*Answer: Ridge (L2) penalty shrinks weights toward zero but never sets them exactly to zero — all features remain in the model. Lasso (L1) can zero out weights, achieving true feature selection.*

**Q4-S6.** A dataset has 5 features. How many weights does a linear regression model have? Name them.  
*Answer: 6 weights: w₀ (intercept/bias) and w₁, w₂, w₃, w₄, w₅ (one slope per feature).*

---

### Topic 5: Clustering

**Q5-S1.** What is the key difference between clustering and classification?  
*Answer: Clustering is unsupervised — class labels are unknown; the algorithm discovers groups. Classification is supervised — class labels are known and the model is trained to assign labels to new instances.*

**Q5-S2.** K-Means is described as a "stochastic" algorithm. What does this mean?  
*Answer: K-Means uses random centroid initialisation; different runs may converge to different local optima, giving different clustering results.*

**Q5-S3.** Explain single, complete, and average linkage in agglomerative clustering.  
*Answer: Single = min distance between any two points across clusters; Complete = max distance; Average = mean of all pairwise distances. Complete and average tend to give more balanced dendrograms.*

**Q5-S4.** A silhouette score of 0.65 is computed for a clustering. What does this suggest?  
*Answer: Instances are on average well-matched to their own cluster (score > 0, closer to 1 than −1). The clustering is reasonably good.*

**Q5-S5.** What is a dendrogram, and how is it used to choose the number of clusters?  
*Answer: A dendrogram is a tree diagram from agglomerative clustering. The height at which two clusters merge represents their dissimilarity. Cutting the dendrogram horizontally at a given height yields a clustering; the cut height determines k without re-running the algorithm.*

**Q5-S6.** What two parameters does DBSCAN require? What kinds of points does it identify?  
*Answer: eps (neighbourhood radius) and minPts (minimum points for a core sample). DBSCAN identifies core points (in dense regions), border points, and noise/outliers.*

**Q5-S7.** You want to cluster 10,000 text documents. Which distance measure would you use and why?  
*Answer: Cosine distance — it measures the angle between document vectors (term frequencies), making it appropriate for high-dimensional sparse text data where magnitude differences are not meaningful.*

**Q5-S8.** Give one advantage and one disadvantage of DBSCAN compared to K-Means.  
*Answer: Advantage: no need to specify K; handles arbitrary shapes and identifies outliers. Disadvantage: sensitive to eps and minPts choices; struggles with datasets having varying density.*

**Q5-S9.** Describe the real-world application of clustering used by Netflix. What is being clustered and why?  
*Answer: Netflix clusters both users and content. Users with similar viewing histories are grouped together; content is clustered by genre/style. When a new user action occurs, the system recommends content that similar users have enjoyed. This is unsupervised — there are no explicit "good recommendation" labels.*

**Q5-S10.** What is the difference between agglomerative and divisive hierarchical clustering? Which is more commonly used and why?  
*Answer: Agglomerative (bottom-up) starts with n individual clusters and merges the two most similar at each step until one cluster remains. Divisive (top-down) starts with one cluster and splits recursively. Agglomerative is more commonly used because it is computationally more efficient and easier to implement.*

**Q5-S11.** Explain why K-Means cannot directly handle categorical features. Suggest an alternative approach.  
*Answer: K-Means computes centroids as the mean of feature values. The mean is undefined for categorical data (e.g., the mean of "red" and "blue" is meaningless). Alternative: use K-Medoids (uses actual data points as cluster centres) or encode categoricals numerically, or use hierarchical clustering with Hamming distance.*

**Q5-S12.** Calculate the silhouette score for a point with a(i) = 0.4 and b(i) = 0.9. Interpret the result.  
*Answer: s(i) = (b(i) − a(i)) / max(a(i), b(i)) = (0.9 − 0.4) / max(0.4, 0.9) = 0.5 / 0.9 ≈ 0.556. This is a positive score closer to 1 than −1, indicating the point is reasonably well-placed in its own cluster.*

**Q5-S13.** What is a dendrogram and how do you use it to choose the number of clusters without re-running the algorithm?  
*Answer: A dendrogram is a tree diagram from hierarchical clustering where the y-axis height represents the dissimilarity at each merge. To choose k clusters: draw a horizontal line across the dendrogram — the number of vertical lines it crosses equals k. Choose the cut height to maximise the gap between merge heights (look for a long vertical line before the cut).*

**Q5-S14.** Compare single, complete, and average linkage in agglomerative clustering. Which tends to produce the most balanced dendrograms?  
*Answer: Single linkage = minimum pairwise distance between clusters — can produce long "chains" where points are absorbed one at a time. Complete linkage = maximum pairwise distance — tends to produce compact, balanced clusters. Average linkage = mean of all pairwise distances — a compromise. Complete and average linkage tend to produce more balanced, interpretable dendrograms.*

**Q5-S15.** [Structured] You are given the following four data points in 1D: A=1, B=2, C=6, D=8. Use agglomerative clustering with **single linkage** and Euclidean distance to construct the dendrogram step by step.

*(a) [2 marks] Compute all pairwise distances.*  
*(b) [3 marks] Show each merge step and the resulting clusters.*

*Answer:*  
*(a) Pairwise distances: D(A,B)=1, D(A,C)=5, D(A,D)=7, D(B,C)=4, D(B,D)=6, D(C,D)=2.*  
*(b) Step 1: Merge A and B (distance=1) → clusters {A,B}, {C}, {D}.*  
*Step 2: Minimum distance: D({C},{D})=2. Merge C and D → clusters {A,B}, {C,D}.*  
*Step 3: Distance between {A,B} and {C,D} using single linkage = min(D(A,C), D(A,D), D(B,C), D(B,D)) = min(5,7,4,6) = 4. Merge → one cluster {A,B,C,D} at height 4.*  
*Dendrogram: A&B merge at 1; C&D merge at 2; all merge at 4.*

---

## Summary Marking Guidance

| Mark range | Descriptor |
|---|---|
| Full marks | Correct answer with complete, clear working or justification as required. |
| 75–99% | Correct method/approach with minor arithmetic error or incomplete justification. |
| 50–74% | Partially correct — correct identification of method or key concept with errors in execution. |
| 25–49% | Some relevant knowledge shown but significant gaps or errors. |
| 0–24% | Little relevant content; incorrect or missing. |

**For calculation questions:** Award method marks even if the final numerical answer is wrong, provided the method is correct. Students must show working to receive full marks.

**For definition/explanation questions:** Award marks for accurate, concise responses in own words. Memorised verbatim definitions without understanding do not score method marks on applied sub-parts.
