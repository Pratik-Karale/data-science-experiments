# Methodologies, Algorithms & Models Reference — Data Science Lab
### What, Why, and When for every technique used across all 13 assignments

---

## 1. METHODOLOGIES (Data Processing Techniques)

---

### 1.1 Missing Value Handling

| Method | How It Works | When to Use | When NOT to Use | Assignment |
|---|---|---|---|---|
| **Drop (`dropna`)** | Remove rows with NaN | <5% of data is missing; data is large enough | Missing data is >5% or contains critical patterns | 1, 6 |
| **Mean Imputation** | Fill NaN with column average | Symmetric, bell-shaped distributions | Skewed data (use median instead) | 6 |
| **Median Imputation** | Fill NaN with column median | Skewed distributions; outliers present | — | 6 |
| **Mode Imputation** | Fill NaN with most frequent value | Categorical columns | Continuous numeric columns | 6 |

**Why:** ML models cannot process NaN values. Dropping loses data; imputation preserves sample size while filling gaps.

---

### 1.2 Outlier Detection & Treatment

| Method | How It Works | When to Use | When NOT to Use | Assignment |
|---|---|---|---|---|
| **IQR Method** | Q1−1.5×IQR → Q3+1.5×IQR is "normal"; outside = outlier | Most general-purpose; robust to skewed data | When you know data follows different distribution | 2, 9 |
| **Z-Score Method** | Values with ∣Z∣ > 3 are outliers (Z = (x−μ)/σ) | Data is **normally distributed** | Skewed data — Z-score itself is distorted by outliers | 6 |
| **Capping (Winsorization)** | Replace outliers with boundary values | Preserve all rows but limit extreme influence | True outliers contain meaningful signals | 2 |

**Why detect outliers?** They distort mean, inflate variance, and can pull regression lines away from the true pattern.

**IQR vs Z-Score decision rule:**
- Check skewness first → if skewed, use **IQR**
- If approximately normal, use **Z-Score**
- When in doubt, IQR is safer (non-parametric)

---

### 1.3 Data Transformation

| Method | Formula | Effect | When to Use | Assignment |
|---|---|---|---|---|
| **Log Transform** | x' = log(x + 1) | Compresses large values; reduces right-skew | Skewed data (e.g., income, prices) | 2 |
| **Square Root** | x' = √x | Milder than log; handles zeros (no +1 needed) | Count data with zeros | — |
| **Box-Cox** | x' = (x^λ − 1)/λ | Automatically finds best λ | Need optimal normalization; data > 0 | — |

**Why transform?** Many models assume features are normally distributed (Linear Regression, Gaussian Naive Bayes). Skewed data violates this assumption.

---

### 1.4 Feature Scaling / Normalization

| Method | Formula | Output Range | When to Use | Assignment |
|---|---|---|---|---|
| **Min-Max Scaling** | x' = (x−min)/(max−min) | [0, 1] | Neural networks; algorithms sensitive to magnitude (KNN, K-Means); when bounds matter | 2, 6 |
| **Standardization (Z-score)** | x' = (x−μ)/σ | mean=0, std=1 | Linear/Logistic Regression, SVM, PCA; algorithms assuming normal distribution | 6 |

**Why scale?** Features with large values (e.g., Salary in thousands) dominate features with small values (e.g., Age in tens) in distance-based algorithms. Scaling puts all features on equal footing.

**Min-Max vs Standardization decision rule:**
- Use **Standardization** when data is roughly normal and algorithm assumes it
- Use **Min-Max** when you need bounded output [0,1] or data has known fixed bounds
- Standardization is more robust to outliers (doesn't compress to a fixed range)

---

### 1.5 Encoding Categorical Variables

| Method | How It Works | When to Use | When NOT | Assignment |
|---|---|---|---|---|
| **Label Encoding** | Each category → integer (Male=0, Female=1) | Binary categories; tree-based models (RF, XGBoost) | Nominal with >2 categories in linear models (creates false ordinality) | 5 |
| **One-Hot Encoding** | Each category → binary column (n categories = n columns) | Nominal categories in linear/logistic regression | High cardinality (>20 categories — creates too many columns) | 6 |
| **Ordinal Encoding** | Ordered categories → ordered integers (Low=0, Med=1, High=2) | Categories have natural order | Nominal (unordered) categories | — |

**Why encode?** ML models require numerical input. The encoding choice affects whether the model sees categories as ordered or independent.

---

### 1.6 Handling Imbalanced Data

| Method | How It Works | When to Use | Assignment |
|---|---|---|---|
| **SMOTE (Synthetic Minority Oversampling)** | Creates synthetic minority samples by interpolating between neighbors | Minority class is <30% | 6 |
| **Random Undersampling** | Randomly removes majority samples | Large dataset; minority class has enough samples | 6 |
| **Class Weighting** | Assigns higher penalty for misclassifying minority | When you don't want to modify the data itself | — |

**Why handle imbalance?** A model that predicts "majority class always" gets high accuracy but is useless. Rebalancing forces the model to actually learn minority patterns.

---

## 2. ALGORITHMS (Supervised Learning Models)

---

### 2.1 Linear Regression

| Aspect | Detail |
|---|---|
| **Type** | Regression (predicts a number) |
| **Output** | Continuous value (price, score, temperature) |
| **Core Idea** | Fit a line/hyperplane minimizing sum of squared errors: y = b₀ + b₁x₁ + ... + bₙxₙ |
| **Training** | Ordinary Least Squares — finds coefficients that minimize MSE |
| **Assignment** | 4 — House price prediction on Boston Housing |

**When to choose Linear Regression:**
- Target is a **continuous number**
- Relationship between features and target is **roughly linear**
- You need **interpretability** — each coefficient tells you exactly how much the target changes per unit of that feature
- You want a simple, fast baseline before trying complex models

**When NOT to choose it:**
- Target is categorical (use Logistic Regression)
- Strong non-linear relationships (use Decision Trees, Polynomial Regression)
- Highly correlated features cause multicollinearity (use Ridge/Lasso)
- Outliers are present and can't be removed (use Huber/RANSAC)

**Key outputs to check:**
- **R² Score:** How much variance is explained. >0.7 is good, <0.5 is weak
- **Coefficients:** Sign tells direction (±), magnitude tells importance
- **Intercept:** Baseline prediction when all features = 0

---

### 2.2 Logistic Regression

| Aspect | Detail |
|---|---|
| **Type** | Binary Classification (yes/no, 0/1) |
| **Output** | Probability [0, 1]; threshold at 0.5 for class decision |
| **Core Idea** | Applies sigmoid function σ(z) = 1/(1+e⁻ᶻ) to linear output, converting any real number to a probability |
| **Training** | Maximum Likelihood Estimation — finds coefficients that maximize probability of observed classes |
| **Assignment** | 5 — Purchase prediction on Social Network Ads |

**When to choose Logistic Regression:**
- Binary classification (two classes only — extend to multinomial for >2)
- You need **probability estimates**, not just class labels
- Features are roughly linearly separable
- You want **interpretable feature importance** (odds ratios)

**When NOT to choose it:**
- Non-linear decision boundaries needed (use SVM with RBF kernel, Decision Trees)
- Multi-class with many categories (consider Softmax/Neural Networks)
- Very high-dimensional sparse data (Naive Bayes or linear SVM may be better)

**Key metrics:**
| Metric | Question Answered | Use When |
|---|---|---|
| **Accuracy** | % correct overall | Classes are balanced |
| **Precision** | Of predicted positives, how many are real? | False positives are costly (spam detection) |
| **Recall** | Of actual positives, how many were found? | False negatives are costly (disease screening) |

---

### 2.3 Naive Bayes

| Aspect | Detail |
|---|---|
| **Type** | Classification (binary or multi-class) |
| **Output** | Class probabilities |
| **Core Idea** | Bayes' Theorem: P(class∣features) ∝ P(class) × P(feature₁∣class) × P(feature₂∣class) × ... Assumes features are **independent** ("naive") |
| **Training** | Learn P(class) from frequencies; learn P(feature∣class) distribution per class |
| **Assignment** | 6 |

**Variants and when to use each:**

| Variant | Assumes Feature Distribution | When to Use |
|---|---|---|
| **GaussianNB** | Features follow normal distribution per class | Continuous numeric features (height, weight, scores) |
| **MultinomialNB** | Features are counts/proportions | Text classification with word counts; discrete features |
| **BernoulliNB** | Features are binary (present/absent) | Binary features; text with word occurrence (not count) |
| **ComplementNB** | Adapted from Multinomial | Imbalanced text datasets |

**When to choose Naive Bayes:**
- **Text classification** (spam detection, sentiment, topic) — it's the classic choice
- Very fast training and prediction (good for large datasets)
- Works well with **high-dimensional** data (many features)
- Small training set — needs less data than logistic regression
- Features are reasonably independent (or you accept the approximation)

**When NOT to choose it:**
- Features are strongly correlated (violates "naive" assumption)
- You need calibrated probabilities (Naive Bayes probabilities are often overconfident)
- Complex feature interactions matter (use tree-based models or neural nets)

**Why "Naive"?** The independence assumption is almost always false in reality, but the model still works surprisingly well in practice — especially for text.

---

### 2.4 Model Selection Decision Tree (Quick Reference)

```
What is your target variable?
│
├─ Continuous number
│   └─ LINEAR REGRESSION (Assignment 4)
│       - First baseline
│       - Need interpretability → Done
│       - Non-linear → Try Decision Trees, Random Forest
│
└─ Category / Class
    │
    ├─ Binary (Yes/No)
    │   ├─ Need probabilities + interpretability → LOGISTIC REGRESSION (Assignment 5)
    │   ├─ Need just class + data is linearly separable → Logistic or Linear SVM
    │   └─ Non-linear boundary → SVM with RBF, Decision Trees
    │
    └─ Multi-class
        ├─ Text data (spam, sentiment, topic) → NAIVE BAYES (Assignment 6)
        ├─ Numeric, small data → Logistic (multinomial) or Naive Bayes
        └─ Numeric, large data → Random Forest, XGBoost, Neural Networks
```

---

## 3. MODEL EVALUATION FRAMEWORK

---

### 3.1 Train-Test Split

**Why split?** If you test on the same data you trained on, the model can memorize (overfit) and you'll get falsely high scores. Testing on unseen data reveals true generalization.

**Standard split:** 80% train, 20% test (or 70/30). Use `random_state` for reproducibility.

### 3.2 Metrics Selection Guide

| Task | Best Metric | Why | Assignment |
|---|---|---|---|
| **Regression** | R² Score | Intuitive: "% of variance explained" | 4 |
| **Regression** | MSE / RMSE | In original units; penalizes large errors | 4 |
| **Classification (balanced)** | Accuracy | Simple, intuitive | 5 |
| **Classification (imbalanced)** | Precision + Recall + F1 | Accuracy is misleading when classes are uneven | 5 |
| **Classification (ranking)** | ROC-AUC | Measures ranking quality, not threshold-dependent | — |

### 3.3 Confusion Matrix Interpretation

```
                    Predicted NO    Predicted YES
Actual NO           TN              FP  (Type I error)
Actual YES          FN  (Type II)   TP
```

- **Type I Error (FP):** False alarm — predicting "buy" when customer won't
- **Type II Error (FN):** Miss — failing to predict "buy" when customer would have

Which error is worse depends on the application:
- Disease screening: **FN is worse** (missing a sick patient) → prioritize Recall
- Spam detection: **FP is worse** (blocking a real email) → prioritize Precision

---

## 4. DATA VISUALIZATION — PLOT SELECTION GUIDE

---

### 4.1 Which Plot for What Question?

| Question You're Asking | Best Plot | Assignment |
|---|---|---|
| "What's the distribution of one variable?" | **Histogram** or **KDE plot** | 7, 8, 9 |
| "How does a numeric variable vary across categories?" | **Box plot** or **Violin plot** | 8, 9 |
| "How many in each category?" | **Count plot** or **Bar chart** | 7, 8 |
| "What's the relationship between two numeric variables?" | **Scatter plot** | 4, 7 |
| "How do all pairs of features relate?" | **Pair plot** | 8, 9 |
| "Which features are correlated?" | **Heatmap** of correlation matrix | 8 |
| "What's the trend over time?" | **Line plot** | 7 |
| "What are the proportions?" | **Pie chart** (≤5 slices) or **Bar chart** (>5) | 7 |
| "Are there outliers?" | **Box plot** | 8, 9 |
| "How does a classifier separate classes?" | **Contour plot** (decision boundary) | 5 |

### 4.2 Box Plot vs Violin Plot

| Aspect | Box Plot | Violin Plot |
|---|---|---|
| Shows | Q1, median, Q3, whiskers, outlier dots | All of box plot + full density shape on both sides |
| Best for | Quick outlier check; comparing many categories | Seeing bimodal/multimodal distributions |
| Limitation | Hides distribution shape (e.g., two peaks look same as one) | Harder to read for many categories |
| Assignment | 8, 9 | 8 |

### 4.3 Matplotlib vs Seaborn

| Aspect | Matplotlib | Seaborn |
|---|---|---|
| Level | Low-level (full control) | High-level (built on matplotlib) |
| Default style | Basic | Polished, modern |
| Statistical plots | Build manually | Built-in (boxplot, pairplot, lmplot) |
| Integration with Pandas | Requires numpy arrays | Accepts DataFrame column names directly |
| Best for | Custom, publication-ready figures | Rapid EDA, statistical visualization |
| Assignment | 7 | 8, 9 |

---

## 5. TEXT ANALYTICS PIPELINE (Assignment 10)

---

### 5.1 Processing Pipeline Steps (in order)

```
Raw Text
  │
  ├─ 1. Lowercasing ─── "Hello World" → "hello world"
  │   Why: "Hello" and "hello" should be the same token
  │
  ├─ 2. Remove Punctuation ─── "hello!" → "hello"
  │   Why: Punctuation adds noise for most analyses
  │
  ├─ 3. Tokenization ─── "hello world" → ["hello", "world"]
  │   Why: Convert string → list of meaningful units
  │
  ├─ 4. Remove Stop Words ─── ["the", "cat", "is"] → ["cat"]
  │   Why: Common words carry little meaning; they dominate frequency counts
  │
  ├─ 5. Stemming OR Lemmatization
  │   Stemming:  "running" → "run", "studies" → "studi" (fast, crude)
  │   Lemmatization: "running" → "run", "better" → "good" (slow, accurate)
  │   Why: Reduce inflectional forms to base — treat "run/runs/running" as one word
  │
  └─ 6. Vectorization ─── words → numbers for ML models
      Options: CountVectorizer (bag-of-words), TF-IDF (weight by rarity)
```

### 5.2 Stemming vs Lemmatization

| Aspect | Stemming (Porter) | Lemmatization (WordNet) |
|---|---|---|
| Speed | Fast | Slower |
| Accuracy | Crude — chops off prefixes/suffixes | Accurate — uses dictionary |
| Output | May not be a real word | Always a real word |
| Example | "studies" → "studi" | "studies" → "study" |
| Use when | Speed matters; large corpus | Accuracy matters; smaller corpus |

### 5.3 Sentiment Analysis Methods

| Method | How It Works | Best For | Assignment |
|---|---|---|---|
| **VADER** | Rule-based; uses lexicon of scored words + grammar rules | Social media, short informal text | 10 |
| **TextBlob** | Pre-built model; returns polarity (-1 to 1) and subjectivity (0 to 1) | Quick general-purpose analysis | 10 |
| **ML-based** | Train Naive Bayes / Logistic Regression on labeled data | Domain-specific sentiment (e.g., product reviews) | — |

---

## 6. BIG DATA TOOLS (Assignments 11 & 12)

---

### 6.1 When to Use Impala vs Spark

| Aspect | Apache Impala | Apache Spark |
|---|---|---|
| **Paradigm** | SQL queries | General-purpose distributed computing |
| **Best for** | Interactive SQL on Hadoop data (ad-hoc queries) | Complex ETL pipelines, ML, streaming |
| **Speed** | Very fast for SQL (uses MPP engine, in-memory) | Fast but has scheduling overhead |
| **Language** | SQL only | Scala, Python, Java, R, SQL |
| **Assignment** | 11 | 12 |

**When to choose Impala:** You have structured data on HDFS and need fast, interactive SQL queries. Users know SQL. Low-latency BI/reporting.

**When to choose Spark:** You need data transformations beyond SQL (ML, graph processing, streaming). Multi-step pipelines. Unstructured data processing.

### 6.2 RDD vs DataFrame (Spark)

| Aspect | RDD | DataFrame |
|---|---|---|
| **Abstraction** | Low-level distributed collection | High-level table with schema |
| **Optimization** | No automatic optimization | Catalyst optimizer + Tungsten execution |
| **Type safety** | Type-safe (Scala) | Schema-enforced, not type-safe |
| **Performance** | Slower | Faster (optimized) |
| **When to use** | Unstructured data, custom partitioning, low-level control | Structured data, SQL queries, standard ETL |

---

## 7. QUICK DECISION CHEAT SHEET

| Situation | Your Choice |
|---|---|
| Predict a numeric value (price, temperature) | **Linear Regression** |
| Classify into 2 categories with probability | **Logistic Regression** |
| Classify text (spam, sentiment, topic) | **Naive Bayes (MultinomialNB)** |
| Detect outliers in skewed data | **IQR Method** |
| Detect outliers in normal data | **Z-Score Method** |
| Reduce right-skew in features | **Log Transform** |
| Scale features for distance-based models | **Min-Max Scaling** or **Standardization** |
| Scale features for linear/logistic regression | **Standardization** |
| Encode binary categorical (Male/Female) | **Label Encoding** |
| Encode multi-category nominal (Red/Blue/Green) | **One-Hot Encoding** |
| See distribution of one variable | **Histogram** |
| Compare distributions across groups | **Box Plot** or **Violin Plot** |
| See all pairwise relationships | **Pair Plot** |
| Visualize correlation matrix | **Heatmap** |
| Quick EDA with nice defaults | **Seaborn** |
| Custom publication-quality figures | **Matplotlib** |
| Fast ad-hoc SQL on Hadoop | **Impala** |
| Complex distributed data pipeline | **Spark (Scala/Python)** |
