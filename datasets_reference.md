# Datasets Reference — Data Science Lab
### Complete documentation of every dataset, its columns, and which models/algorithms were applied

---

## 1. Iris Dataset

**File:** `datasets/iris.csv` | **Source:** UCI ML Repository / sklearn | **Records:** 150 | **No missing values**

Classic multi-class classification dataset. Each row is a flower from one of three Iris species.

| # | Column | Type | Description | Range/Values |
|---|--------|------|-------------|--------------|
| 1 | `sepal length (cm)` | float64 | Length of the sepal | 4.3 – 7.9 cm |
| 2 | `sepal width (cm)` | float64 | Width of the sepal | 2.0 – 4.4 cm |
| 3 | `petal length (cm)` | float64 | Length of the petal | 1.0 – 6.9 cm |
| 4 | `petal width (cm)` | float64 | Width of the petal | 0.1 – 2.5 cm |
| 5 | `species` | categorical | Iris species (target) | **setosa** (50), **versicolor** (50), **virginica** (50) |

**Key facts:**
- Balanced: exactly 50 samples per species
- Setosa is linearly separable from the other two
- Petal length and petal width are highly correlated (r ≈ 0.96)
- The most-used dataset in ML education

**Used in Assignments:**

| Assignment | What Was Done | Models/Tech Used |
|---|---|---|
| **1** (Data Wrangling I) | DataFrame inspection, head(), info(), describe(), groupby() | Pandas |
| **3** (Descriptive Statistics) | Mean, median, std, percentiles grouped by species | Pandas groupby, describe |
| **6** (Naive Bayes) | Classification with GaussianNB; confusion matrix | GaussianNB, accuracy, precision, recall, F1 |
| **8** (Data Viz II) | Hist+KDE, violin, pairplot, heatmap, joint plot | Seaborn |
| **9** (Data Viz III) | Histograms, boxplots, outlier detection via IQR | Matplotlib, Seaborn, IQR |
| **12** (Scala Spark) | Demonstrated Spark DataFrame ops using iris | Pandas (simulating Spark) |

---

## 2. Diamonds Dataset

**File:** `datasets/diamonds.csv` | **Source:** seaborn (built-in) | **Records:** 53,940 | **No missing values**

Pricing data for diamonds with physical measurements and quality grades.

| # | Column | Type | Description | Range/Values |
|---|--------|------|-------------|--------------|
| 1 | `carat` | float64 | Weight of the diamond | 0.2 – 5.01 |
| 2 | `cut` | categorical | Cut quality (ordered) | Fair, Good, Very Good, Premium, **Ideal** (most common) |
| 3 | `color` | categorical | Color grade: D(best) to J(worst) | D, E, F, G, H, I, J |
| 4 | `clarity` | categorical | Clarity: IF(best) to I1(worst) | IF, VVS1, VVS2, VS1, VS2, SI1, SI2, I1 |
| 5 | `depth` | float64 | Total depth % = 2 * z / (x + y) | 43.0 – 79.0 |
| 6 | `table` | float64 | Width of top facet relative to widest point (%) | 43.0 – 95.0 |
| 7 | `price` | int64 | Price in USD (**target variable**) | $326 – $18,823 |
| 8 | `x` | float64 | Length in mm | 0 – 10.74 |
| 9 | `y` | float64 | Width in mm | 0 – 58.9 |
| 10 | `z` | float64 | Depth in mm | 0 – 31.8 |

**Key facts:**
- Carat and price have a non-linear positive relationship
- The "cut paradox": better cuts (Ideal) have lower median prices because they're smaller
- x, y, z dimensions are highly correlated with carat
- Right-skewed distributions on price and carat
- The largest dataset used (54K records)

**Used in Assignments:**

| Assignment | What Was Done | Models/Tech Used |
|---|---|---|
| **13** (Case Study) | Full EDA: univariate, bivariate, multivariate, dashboard layout, pivot table, correlation heatmap | Pandas, Seaborn, Matplotlib, IQR |
| **Mini Project** | End-to-end: EDA → ordinal encoding → feature engineering → StandardScaler → Linear Regression → Random Forest → feature importance | LinearRegression, RandomForestRegressor, StandardScaler, R², RMSE, MAE |

---

## 3. Titanic Dataset

**File:** `datasets/titanic.csv` | **Source:** seaborn / Kaggle | **Records:** 891 | **Missing:** age (177), embarked (2), deck (688)

Passenger data from the Titanic disaster. The classic binary classification: predict survival.

| # | Column | Type | Description | Range/Values |
|---|--------|------|-------------|--------------|
| 1 | `survived` | int64 | Survival status (**target**) | 0 = No, 1 = Yes |
| 2 | `pclass` | int64 | Passenger class (proxy for wealth) | 1 = First, 2 = Second, 3 = Third |
| 3 | `sex` | categorical | Gender | male, female |
| 4 | `age` | float64 | Age in years (177 missing) | 0.42 – 80.0 |
| 5 | `sibsp` | int64 | # of siblings/spouses aboard | 0 – 8 |
| 6 | `parch` | int64 | # of parents/children aboard | 0 – 6 |
| 7 | `fare` | float64 | Ticket fare paid | 0 – 512.33 |
| 8 | `embarked` | categorical | Port of embarkation (2 missing) | C = Cherbourg, Q = Queenstown, S = Southampton |
| 9 | `class` | categorical | Same as pclass, as text | First, Second, Third |
| 10 | `who` | categorical | Age/gender category | man, woman, child |
| 11 | `adult_male` | bool | Is adult male? | True, False |
| 12 | `deck` | categorical | Cabin deck letter (688 missing) | A, B, C, D, E, F, G |
| 13 | `embark_town` | categorical | Full embark town name (2 missing) | Cherbourg, Queenstown, Southampton |
| 14 | `alive` | categorical | Same as survived, as text | yes, no |
| 15 | `alone` | bool | Travelling alone? | True, False |

**Key facts:**
- 38.4% survived (imbalanced, but not severely)
- Women and children had much higher survival rates
- First class passengers survived at ~63%, third class at ~24%
- Age has 20% missing — requires imputation
- deck column is 77% missing — typically dropped

**Used in Assignments:**

| Assignment | What Was Done | Models/Tech Used |
|---|---|---|
| **6** (Naive Bayes) | Classification with MultinomialNB; compared against GaussianNB on Iris | MultinomialNB, MinMaxScaler, accuracy, confusion matrix |

---

## 4. Tips Dataset

**File:** `datasets/tips.csv` | **Source:** seaborn (built-in) | **Records:** 244 | **No missing values**

Restaurant tipping data. Used primarily for visualization and EDA.

| # | Column | Type | Description | Range/Values |
|---|--------|------|-------------|--------------|
| 1 | `total_bill` | float64 | Total bill amount in USD | 3.07 – 50.81 |
| 2 | `tip` | float64 | Tip amount in USD | 1.00 – 10.00 |
| 3 | `sex` | categorical | Gender of payer | Female, Male |
| 4 | `smoker` | categorical | Smoker in party? | Yes, No |
| 5 | `day` | categorical | Day of week | Thur, Fri, Sat, Sun |
| 6 | `time` | categorical | Meal time | Lunch, Dinner |
| 7 | `size` | int64 | Party size | 1 – 6 |

**Key facts:**
- Positive linear relationship between total_bill and tip
- Saturday has the most transactions; Friday the least
- Dinner parties are larger and tip more than Lunch
- Males tend to tip slightly more on average
- Small, clean dataset ideal for visualization practice

**Used in Assignments:**

| Assignment | What Was Done | Models/Tech Used |
|---|---|---|
| **7** (Data Viz I) | Line plots, bar charts (vertical/horizontal), histogram, pie chart, scatter (color+size) | Matplotlib (all plot types) |
| **8** (Data Viz II) | Box plot, count plot, bar plot, joint plot, lmplot | Seaborn |

---

## 5. Penguins Dataset

**File:** `datasets/penguins.csv` | **Source:** Palmer Penguins / seaborn | **Records:** 344 | **Missing:** bill_length (2), bill_depth (2), flipper_length (2), body_mass (2), sex (11)

Physical measurements of penguins across 3 species. A modern alternative to Iris.

| # | Column | Type | Description | Range/Values |
|---|--------|------|-------------|--------------|
| 1 | `species` | categorical | Penguin species | Adelie (152), Chinstrap (68), Gentoo (124) |
| 2 | `island` | categorical | Island of observation | Torgersen, Biscoe, Dream |
| 3 | `bill_length_mm` | float64 | Bill length in mm | 32.1 – 59.6 |
| 4 | `bill_depth_mm` | float64 | Bill depth in mm | 13.1 – 21.5 |
| 5 | `flipper_length_mm` | float64 | Flipper length in mm | 172 – 231 |
| 6 | `body_mass_g` | float64 | Body mass in grams | 2700 – 6300 |
| 7 | `sex` | categorical | Sex (11 missing) | Female, Male |

**Key facts:**
- Gentoo has largest body mass and bill length
- Strong sexual dimorphism (males larger in all measurements)
- Few missing values — easy to drop or impute
- Good alternative to Iris with more practical real-world messiness

**Used in Assignments:**

| Assignment | What Was Done | Models/Tech Used |
|---|---|---|
| **8** (Data Viz II) | Violin plot (bill_length by species+sex), correlation heatmap | Seaborn |

---

## 6. Boston Housing Dataset

**File:** loaded from URL | **Source:** CMU StatLib | **Records:** 506 | **No missing values**

**Note:** This dataset is deprecated in newer sklearn versions due to ethical concerns (feature 'B' was calculated using race). Loaded from the original URL in the notebook.

| # | Column | Type | Description |
|---|--------|------|-------------|
| 1 | `CRIM` | float64 | Per capita crime rate by town |
| 2 | `ZN` | float64 | Proportion of residential land zoned for lots > 25,000 sq.ft |
| 3 | `INDUS` | float64 | Proportion of non-retail business acres per town |
| 4 | `CHAS` | int64 | Charles River dummy (1 if bounds river, 0 otherwise) |
| 5 | `NOX` | float64 | Nitric oxide concentration (parts per 10 million) |
| 6 | `RM` | float64 | Average number of rooms per dwelling |
| 7 | `AGE` | float64 | Proportion of owner-occupied units built before 1940 |
| 8 | `DIS` | float64 | Weighted distances to 5 Boston employment centers |
| 9 | `RAD` | int64 | Index of accessibility to radial highways |
| 10 | `TAX` | float64 | Full-value property tax rate per $10,000 |
| 11 | `PTRATIO` | float64 | Pupil-teacher ratio by town |
| 12 | `B` | float64 | 1000(Bk − 0.63)² where Bk is proportion of Black residents |
| 13 | `LSTAT` | float64 | % lower status of the population |
| — | `MEDV` | float64 | **Target:** Median value of owner-occupied homes in $1000s |

**Key facts:**
- Classic regression benchmark
- RM (rooms) and LSTAT (lower status %) are the strongest price predictors
- Features have very different scales — benefits from StandardScaler

**Used in Assignments:**

| Assignment | What Was Done | Models/Tech Used |
|---|---|---|
| **4** (Linear Regression) | Built linear regression model; evaluated with MSE and R²; visualized actual vs predicted | LinearRegression, train_test_split, R² Score, MSE |

---

## 7. Social Network Ads Dataset

**File:** loaded from URL | **Source:** GitHub (`neha-93/datasets`) | **Records:** ~400

Simulated data for predicting whether a user purchases a product based on age and salary.

| # | Column | Type | Description | Range/Values |
|---|--------|------|-------------|--------------|
| 1 | `User ID` | int64 | Unique user identifier (dropped for modeling) | — |
| 2 | `Gender` | categorical | Gender | Male, Female |
| 3 | `Age` | int64 | User's age | ~18 – 60 |
| 4 | `EstimatedSalary` | float64 | Estimated annual salary | ~$15,000 – $150,000 |
| 5 | `Purchased` | int64 | **Target:** Did user purchase? | 0 = No, 1 = Yes |

**Used in Assignments:**

| Assignment | What Was Done | Models/Tech Used |
|---|---|---|
| **5** (Logistic Regression) | Binary classification; LabelEncoder for Gender; decision boundary visualization; confusion matrix | LogisticRegression, LabelEncoder, accuracy, precision, recall, confusion_matrix |

---

## 8. Flights Dataset

**File:** `datasets/flights.csv` | **Source:** seaborn (built-in) | **Records:** 144 | **No missing values**

Monthly airline passenger counts, 1949–1960. Classic time series dataset.

| # | Column | Type | Description | Range/Values |
|---|--------|------|-------------|--------------|
| 1 | `year` | int64 | Year | 1949 – 1960 |
| 2 | `month` | categorical | Month (3-letter abbreviation) | Jan – Dec |
| 3 | `passengers` | int64 | Number of passengers that month | 104 – 622 |

**Key facts:**
- Clear upward trend + seasonal pattern (summer peaks)
- Classic demonstration of time series decomposition
- **Not explicitly used** in any current assignment (available for exercises)

---

## 9. MPG (Auto MPG) Dataset

**File:** `datasets/mpg.csv` | **Source:** seaborn (built-in from UCI) | **Records:** 398 | **Missing:** horsepower (6)

Fuel efficiency of cars from the 1970s and early 1980s.

| # | Column | Type | Description | Range/Values |
|---|--------|------|-------------|--------------|
| 1 | `mpg` | float64 | Miles per gallon (**target**) | 9.0 – 46.6 |
| 2 | `cylinders` | int64 | Number of cylinders | 3, 4, 5, 6, 8 |
| 3 | `displacement` | float64 | Engine displacement (cubic inches) | 68 – 455 |
| 4 | `horsepower` | float64 | Engine horsepower (6 missing) | 46 – 230 |
| 5 | `weight` | int64 | Vehicle weight (lbs) | 1613 – 5140 |
| 6 | `acceleration` | float64 | Acceleration (0-60 mph in seconds) | 8.0 – 24.8 |
| 7 | `model_year` | int64 | Model year (2-digit) | 70 – 82 |
| 8 | `origin` | categorical | Region of origin | usa, europe, japan |
| 9 | `name` | categorical | Vehicle model name | 305 unique names |

**Key facts:**
- Weight is the strongest predictor of MPG (negative correlation)
- Japanese cars tend to have better fuel efficiency
- 6 missing horsepower values — requires imputation
- **Not explicitly used** in any current assignment (available for exercises)

---

## 10. Exercise Dataset

**File:** `datasets/exercise.csv` | **Source:** seaborn (built-in) | **Records:** 90 | **No missing values**

Pulse rate measurements under different diet and activity conditions.

| # | Column | Type | Description | Range/Values |
|---|--------|------|-------------|--------------|
| 1 | `Unnamed: 0` | int64 | Row index (redundant) | 0 – 89 |
| 2 | `id` | int64 | Participant ID | 1 – 30 |
| 3 | `diet` | categorical | Diet type | low fat, no fat |
| 4 | `pulse` | int64 | Pulse rate (beats per min) | 80 – 130 |
| 5 | `time` | categorical | Measurement time | 1 min, 15 min, 30 min |
| 6 | `kind` | categorical | Activity type | rest, walking, running |

**Key facts:**
- Repeated measures design (same person measured at 3 times)
- Pulse increases from rest → walking → running
- **Not explicitly used** in any current assignment (available for exercises)

---

## 11. Anscombe's Quartet

**File:** `datasets/anscombe.csv` | **Source:** seaborn (built-in) | **Records:** 44 (4 datasets × 11 points) | **No missing values**

Four datasets with nearly identical summary statistics but radically different distributions. Created to demonstrate the importance of visualization before statistical analysis.

| # | Column | Type | Description | Range/Values |
|---|--------|------|-------------|--------------|
| 1 | `dataset` | categorical | Which quartet | I, II, III, IV (11 points each) |
| 2 | `x` | float64 | X value | 4 – 19 |
| 3 | `y` | float64 | Y value | 3.74 – 12.74 |

**Key facts:**
- All four have: mean(x) ≈ 9.0, mean(y) ≈ 7.5, corr(x,y) ≈ 0.816
- Linear regression line is identical: y = 3.0 + 0.5x
- But the plots are wildly different — I: linear, II: quadratic, III: linear+outlier, IV: vertical + outlier
- **Not explicitly used** in any current assignment (excellent for teaching why you should always plot data)

---

## 12. Planets Dataset

**File:** `datasets/planets.csv` | **Source:** seaborn (built-in / NASA Exoplanet Archive) | **Records:** 1,035 | **Missing:** orbital_period (43), mass (522), distance (227)

Exoplanets discovered by various methods through 2014.

| # | Column | Type | Description | Range/Values |
|---|--------|------|-------------|--------------|
| 1 | `method` | categorical | Discovery method | 10 methods (Transit, Radial Velocity most common) |
| 2 | `number` | int64 | Number of planets in the system | 1 – 7 |
| 3 | `orbital_period` | float64 | Orbital period in days (43 missing) | 0.09 – 730,000 |
| 4 | `mass` | float64 | Planet mass (Jupiter masses) (522 missing) | 0.00003 – 25.0 |
| 5 | `distance` | float64 | Distance from Earth (parsecs) (227 missing) | 1.35 – 8,500 |
| 6 | `year` | int64 | Year of discovery | 1989 – 2014 |

**Key facts:**
- Transit method discovers the most planets
- Lots of missing values — good for practicing imputation strategies
- Multiple discovery methods — categorical encoding practice
- **Not explicitly used** in any current assignment (available for exercises)

---

## 13. Synthetic Academic Performance Dataset

**Created in:** Assignment 2 | **Records:** 200 | **No missing values** (synthetically generated)

Generated programmatically using NumPy within the notebook to demonstrate data wrangling.

| # | Column | Type | Description |
|---|--------|------|-------------|
| 1 | `student_id` | int64 | Unique student ID (1–200) |
| 2 | `gender` | categorical | Male, Female |
| 3 | `age` | int64 | Student age (18–25) |
| 4 | `study_hours` | float64 | Hours studied per week (normal dist, mean=5, std=2) |
| 5 | `attendance` | float64 | Attendance percentage (50–100%) |
| 6 | `previous_gpa` | float64 | Previous GPA (2.00–4.00) |
| 7 | `final_exam_score` | int64 | Final exam score (30–100) |
| 8 | `parent_education` | categorical | Highest parent education (High School, Bachelor, Master, PhD) |
| 9 | `internet_access` | categorical | Internet access at home? (Yes, No) |

**Key facts:**
- Created with `np.random.seed(42)` for reproducibility
- 4 deliberate outliers added: study_hours (20.0, 0.1), final_exam_score (5, 120)
- Used to demonstrate IQR outlier detection and capping

**Used in Assignments:**

| Assignment | What Was Done | Models/Tech Used |
|---|---|---|
| **2** (Data Wrangling II) | Missing value check, categorical conversion (astype), IQR outlier detection, capping (Winsorization), log transform, Min-Max scaling | Pandas, NumPy, MinMaxScaler, IQR method |

---

## 14. Custom Text Corpus

**Created in:** Assignment 10 | **Type:** Raw text string

A custom paragraph of text about data science topics, written inline in the notebook for text analytics demonstration.

**Content:** ~500 characters about data science, Python, machine learning, deep learning, big data, and statistics.

**Used in Assignments:**

| Assignment | What Was Done | Models/Tech Used |
|---|---|---|
| **10** (Text Analytics) | Tokenization, lowercasing, punctuation/stop-word removal, stemming (Porter), lemmatization (WordNet), word frequency, WordCloud, bigram/trigram, sentiment (TextBlob + VADER) | NLTK, TextBlob, VADER, WordCloud |

---

## 15. Simulated Sales & Customers Tables

**Created in:** Assignment 11 | **Records:** Sales (100), Customers (10)

SQLite tables created within the notebook using NumPy random generation to demonstrate Impala SQL concepts.

**Sales columns:** sale_id, product, category, quantity, price, sale_date, cust_id

**Customers columns:** cust_id, name, region, tier

**Used in Assignments:**

| Assignment | What Was Done | Models/Tech Used |
|---|---|---|
| **11** (Impala) | SELECT, WHERE, ORDER BY, GROUP BY, aggregate functions, JOIN, window functions (ROW_NUMBER), subqueries, export | SQL (via SQLite simulating Impala) |

---

## Quick Reference: Dataset → Assignment → Model Matrix

| Dataset | Assignment(s) | Models / Key Techniques |
|---|---|---|
| **Iris** | 1, 3, 6, 8, 9, 12 | GaussianNB, IQR outlier detection, groupby stats, pairplot, violin plot |
| **Diamonds** | 13, Mini Project | LinearRegression, RandomForestRegressor, StandardScaler, ordinal encoding, feature engineering |
| **Titanic** | 6 | MultinomialNB, MinMaxScaler |
| **Tips** | 7, 8 | Matplotlib (line, bar, histogram, pie, scatter), Seaborn (box, count, bar, joint, lmplot) |
| **Penguins** | 8 | Seaborn (violin, heatmap) |
| **Boston Housing** | 4 | LinearRegression, R², MSE |
| **Social Network Ads** | 5 | LogisticRegression, LabelEncoder, confusion matrix, accuracy, precision, recall |
| **Academic (synthetic)** | 2 | IQR outlier detection, capping/Winsorization, log transform, MinMaxScaler |
| **Text Corpus** | 10 | Tokenization, stemming, lemmatization, WordCloud, TextBlob, VADER |
| **Sales/Customers (SQL)** | 11 | Impala SQL (SELECT, JOIN, GROUP BY, window functions) |
| **Flights** | — | Available for time series exercises |
| **MPG** | — | Available for regression/missing value exercises |
| **Exercise** | — | Available for repeated-measures/ANOVA exercises |
| **Anscombe** | — | Available for visualization importance demonstration |
| **Planets** | — | Available for missing value imputation exercises |

---

## Which Dataset Should I Use For...

| Task | Best Dataset | Why |
|---|---|---|
| Multi-class classification | **Iris** | Balanced, clean, well-studied |
| Binary classification | **Titanic** | Real-world messiness, many EDA opportunities |
| Regression | **Boston Housing** or **Diamonds** | Boston: simple; Diamonds: complex non-linear |
| Data wrangling practice | **Titanic** or **MPG** | Missing values, mixed types, outliers |
| Visualization practice | **Tips** | Small, clean, nice mix of numeric + categorical |
| Text analytics | **Custom text** (or real corpus) | Depends on domain |
| Big data / SQL | **Diamonds** (54K rows) | Largest local dataset |
| End-to-end project | **Diamonds** | Mix of numeric + categorical, large enough, real business problem |
