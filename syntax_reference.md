# Python Syntax Reference — Data Science Lab
### Every function, method & parameter used across all 13 assignments

---

## 1. NUMPY (`import numpy as np`)

### Array Creation

| Syntax | Why Use It | Key Parameters | Assignment |
|---|---|---|---|
| `np.random.seed(n)` | Fix random state so results are **reproducible** across runs | `n` — any integer | 2 |
| `np.random.choice(arr, size)` | Randomly pick elements from a list — used for creating synthetic categorical data | `arr` — source array; `size` — how many to pick | 2 |
| `np.random.randint(low, high, size)` | Random integers in a range — used for age, scores, etc. | `low`, `high` — range; `size` — output shape | 2 |
| `np.random.normal(mean, std, size)` | Samples from a **bell curve** — used for realistic continuous data (e.g., study hours) | `mean` — center; `std` — spread; `size` — count | 2 |
| `np.random.uniform(low, high, size)` | Uniform random floats — equal probability across range | `low`, `high` — range; `size` — count | 2 |
| `np.where(cond, x, y)` | Vectorized **if-else** — replace values where condition is true | `cond` — boolean; `x` — if True; `y` — if False | 2 |
| `np.hstack([arr1, arr2])` | **Stack arrays horizontally** (side-by-side) — used to combine data columns | list of arrays | 4 |
| `np.meshgrid(x, y)` | Create a **grid of coordinates** from 1D arrays — used for plotting decision boundaries | `x`, `y` — 1D coordinate arrays | 5 |
| `np.arange(start, stop, step)` | Evenly spaced values — like `range()` but returns array | `start`, `stop`, `step` — step size | 5 |
| `np.c_[arr1, arr2]` | **Column-stack** 1D arrays into 2D — used to combine features for prediction | arrays to stack | 5 |
| `.ravel()` | Flatten any array to **1D** — used before meshgrid prediction | — | 5 |
| `.reshape(shape)` | Change array dimensions without changing data | new shape tuple | 1, 5 |
| `.T` / `np.transpose()` | **Transpose** a matrix (swap rows ↔ cols) | — | 1 |
| `np.dot(a, b)` / `@` | **Matrix multiplication** | — | 1 |
| `np.mean()`, `np.median()`, `np.std()`, `np.min()`, `np.max()` | Statistical ops on arrays | `axis` — 0=cols, 1=rows | 1 |

### Array Indexing & Slicing

| Syntax | Meaning | Assignment |
|---|---|---|
| `arr[0]` | First element | 1 |
| `arr[-1]` | Last element | 1 |
| `arr[2:5]` | Elements at indices 2,3,4 | 1 |
| `arr[:, 0]` | All rows, first column | 1 |
| `arr[arr > 5]` | Boolean indexing — all values > 5 | 1 |

---

## 2. PANDAS (`import pandas as pd`)

### Input / Output

| Syntax | Why Use It | Key Parameters | Assignment |
|---|---|---|---|
| `pd.read_csv(url_or_path)` | Load CSV into a DataFrame | `sep` — delimiter (default `,`; use `r"\s+"` for whitespace); `skiprows` — skip header lines; `header` — which row is column names | 2, 3, 4, 5 |
| `df.to_csv(path, index=False)` | Save DataFrame to CSV | `index=False` — don't write row numbers | datasets |

### DataFrame Creation

| Syntax | Why | Assignment |
|---|---|---|
| `pd.DataFrame(dict)` | Create table from a Python dict — keys = column names, values = lists | 2, 3 |
| `pd.DataFrame(array, columns=names)` | Wrap a numpy array with column labels | 3, 4, 9 |
| `pd.Categorical(list)` | Convert to **category dtype** — saves memory for repeated strings | 3 |

### Inspection

| Syntax | What It Returns | Why Use It | Assignment |
|---|---|---|---|
| `df.head(n)` | First n rows (default 5) | Quick look at data | 2, 3, 4, 5, 9 |
| `df.tail(n)` | Last n rows | See end of data | 1 |
| `df.info()` | Column names, dtypes, non-null counts | Check for missing values & types | 1, 9 |
| `df.describe()` | count, mean, std, min, 25%, 50%, 75%, max | Summary stats for numeric columns | 2, 3, 9 |
| `df.describe(percentiles=[...])` | Same + custom percentiles | Custom distribution view | 3 |
| `df.shape` | (rows, cols) tuple | Check dimensions | 2, 3, 4, 5 |
| `df.columns` | Column index | See all column names | 3, 4 |
| `df.columns.tolist()` | Column names as list | Easier to read/print | 3 |
| `df.dtypes` | Data type of each column | Identify numeric vs categorical | 2, 9 |
| `df.isnull().sum()` | Count of nulls per column | Detect missing values | 2 |
| `df['col'].value_counts()` | Frequency of each unique value | Distribution of categoricals | 3 |
| `df['col'].unique()` | Array of unique values | List categories | 3 |
| `df['col'].nunique()` | Count of unique values | Cardinality check | — |

### Selection & Filtering

| Syntax | What It Does | Assignment |
|---|---|---|
| `df['col']` | Select one column (returns Series) | All |
| `df[['col1', 'col2']]` | Select multiple columns | 2, 3, 4, 5 |
| `df.loc[row_label, col_label]` | Select by **label/index name** | 1 |
| `df.iloc[row_pos, col_pos]` | Select by **integer position** | 1 |
| `df[df['col'] > value]` | Filter rows by condition | 1, 3 |
| `df[df['col'].isin([...])]` | Filter rows where col is in a list | 1 |
| `df['col'].astype('category')` | Convert column to category dtype | 2 |
| `df.drop('col', axis=1)` | Remove a column | 5 |
| `df.dropna()` | Drop rows with any NaN | 1, 6 |
| `df.fillna(value)` | Replace NaN with a value (mean, median, etc.) | 1, 6 |
| `df.copy()` | Deep copy — don't modify original | 2 |

### Aggregation & Grouping

| Syntax | Why Use It | Key Parameters | Assignment |
|---|---|---|---|
| `df.groupby('col')` | Split data into groups by a categorical column | column name to group by | 3 |
| `grouped.mean()` | Mean of each numeric column per group | — | 3 |
| `grouped.median()` | Median per group | — | 3 |
| `grouped.std()` | Standard deviation per group | — | 3 |
| `grouped.size()` | Row count per group | — | 3 |
| `grouped.agg(['mean', 'std'])` | Multiple aggregations at once | list of function names | 3 |
| `grouped.groups.keys()` | Get group names | — | 3 |
| `df['col'].quantile(q)` | Value at q-th percentile (0.25=Q1, 0.75=Q3) | `q` — 0 to 1 | 2, 9 |

### Sorting & Merging

| Syntax | Why | Assignment |
|---|---|---|
| `df.sort_values('col')` | Sort rows by column value | 1 |
| `df.sort_values('col', ascending=False)` | Descending sort | 1 |
| `pd.merge(left, right, on='col')` | SQL-style join on a common column | 2 |
| `pd.concat([df1, df2])` | Stack DataFrames vertically (union) | 2 |
| `df.drop_duplicates()` | Remove duplicate rows | 2 |
| `pd.pivot_table(df, values, index, columns, aggfunc)` | Reshape data — summarize by categories | 2 |
| `pd.crosstab(row, col)` | Frequency table of two categorical variables | 2 |

### Apply & Map

| Syntax | Why | Assignment |
|---|---|---|
| `df['col'].apply(func)` | Run a custom function on every element of a Series | 2 |
| `df.apply(func, axis=0)` | Run function on each column | 2 |
| `df.apply(func, axis=1)` | Run function on each row | 2 |
| `df['col'].map(dict)` | Replace values using a lookup dictionary | 2, 3 |

---

## 3. SCIKIT-LEARN (`from sklearn...`)

### Datasets

| Syntax | What It Loads | Why | Assignment |
|---|---|---|---|
| `load_iris()` | Iris dataset (150 flowers × 4 features) | Returns a Bunch with `.data`, `.target`, `.feature_names`, `.target_names` | 3, 9 |

### Preprocessing

| Syntax | What It Does | Why Use It | Key Parameters | Assignment |
|---|---|---|---|---|
| `LabelEncoder()` | Converts strings → integers (e.g., Male=1, Female=0) | ML models need numbers, not text | — | 5 |
| `.fit_transform(col)` | Fit encoder + transform in one call | Learns mapping then applies it | — | 5 |
| `MinMaxScaler()` | Scales values to [0, 1] range | Normalizes features so none dominates by scale | `feature_range` — default (0,1) | 2, 6 |
| `scaler.fit_transform(data)` | Fit scaler + transform | Computes min/max then scales | — | 2 |
| `StandardScaler()` | Scales to mean=0, std=1 (Z-score) | For algorithms assuming normal distribution | — | 6 |
| `train_test_split(X, y, test_size, random_state)` | Split data into train/test sets (e.g., 80/20) | Evaluate on **unseen data** to measure generalization | `test_size` — fraction for test; `random_state` — seed for reproducibility | 4, 5, 6 |

### Models

| Syntax | Model Type | Why Choose It | Key Parameters | Assignment |
|---|---|---|---|---|
| `LinearRegression()` | Linear regression | Predicts a **continuous number** (price, score) | — | 4 |
| `.fit(X, y)` | Train the model | Learn coefficients from training data | — | 4, 5 |
| `.predict(X)` | Make predictions | Apply learned model to new data | — | 4, 5 |
| `.coef_` | Feature coefficients | Shows each feature's **impact** on target (sign=direction, magnitude=importance) | — | 4 |
| `.intercept_` | Bias term (b0) | Base value when all features = 0 | — | 4, 5 |
| `LogisticRegression(random_state)` | Logistic regression | Classifies into **two categories** (yes/no, buy/not buy) | `random_state` — reproducible | 5 |
| `GaussianNB()` | Naive Bayes (Gaussian) | Fast classifier for **continuous features** — assumes normal distribution per class | — | 6 |
| `MultinomialNB()` | Naive Bayes (Multinomial) | For **count-based features** (text word counts) | — | 6 |

### Metrics

| Syntax | What It Measures | Formula / Range | Assignment |
|---|---|---|---|
| `mean_squared_error(y_true, y_pred)` | Average squared error | Lower = better; 0 is perfect | 4 |
| `r2_score(y_true, y_pred)` | % of variance explained | 0 to 1; closer to 1 = better fit | 4 |
| `confusion_matrix(y_true, y_pred)` | Matrix: TN, FP, FN, TP | Raw counts of correct/incorrect predictions | 5 |
| `accuracy_score(y_true, y_pred)` | (TP+TN) / Total | 0 to 1; % correctly classified | 5 |
| `precision_score(y_true, y_pred)` | TP / (TP + FP) | How many **predicted positives** are actually positive | 5 |
| `recall_score(y_true, y_pred)` | TP / (TP + FN) | How many **actual positives** were found | 5 |

---

## 4. MATPLOTLIB (`import matplotlib.pyplot as plt`)

### Figure Setup

| Syntax | Why | Assignment |
|---|---|---|
| `plt.figure(figsize=(w, h))` | Create figure with custom size (inches) | 4, 9 |
| `plt.subplots(rows, cols, figsize)` | Create grid of subplots — returns `(fig, axes)` array | 2, 5 |
| `plt.tight_layout()` | Auto-adjust spacing so labels don't overlap | 2, 4, 5 |
| `plt.show()` | Render the plot | All |

### Plot Types

| Syntax | What It Draws | Best For | Key Parameters | Assignment |
|---|---|---|---|---|
| `plt.plot(x, y)` | Line plot | Time trends, functions | `color`, `linestyle`, `linewidth`, `label`, `marker` | 7 |
| `plt.bar(x, height)` | Vertical bar chart | Comparing categories | `color`, `width`, `label` | 7 |
| `plt.barh(x, width)` | Horizontal bar chart | Categories with long names | same as bar | 7 |
| `plt.hist(data, bins)` | Histogram | Showing **distribution** of a variable | `bins` — number of bars; `color`; `edgecolor` | 2, 7, 9 |
| `plt.scatter(x, y)` | Scatter plot | Relationship between **two numeric** variables | `c` — color; `s` — size; `alpha` — transparency; `edgecolor` | 4, 5, 7 |
| `plt.pie(sizes, labels, explode)` | Pie chart | Proportions of a whole | `labels`; `autopct` — show %; `explode` — pop out slices | 7 |
| `plt.contourf(X, Y, Z, alpha, cmap)` | Filled contour plot | **Decision boundaries** of classifiers | `alpha` — transparency; `cmap` — color map | 5 |

### Customization

| Syntax | What It Does | Assignment |
|---|---|---|
| `plt.title('text')` | Add title | 4, 7, 9 |
| `plt.xlabel('text')`, `plt.ylabel('text')` | Label axes | 4, 5, 7 |
| `plt.legend()` | Show legend from labeled data | 4, 5, 7 |
| `plt.grid(True, alpha)` | Add gridlines | 4 |
| `plt.xlim(min, max)`, `plt.ylim(min, max)` | Set axis range | — |
| `plt.xticks(rotation=45)` | Rotate tick labels for readability | — |
| `plt.annotate('text', xy=point)` | Add text annotations at specific points | 7 |
| `plt.savefig('file.png', dpi=300)` | Save figure to disk | 7 |

### Styling

| Syntax | Why | Assignment |
|---|---|---|
| `plt.style.use('ggplot')` / `'seaborn'` / `'fivethirtyeight'` | Apply a pre-built style theme | — |
| `ListedColormap(['color1', 'color2'])` | Custom color map from a list of colors | 5 |

---

## 5. SEABORN (`import seaborn as sns`)

Each seaborn function takes a `data` parameter (DataFrame) and column names as strings.

| Syntax | What It Draws | Best For | Key Parameters | Assignment |
|---|---|---|---|---|
| `sns.load_dataset('name')` | Load built-in dataset | Quick access to iris, titanic, tips, etc. | dataset name as string | datasets, 8 |
| `sns.histplot(data, x, kde)` | Histogram + optional KDE curve | Distribution of a variable | `kde=True` — overlay smooth density curve | 8, 9 |
| `sns.kdeplot(data, x)` | KDE (smooth density curve) alone | Distribution shape without binning | — | 8 |
| `sns.boxplot(data, x, y)` | Box plot: Q1, median, Q3, whiskers, outliers | **Compare distributions** across categories + detect outliers | `x` — category; `y` — numeric (or just `y` for single) | 8, 9 |
| `sns.violinplot(data, x, y)` | Violin plot: box + density on both sides | Richer distribution view than boxplot — shows **shape** | `x` — category; `y` — numeric | 8 |
| `sns.countplot(data, x)` | Bar chart of value counts | Frequency of categories | `hue` — split by another category | 8 |
| `sns.barplot(data, x, y, hue)` | Bar chart with **mean + error bar** | Compare average of Y across categories | `hue` — subgrouping | 8 |
| `sns.pairplot(data, hue)` | Grid of scatter plots for **all feature pairs** | See pairwise relationships at once; `hue` color-codes by category | `hue` — categorical column for color | 8, 9 |
| `sns.heatmap(data, annot, cmap)` | Color-grid of values | **Correlation matrix** visualization | `annot=True` — show numbers in cells; `cmap` — color scheme | 8 |
| `sns.jointplot(data, x, y, kind)` | Scatter + marginal histograms | See 2D relationship + 1D distributions simultaneously | `kind` — 'scatter', 'kde', 'hex', 'reg' | 8 |
| `sns.lmplot(data, x, y, hue)` | Scatter + regression line | Visualizing linear relationship with **fitted line** | `hue` — color by group; `col` — facet by group | 8 |
| `df.hist(bins, figsize)` | Pandas wrapper for multiple histograms | Quick grid of histograms for all numeric cols | `bins` — bar count; `figsize` | 9 |

---

## 6. NLP / TEXT ANALYTICS LIBRARIES

### NLTK (`import nltk`)

| Syntax | What It Does | Why | Assignment |
|---|---|---|---|
| `nltk.download('punkt')` | Download tokenizer model | Required before first use | 10 |
| `nltk.download('stopwords')` | Download stopwords list | Required for stop word removal | 10 |
| `nltk.download('wordnet')` | Download WordNet lexicon | Required for lemmatization | 10 |
| `nltk.word_tokenize(text)` | Split text into **word** tokens | First step of text preprocessing | 10 |
| `nltk.sent_tokenize(text)` | Split text into **sentence** tokens | Sentence-level analysis | 10 |
| `nltk.corpus.stopwords.words('english')` | Get English stop words list | Filter out "the", "is", "and", etc. | 10 |
| `PorterStemmer().stem(word)` | Reduce word to root form (crude) | "running" → "run", "studies" → "studi" | 10 |
| `WordNetLemmatizer().lemmatize(word)` | Reduce word to dictionary form (smarter) | "running" → "run", "better" → "good" | 10 |
| `nltk.FreqDist(word_list)` | Count frequency of each word | Find most common terms | 10 |
| `nltk.bigrams(word_list)` | Generate (word1, word2) pairs | N-gram analysis | 10 |
| `nltk.trigrams(word_list)` | Generate (word1, word2, word3) triples | N-gram analysis | 10 |

### TextBlob (`from textblob import TextBlob`)

| Syntax | What It Does | Why | Assignment |
|---|---|---|---|
| `TextBlob(text)` | Create TextBlob object | Entry point for all TextBlob operations | 10 |
| `.sentiment.polarity` | Polarity score: -1 (negative) to +1 (positive) | Quick sentiment analysis | 10 |
| `.sentiment.subjectivity` | Subjectivity: 0 (objective) to 1 (subjective) | Fact vs opinion | 10 |
| `.words` | Word tokens | Quick tokenization | 10 |
| `.sentences` | Sentence tokens | Sentence splitting | 10 |

### VADER Sentiment (`from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer`)

| Syntax | What It Does | Why | Assignment |
|---|---|---|---|
| `SentimentIntensityAnalyzer()` | Create VADER analyzer | Specifically tuned for social media / short text | 10 |
| `.polarity_scores(text)` | Returns `{'neg', 'neu', 'pos', 'compound'}` | Compound: -1 to +1 overall sentiment | 10 |

### WordCloud (`from wordcloud import WordCloud`)

| Syntax | What It Does | Why | Key Parameters | Assignment |
|---|---|---|---|---|
| `WordCloud()` | Create a word cloud generator | Visual word frequency representation | `width`, `height` — size; `background_color`; `stopwords` — set of words to exclude | 10 |
| `.generate(text)` | Build word cloud from raw text | Frequency-based sizing | — | 10 |
| `.generate_from_frequencies(dict)` | Build cloud from {word: count} dict | When you have pre-computed frequencies | — | 10 |
| `plt.imshow(wc, interpolation)` | Display word cloud in matplotlib | Render the image | `interpolation='bilinear'` for smoothness | 10 |

---

## 7. STANDARD LIBRARY

### `re` (Regex)

| Syntax | Why | Assignment |
|---|---|---|
| `re.compile(pattern, flags)` | Pre-compile regex for repeated use — faster | random_picker |
| `re.MULTILINE` | `^` and `$` match start/end of each line | random_picker |
| `re.DOTALL` | `.` also matches newlines | random_picker |
| `pattern.finditer(string)` | Iterate over all matches | random_picker |
| `re.search(pattern, string)` | Find first match | random_picker |
| `re.sub(pattern, repl, string)` | Replace pattern matches | 10 |

### `os`

| Syntax | Why | Assignment |
|---|---|---|
| `os.path.join(dir, file)` | Cross-platform path joining | random_picker, datasets |
| `os.path.dirname(path)` | Get directory part of path | random_picker, datasets |
| `os.path.abspath(path)` | Get absolute path | random_picker, datasets |
| `os.path.exists(path)` | Check if file/dir exists | random_picker |
| `os.listdir(dir)` | List files in directory | datasets |
| `os.path.getsize(path)` | Get file size in bytes | datasets |

### `string` Operations

| Syntax | Why | Assignment |
|---|---|---|
| `.lower()` | Lowercase text — normalize for comparison | 1, 10 |
| `.strip()` | Remove leading/trailing whitespace | 1, 10 |
| `.replace(old, new)` | Replace substrings | 1, 10 |
| `string.punctuation` | All punctuation characters: `!"#$%...` | 10 |

---

## 8. SPARK / SCALA SYNTAX (Assignment 12)

### SparkSession & RDD

| Syntax | What It Does |
|---|---|
| `SparkSession.builder.appName("name").getOrCreate()` | Create Spark entry point |
| `spark.sparkContext` | Get SparkContext from session |
| `sc.parallelize(collection)` | Create RDD from a Python list |
| `sc.textFile("path")` | Create RDD from a text file |

### RDD Transformations (lazy)

| Syntax | What It Does |
|---|---|
| `.map(func)` | Apply function to each element — **1:1** |
| `.flatMap(func)` | Apply function + flatten — **1:many** (e.g., split sentence → words) |
| `.filter(func)` | Keep elements where func returns True |
| `.distinct()` | Remove duplicates |
| `.union(rdd)` | Combine two RDDs |
| `.intersection(rdd)` | Common elements only |
| `.cartesian(rdd)` | All pairwise combinations |

### RDD Actions (trigger computation)

| Syntax | What It Does |
|---|---|
| `.collect()` | Return all elements to driver (use on small data!) |
| `.count()` | Count number of elements |
| `.first()` | Return first element |
| `.take(n)` | Return first n elements |
| `.reduce(func)` | Aggregate using a binary function (e.g., lambda a,b: a+b) |
| `.foreach(func)` | Apply func to each element (side effects) |
| `.countByKey()` | Count per key for paired RDDs |

### Spark DataFrames

| Syntax | What It Does |
|---|---|
| `spark.read.csv("path", header=True, inferSchema=True)` | Read CSV → Spark DataFrame |
| `spark.read.json("path")` | Read JSON → Spark DataFrame |
| `df.select("col1", "col2")` | Select columns |
| `df.filter(df.col > value)` | Filter rows |
| `df.groupBy("col")` | Group by column |
| `df.groupBy("col").agg({"col2": "avg"})` | Group + aggregate |
| `df.createOrReplaceTempView("name")` | Register DataFrame as SQL table |
| `spark.sql("SELECT ... FROM name")` | Run SQL on registered DataFrame |

---

## 9. IMPALA SQL SYNTAX (Assignment 11)

Apache Impala runs standard SQL on Hadoop data:

| Syntax | What It Does |
|---|---|
| `CREATE TABLE t (col TYPE) STORED AS PARQUET` | Create table |
| `LOAD DATA INPATH 'hdfs://...' INTO TABLE t` | Load data |
| `SELECT col1, col2 FROM t WHERE cond` | Basic query |
| `GROUP BY col` | Aggregate grouping |
| `ORDER BY col ASC/DESC` | Sort results |
| `COUNT(*)`, `SUM()`, `AVG()`, `MIN()`, `MAX()` | Aggregate functions |
| `JOIN t2 ON t1.key = t2.key` | Join tables |
| `LEFT JOIN`, `RIGHT JOIN`, `FULL OUTER JOIN` | Join variants |
| `ROW_NUMBER() OVER (PARTITION BY col ORDER BY col)` | Window function |
| `RANK()`, `DENSE_RANK() OVER (...)` | Ranking window functions |
| `INSERT OVERWRITE DIRECTORY 'hdfs://...' SELECT ...` | Export results |
