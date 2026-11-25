# Software Application for Data Science and Analytics (SADSA)

**Version 2**  
**Author:** Dr. Kamakshaiah Musunuru  
**Firm:** Amchik Solutions, India  
**Contact:** dr.m.kamakshaiah@gmail.com  
**Profile:** https://github.com/codingfigs

SADSA is a desktop application for data science and analytics built around a single Python file. It provides editable CSV workflows, descriptive and inferential statistics, factor and cluster analysis, time-series analysis, machine learning, transactional analytics, advanced missing-data treatment, flexible plotting, and a persistent Python console with `app` and `df` objects.

## Highlights

- Single-file Tkinter application with menus, toolbars, dialogs, reports, and exportable charts.
- 20 configurable plot types through `Plots > Plot Builder` and `app.visualize(...)`.
- Supervised and unsupervised machine learning with simulation data and detailed report windows.
- Transactional analytics including Apriori, FP-Growth, Eclat, closed/maximal itemsets, RFM, sequence patterns, utility rules, and recommendations.
- Time-series simulations for stationary, non-stationary, seasonal, Holt-Winters, moving-average, random-walk, and custom series.
- Advanced missing-data treatment with regression, random forest, KNN, iterative imputation, MCAR/MAR/MNAR simulation, and missingness significance tests.
- Downloadable documentation in TXT, Markdown, DOCX, and PDF from `Help > Documentation`.
- Built-in Python console with script editor and programmatic API access.
- Windows portable EXE, Windows installer, and Linux setup script.

## Repository contents

- `pyfda.py` - complete application source.
- `requirements.txt` - Python packages required to run `pyfda.py`.

## Installation

### Windows portable

1. Extract `SADSA_v2_portable.zip` or copy `dist/SADSA_v2.exe` together with `license.json`, `license.txt`, and the `Data` folder.
2. Double-click `SADSA_v2.exe`.
3. Enter a valid license key (contact dr.m.kamakshaiah@gmail.com for free license key for full version) when prompted, or continue in trial mode (30 day full version).

### Windows setup

1. Run `installer_output/SADSA_v2_Setup.exe`.
2. Choose an installation folder and accept the defaults for a standard installation.
3. Launch SADSA from the Start menu or desktop shortcut.

The installer bundles the Python runtime, application files, icon, sample data, license files, and this README.

### Linux

1. Make the script executable: `chmod +x install_linux.sh`.
2. Run it: `./install_linux.sh`.
3. The script creates a virtual environment in `.venv`, installs `requirements.txt`, and launches SADSA.
4. Start it later with `./install_linux.sh` or `.venv/bin/python pyfda.py`.

Linux requires a desktop environment with Tkinter available. If Tkinter is missing, install the distribution package such as `python3-tk` on Debian/Ubuntu.

## Quick start

1. Open a CSV with `File > Open CSV File`, or generate demo data from `Transactional Analytics`, `Machine Learning > Simulation Data`, or `Data Analytics > Time Series Analysis > Data Simulations`.
2. Edit cells directly in the grid or use `Edit` and `Transformations` menus.
3. Run analyses from `Data Analytics`, `Machine Learning`, or `Transactional Analytics`.
4. Build charts from `Plots > Plot Builder` or call `app.visualize(...)` in the Python console.
5. Open `Help > Documentation` to view or download the full manual.

## Python console

Open `Python Console` from the menu. The console preloads `app`, `df`, pandas, NumPy, Matplotlib, seaborn, scipy, scikit-learn, and common path helpers.

```python
app.head()
app.describe()
app.missing()
app.treat_missing_advanced("income", "regression", ["age", "score"])
app.simulate_missing_data(["age", "income"], 0.2, "mar", "score", 42)
app.test_missing_significance(["income"], group_column="region")
app.simulate_time_series(preset="holt_winters", rows=96)
app.visualize("scatter", x="sales", y="profit", hue="region")
app.classify("random_forest", "Target", ["Feature_1", "Feature_2"], show=True)
app.open_documentation()
```

## Complete documentation from Help > Documentation

## Title Section

| Field | Detail |
| --- | --- |
| Title | Software Application for Data Science and Analytics (SADSA) |
| Version | Version 2 |
| Author | Dr. Kamakshaiah Musunuru |
| Firm | Amchik Solutions, India |
| Email | dr.m.kamakshaiah@gmail.com |
| Author Profile | https://github.com/Kamakshaiah |
| License Status | TRIAL |
| License Details | TRIAL license active; report and plot exports are locked until a FULL key is activated. |

## Contents

1. File
  1.1. Main Menu
2. Edit
  2.1. Main Menu
  2.2. Missing Data Treatment
3. Transformations
  3.1. Main Menu
  3.2. Data Decomposition
  3.3. Data Standardization
4. Data Analytics
  4.1. Descriptive Statistics
  4.2. Inferential Statistics
  4.3. Factor Analysis
  4.4. Correlation & Regression
  4.5. Cluster Analysis
  4.6. Time Series Analysis
  4.7. Time Series Analysis > Data Simulations
5. Machine Learning
  5.1. Supervised Learning
  5.2. Unsupervised Learning
  5.3. Simulation Data
6. Transactional Analytics
  6.1. Generate Demo Data
  6.2. Frequent Patterns
  6.3. Business Applications
7. Plots
  7.1. Main Menu
  7.2. Distribution
  7.3. Categorical
  7.4. Relationships
  7.5. Business Analytics
8. Python Console
  8.1. Main Menu
9. Help
  9.1. Main Menu

## Technique Documentation

Each technique lists inputs, processing steps, expected outputs, and console examples for both `app` and `df`.

## 1. File

### 1.1. Main Menu

#### 1.1.1. Open CSV File

**Menu path:** File > Open CSV File

- **Inputs:** A CSV file path chosen in the file dialog.
- **Process:** Reads the CSV with pandas and renders it in the data grid.
- **Expected outputs:** Loaded data grid and refreshed console df.

**App console**

```python
app.load_csv("data.csv")
```

**df console**

```python
df = pd.read_csv("data.csv"); app.set_df(df)
```

#### 1.1.2. Save Data

**Menu path:** File > Save Data

- **Inputs:** Edited cells in the visible grid.
- **Process:** Writes grid edits back to the in-memory DataFrame.
- **Expected outputs:** Updated in-memory data.

**App console**

```python
app.save_csv(app.get_df(), "saved.csv")
```

**df console**

```python
app.save_data()
```

#### 1.1.3. Save Data File

**Menu path:** File > Save Data File

- **Inputs:** A target CSV file name.
- **Process:** Exports the current DataFrame without the index.
- **Expected outputs:** CSV file on disk.

**App console**

```python
app.save_csv("saved.csv")
```

**df console**

```python
df.to_csv("saved.csv", index=False)
```

#### 1.1.4. Clear Data

**Menu path:** File > Clear Data

- **Inputs:** Confirmation to clear the grid.
- **Process:** Removes all rows and columns from the current data.
- **Expected outputs:** Empty data grid.

**App console**

```python
app.set_df(pd.DataFrame())
```

**df console**

```python
df = df.iloc[0:0]
```

#### 1.1.5. View License

**Menu path:** File > View License

- **Inputs:** License key string stored in license.json.
- **Process:** Displays the license agreement and activation state.
- **Expected outputs:** License window.

**App console**

```python
print(app.license_type)
```

**df console**

```python
license_status = app.license_type
```

#### 1.1.6. Exit

**Menu path:** File > Exit

- **Inputs:** Confirmation if data is present.
- **Process:** Closes the application window.
- **Expected outputs:** Application exit.

**App console**

```python
root.destroy()
```

**df console**

```python
df = None
```

## 2. Edit

### 2.1. Main Menu

#### 2.1.1. Rename Columns

**Menu path:** Edit > Rename Columns

- **Inputs:** Existing column name and a new name.
- **Process:** Renames the selected column and refreshes the grid.
- **Expected outputs:** Updated column header.

**App console**

```python
app.set_df(app.get_df().rename(columns={"old": "new"}))
```

**df console**

```python
df = df.rename(columns={"old": "new"}); app.set_df(df)
```

#### 2.1.2. Compute Variable

**Menu path:** Edit > Compute Variable

- **Inputs:** New variable name and a pandas expression.
- **Process:** Evaluates the expression row-wise and creates a column.
- **Expected outputs:** New computed column.

**App console**

```python
app.compute_column("Total", "Price * Qty")
```

**df console**

```python
df["Total"] = df["Price"] * df["Qty"]
```

#### 2.1.3. Recode Variable

**Menu path:** Edit > Recode Variable

- **Inputs:** Source column and old-to-new value mapping.
- **Process:** Replaces values using the supplied mapping.
- **Expected outputs:** Recoded column.

**App console**

```python
app.recode_column("gender", {"M": "Male", "F": "Female"})
```

**df console**

```python
df["gender"] = df["gender"].replace({"M": "Male", "F": "Female"})
```

#### 2.1.4. Set Values

**Menu path:** Edit > Set Values

- **Inputs:** Row id, column, and replacement value.
- **Process:** Updates the selected cell in the DataFrame.
- **Expected outputs:** Edited data grid.

**App console**

```python
app.get_df().loc[2, "age"] = 30
```

**df console**

```python
df.loc[2, "age"] = 30; app.refresh_df()
```

### 2.2. Missing Data Treatment

#### 2.2.1. Basic Treatment

**Menu path:** Edit > Missing Data Treatment > Basic Treatment

- **Inputs:** Column and treatment method such as mean, median, mode, or drop.
- **Process:** Fills or removes missing values and refreshes the grid.
- **Expected outputs:** Treated column.

**App console**

```python
app.treat_missing("age", "mean")
```

**df console**

```python
df["age"] = df["age"].fillna(df["age"].mean())
```

#### 2.2.2. Regression Imputation

**Menu path:** Edit > Missing Data Treatment > Regression Imputation

- **Inputs:** Numeric target column and numeric predictor columns.
- **Process:** Fits a linear regression on observed rows and predicts missing target values.
- **Expected outputs:** Imputed target column and before/after report with a chart.

**App console**

```python
app.treat_missing_advanced("income", "regression", ["age", "score"])
```

**df console**

```python
df["income"] = df["income"].fillna(df["income"].mean())
```

#### 2.2.3. Random Forest Imputation

**Menu path:** Edit > Missing Data Treatment > Random Forest Imputation

- **Inputs:** Target column and predictor columns; the target may be numeric or categorical.
- **Process:** Fits a random forest model on observed rows and predicts missing values.
- **Expected outputs:** Imputed target column and model-based report.

**App console**

```python
app.treat_missing_advanced("segment", "random_forest", ["age", "income"])
```

**df console**

```python
df["segment"] = df["segment"].fillna(df["segment"].mode().iloc[0])
```

#### 2.2.4. KNN Imputation

**Menu path:** Edit > Missing Data Treatment > KNN Imputation

- **Inputs:** Numeric target and predictor columns.
- **Process:** Uses KNNImputer to estimate missing values from the nearest complete cases.
- **Expected outputs:** Imputed target column and before/after report.

**App console**

```python
app.treat_missing_advanced("income", "knn", ["age", "score"])
```

**df console**

```python
from sklearn.impute import KNNImputer; df[["income"]] = KNNImputer().fit_transform(df[["income"]])
```

#### 2.2.5. Iterative Imputation

**Menu path:** Edit > Missing Data Treatment > Iterative Imputation

- **Inputs:** Numeric target and predictor columns.
- **Process:** Uses iterative multivariate imputation with scikit-learn IterativeImputer.
- **Expected outputs:** Imputed target column and before/after report.

**App console**

```python
app.treat_missing_advanced("income", "iterative", ["age", "score"])
```

**df console**

```python
from sklearn.experimental import enable_iterative_imputer; from sklearn.impute import IterativeImputer
```

#### 2.2.6. Simulate Missing Data

**Menu path:** Edit > Missing Data Treatment > Simulate Missing Data

- **Inputs:** Columns, missing rate, mechanism, optional MAR predictor, and seed.
- **Process:** Introduces MCAR, MAR, or MNAR missing values for demonstrations and tests.
- **Expected outputs:** Updated grid with simulated missing cells and a missing-count report.

**App console**

```python
app.simulate_missing_data(["age", "income"], 0.2, "mar", "score", 42)
```

**df console**

```python
df.loc[df.sample(frac=0.2).index, "age"] = np.nan
```

#### 2.2.7. Missingness Significance Test

**Menu path:** Edit > Missing Data Treatment > Missingness Significance Test

- **Inputs:** Columns to test, optional group column, and alpha.
- **Process:** Runs chi-square or observed-variable screening with multiple-comparison correction.
- **Expected outputs:** Test statistics, p-values, adjusted thresholds, and interpretation.

**App console**

```python
app.test_missing_significance(["age", "income"], group_column="region")
```

**df console**

```python
pd.crosstab(df["region"], df["age"].isna())
```

#### 2.2.8. Missingness Report

**Menu path:** Edit > Missing Data Treatment > Missingness Report

- **Inputs:** The current data grid.
- **Process:** Summarizes missing counts, percentages, and missingness patterns.
- **Expected outputs:** Missingness summary report with a percentage chart.

**App console**

```python
app.missingness_report()
```

**df console**

```python
df.isna().sum()
```

## 3. Transformations

### 3.1. Main Menu

#### 3.1.1. Data Simulations

**Menu path:** Transformations > Data Simulations

- **Inputs:** Pre-filled rows, seed, and variable specifications in one dialog.
- **Process:** Generates int, float, normal, or category variables from inferred defaults.
- **Expected outputs:** Simulated columns appended to the grid.

**App console**

```python
app.simulate({'age': ('int', 18, 65)}, rows=100, seed=42)
```

**df console**

```python
df["age"] = np.random.default_rng(42).integers(18, 65, 100)
```

#### 3.1.2. Multivariate Normal Distribution

**Menu path:** Transformations > Multivariate Normal Distribution

- **Inputs:** Pre-filled names, means, covariance matrix, samples, and seed.
- **Process:** Draws samples from a multivariate normal distribution.
- **Expected outputs:** Simulated numeric columns appended to the grid.

**App console**

```python
app.simulate_multivariate_normal(['x', 'y'], [0, 1], [[1, 0.5], [0.5, 1]], samples=100)
```

**df console**

```python
df[["x", "y"]] = pd.DataFrame(np.random.multivariate_normal([0, 1], [[1, .5], [.5, 1]], 100), columns=["x", "y"])
```

### 3.2. Data Decomposition

#### 3.2.1. Cholesky Decomposition

**Menu path:** Transformations > Data Decomposition > Cholesky Decomposition

- **Inputs:** Numeric columns from the grid.
- **Process:** Computes the CHOLESKY matrix factorization.
- **Expected outputs:** Factor matrices rendered in a report.

**App console**

```python
app.decompose_columns(["x", "y"], "cholesky")
```

**df console**

```python
U, S, Vt = np.linalg.svd(df[["x", "y"]])
```

#### 3.2.2. QR Decomposition

**Menu path:** Transformations > Data Decomposition > QR Decomposition

- **Inputs:** Numeric columns from the grid.
- **Process:** Computes the QR matrix factorization.
- **Expected outputs:** Factor matrices rendered in a report.

**App console**

```python
app.decompose_columns(["x", "y"], "qr")
```

**df console**

```python
U, S, Vt = np.linalg.svd(df[["x", "y"]])
```

#### 3.2.3. SVD Decomposition

**Menu path:** Transformations > Data Decomposition > SVD Decomposition

- **Inputs:** Numeric columns from the grid.
- **Process:** Computes the SVD matrix factorization.
- **Expected outputs:** Factor matrices rendered in a report.

**App console**

```python
app.decompose_columns(["x", "y"], "svd")
```

**df console**

```python
U, S, Vt = np.linalg.svd(df[["x", "y"]])
```

#### 3.2.4. Eigen Decomposition

**Menu path:** Transformations > Data Decomposition > Eigen Decomposition

- **Inputs:** Numeric columns from the grid.
- **Process:** Computes the EIG matrix factorization.
- **Expected outputs:** Factor matrices rendered in a report.

**App console**

```python
app.decompose_columns(["x", "y"], "eig")
```

**df console**

```python
U, S, Vt = np.linalg.svd(df[["x", "y"]])
```

### 3.3. Data Standardization

#### 3.3.1. Min-Max Scaling

**Menu path:** Transformations > Data Standardization > Min-Max Scaling

- **Inputs:** Numeric variable to transform.
- **Process:** Applies min-max scaling and stores the result as a new column.
- **Expected outputs:** Standardized or transformed variable.

**App console**

```python
app.standardize_column("x", "Min-Max")
```

**df console**

```python
df["x_scaled"] = (df["x"] - df["x"].min()) / (df["x"].max() - df["x"].min())
```

#### 3.3.2. Z-Score Standardization

**Menu path:** Transformations > Data Standardization > Z-Score Standardization

- **Inputs:** Numeric variable to transform.
- **Process:** Applies z-score standardization and stores the result as a new column.
- **Expected outputs:** Standardized or transformed variable.

**App console**

```python
app.standardize_column("x", "Z-Score")
```

**df console**

```python
df["x_scaled"] = (df["x"] - df["x"].min()) / (df["x"].max() - df["x"].min())
```

#### 3.3.3. Decimal Scaling

**Menu path:** Transformations > Data Standardization > Decimal Scaling

- **Inputs:** Numeric variable to transform.
- **Process:** Applies decimal scaling and stores the result as a new column.
- **Expected outputs:** Standardized or transformed variable.

**App console**

```python
app.standardize_column("x", "Decimal Scaling")
```

**df console**

```python
df["x_scaled"] = (df["x"] - df["x"].min()) / (df["x"].max() - df["x"].min())
```

#### 3.3.4. Log Transformation

**Menu path:** Transformations > Data Standardization > Log Transformation

- **Inputs:** Numeric variable to transform.
- **Process:** Applies log transformation and stores the result as a new column.
- **Expected outputs:** Standardized or transformed variable.

**App console**

```python
app.standardize_column("x", "Log")
```

**df console**

```python
df["x_scaled"] = (df["x"] - df["x"].min()) / (df["x"].max() - df["x"].min())
```

#### 3.3.5. Log-Normal Transformation

**Menu path:** Transformations > Data Standardization > Log-Normal Transformation

- **Inputs:** Numeric variable to transform.
- **Process:** Applies log-normal transformation and stores the result as a new column.
- **Expected outputs:** Standardized or transformed variable.

**App console**

```python
app.standardize_column("x", "Log-Normal")
```

**df console**

```python
df["x_scaled"] = (df["x"] - df["x"].min()) / (df["x"].max() - df["x"].min())
```

## 4. Data Analytics

### 4.1. Descriptive Statistics

#### 4.1.1. Frequency Tables

**Menu path:** Data Analytics > Descriptive Statistics > Frequency Tables

- **Inputs:** One or more categorical columns.
- **Process:** Counts each distinct value and builds a grouped chart.
- **Expected outputs:** Frequency tables and bar chart.

**App console**

```python
app.frequencies(['gender', 'education'])
```

**df console**

```python
df["gender"].value_counts()
```

#### 4.1.2. Summary Statistics

**Menu path:** Data Analytics > Descriptive Statistics > Summary Statistics

- **Inputs:** Numeric columns.
- **Process:** Computes count, mean, standard deviation, quartiles, min, and max.
- **Expected outputs:** Descriptive statistics report and boxplot.

**App console**

```python
app.summary_stats(['age', 'income'])
```

**df console**

```python
df[['age', 'income']].describe()
```

### 4.2. Inferential Statistics

#### 4.2.1. T-Test

**Menu path:** Data Analytics > Inferential Statistics > T-Test

- **Inputs:** Two numeric columns.
- **Process:** Runs an independent two-sample t-test.
- **Expected outputs:** Statistic, p-value, and boxplot.

**App console**

```python
app.ttest("score_a", "score_b")
```

**df console**

```python
stats.ttest_ind(df["score_a"], df["score_b"], nan_policy="omit")
```

#### 4.2.2. Chi-Square Test

**Menu path:** Data Analytics > Inferential Statistics > Chi-Square Test

- **Inputs:** Two categorical columns.
- **Process:** Tests independence using a contingency table.
- **Expected outputs:** Chi-square statistic, p-value, degrees of freedom, chart.

**App console**

```python
app.chisquare("gender", "purchased")
```

**df console**

```python
stats.chi2_contingency(pd.crosstab(df["gender"], df["purchased"]))
```

#### 4.2.3. Normality Tests

**Menu path:** Data Analytics > Inferential Statistics > Normality Tests

- **Inputs:** Numeric columns.
- **Process:** Runs Shapiro-Wilk and Kolmogorov-Smirnov tests.
- **Expected outputs:** Per-column test statistics and histograms.

**App console**

```python
app.normality(['age', 'income'])
```

**df console**

```python
stats.shapiro(df["age"].dropna())
```

#### 4.2.4. ANOVA

**Menu path:** Data Analytics > Inferential Statistics > ANOVA

- **Inputs:** Dependent numeric variable and a categorical factor.
- **Process:** Runs one-way analysis of variance.
- **Expected outputs:** F-statistic, p-value, and group boxplot.

**App console**

```python
app.anova("score", "group")
```

**df console**

```python
stats.f_oneway(*[g["score"] for _, g in df.groupby("group")])
```

#### 4.2.5. MANOVA

**Menu path:** Data Analytics > Inferential Statistics > MANOVA

- **Inputs:** Multiple dependent variables and a categorical factor.
- **Process:** Fits a multivariate analysis of variance model.
- **Expected outputs:** MANOVA result table and scatter matrix.

**App console**

```python
app.manova(["score_a", "score_b"], "group")
```

**df console**

```python
MANOVA.from_formula("score_a + score_b ~ group", data=df).mv_test()
```

### 4.3. Factor Analysis

#### 4.3.1. Exploratory Factor Analysis

**Menu path:** Data Analytics > Factor Analysis > Exploratory Factor Analysis

- **Inputs:** Numeric columns and number of factors.
- **Process:** Extracts latent factors and computes Bartlett and KMO tests.
- **Expected outputs:** Loadings, test results, scree and loading plots.

**App console**

```python
app.efa(['q1', 'q2', 'q3'], n_factors=2)
```

**df console**

```python
FactorAnalyzer(n_factors=2).fit(df[["q1", "q2", "q3"]])
```

#### 4.3.2. Confirmatory Factor Analysis

**Menu path:** Data Analytics > Factor Analysis > Confirmatory Factor Analysis

- **Inputs:** semopy model description and columns.
- **Process:** Fits the specified structural equation model.
- **Expected outputs:** Model inspection table and SEM path diagram.

**App console**

```python
app.cfa("F1 =~ q1 + q2 + q3")
```

**df console**

```python
Model("F1 =~ q1 + q2 + q3").fit(df)
```

### 4.4. Correlation & Regression

#### 4.4.1. Correlation Analysis

**Menu path:** Data Analytics > Correlation & Regression > Correlation Analysis

- **Inputs:** Numeric columns or two variable sets.
- **Process:** Computes Pearson correlation or canonical correlation.
- **Expected outputs:** Coefficient table, p-values, heatmap or scatter plot.

**App console**

```python
app.correlation(['x', 'y', 'z'])
```

**df console**

```python
df[["x", "y", "z"]].corr()
```

#### 4.4.2. Regression Analysis

**Menu path:** Data Analytics > Correlation & Regression > Regression Analysis

- **Inputs:** Dependent variable, independent variables, and model kind.
- **Process:** Fits OLS or GLM regression with statsmodels.
- **Expected outputs:** Coefficient table and fitted-values plot.

**App console**

```python
app.regression("sales", ["price", "ads"], kind="ols")
```

**df console**

```python
sm.OLS(df["sales"], sm.add_constant(df[["price", "ads"]])).fit()
```

### 4.5. Cluster Analysis

#### 4.5.1. K-Means Clustering

**Menu path:** Data Analytics > Cluster Analysis > K-Means Clustering

- **Inputs:** Numeric columns and cluster count.
- **Process:** Standardized k-means fit with cluster labels written to the grid.
- **Expected outputs:** Centers, cluster counts, and scatter plot.

**App console**

```python
app.kmeans(['x', 'y'], n_clusters=3, show=True)
```

**df console**

```python
KMeans(n_clusters=3, n_init=10).fit_predict(df[["x", "y"]])
```

#### 4.5.2. Hierarchical Clustering

**Menu path:** Data Analytics > Cluster Analysis > Hierarchical Clustering

- **Inputs:** Numeric columns.
- **Process:** Builds a Ward linkage tree and cuts it into clusters.
- **Expected outputs:** Linkage matrix, cluster counts, dendrogram.

**App console**

```python
app.hierarchical(['x', 'y'], show=True)
```

**df console**

```python
linkage(df[['x', 'y']], method='ward')
```

### 4.6. Time Series Analysis

#### 4.6.1. Stationarity Tests

**Menu path:** Data Analytics > Time Series Analysis > Stationarity Tests

- **Inputs:** A time series column.
- **Process:** Runs ADF and KPSS tests.
- **Expected outputs:** Test statistics, p-values, and series plot.

**App console**

```python
app.stationarity("sales")
```

**df console**

```python
adfuller(df["sales"].dropna())
```

#### 4.6.2. Seasonal Decomposition

**Menu path:** Data Analytics > Time Series Analysis > Seasonal Decomposition

- **Inputs:** A time series column and period.
- **Process:** Decomposes the series into trend, seasonal, and residual parts.
- **Expected outputs:** Component table and four-panel plot.

**App console**

```python
app.seasonal_decomposition("sales", period=12)
```

**df console**

```python
seasonal_decompose(df["sales"].dropna(), period=12)
```

#### 4.6.3. Holt-Winters Method

**Menu path:** Data Analytics > Time Series Analysis > Holt-Winters Method

- **Inputs:** A time series column and seasonal periods.
- **Process:** Fits exponential smoothing with trend and seasonality.
- **Expected outputs:** Fitness measures and actual-versus-forecast plot.

**App console**

```python
app.holt_winters("sales", seasonal_periods=12)
```

**df console**

```python
ExponentialSmoothing(df["sales"], seasonal_periods=12).fit()
```

#### 4.6.4. Moving Averages

**Menu path:** Data Analytics > Time Series Analysis > Moving Averages

- **Inputs:** A time series column and window size.
- **Process:** Computes rolling mean and error metrics.
- **Expected outputs:** Moving-average table, metrics, and chart.

**App console**

```python
app.moving_average("sales", window=3)[0]
```

**df console**

```python
df["sales"].rolling(3).mean()
```

### 4.7. Time Series Analysis > Data Simulations

#### 4.7.1. Simulate Stationary Series

**Menu path:** Data Analytics > Time Series Analysis > Data Simulations > Simulate Stationary Series

- **Inputs:** Time variable, data variable, rows, start, frequency, base, noise, and seed; prefilled from current data.
- **Process:** Generates a flat series with noise and no trend or seasonality.
- **Expected outputs:** New time-series grid suited to ADF and KPSS stationarity tests.

**App console**

```python
app.simulate_time_series(preset="stationary", time_column="Date", value_column="Sales", rows=96)
```

**df console**

```python
df = pd.DataFrame({"Date": pd.date_range("2020-01-01", periods=96, freq="ME"), "Sales": np.random.normal(100, 2, 96)})
```

#### 4.7.2. Simulate Non-Stationary Series

**Menu path:** Data Analytics > Time Series Analysis > Data Simulations > Simulate Non-Stationary Series

- **Inputs:** Prefilled time and data options with trend slope and noise.
- **Process:** Generates a series with a linear trend and no seasonality.
- **Expected outputs:** New grid for demonstrating non-stationary behavior.

**App console**

```python
app.simulate_time_series(preset="non_stationary", trend=2, rows=96)
```

**df console**

```python
df["Sales"] = 100 + 2 * np.arange(96) + np.random.normal(0, 3, 96)
```

#### 4.7.3. Simulate Seasonal Series

**Menu path:** Data Analytics > Time Series Analysis > Data Simulations > Simulate Seasonal Series

- **Inputs:** Prefilled time and data options with seasonal model, amplitude, and period.
- **Process:** Generates additive or multiplicative seasonality with optional trend.
- **Expected outputs:** New grid for seasonal decomposition demonstrations.

**App console**

```python
app.simulate_time_series(preset="seasonal", seasonality=15, period=12, rows=96)
```

**df console**

```python
df["Sales"] = 100 + 15 * np.sin(2 * np.pi * np.arange(96) / 12)
```

#### 4.7.4. Simulate Trend + Seasonal Series

**Menu path:** Data Analytics > Time Series Analysis > Data Simulations > Simulate Trend + Seasonal Series

- **Inputs:** Prefilled trend type, trend slope, seasonal amplitude, period, and noise.
- **Process:** Generates trend-plus-seasonality data suited to Holt-Winters fitting.
- **Expected outputs:** New grid with a complete actual-versus-forecast demo dataset.

**App console**

```python
app.simulate_time_series(preset="holt_winters", trend=1.5, seasonality=20, period=12)
```

**df console**

```python
df["Sales"] = 100 + 1.5 * np.arange(96) + 20 * np.sin(2 * np.pi * np.arange(96) / 12)
```

#### 4.7.5. Simulate Moving-Average Series

**Menu path:** Data Analytics > Time Series Analysis > Data Simulations > Simulate Moving-Average Series

- **Inputs:** Prefilled smoothing-friendly trend, noise, and window context.
- **Process:** Generates a noisy but smooth trend series for moving-average analysis.
- **Expected outputs:** New grid suitable for rolling-mean demonstrations.

**App console**

```python
app.simulate_time_series(preset="moving_average", noise=4, rows=96)
```

**df console**

```python
df["Sales"] = df["Sales"].rolling(3).mean()
```

#### 4.7.6. Simulate Random Walk Series

**Menu path:** Data Analytics > Time Series Analysis > Data Simulations > Simulate Random Walk Series

- **Inputs:** Prefilled base, drift, noise, and seed.
- **Process:** Generates an integrated random-walk series.
- **Expected outputs:** New grid for demonstrating strong non-stationarity.

**App console**

```python
app.simulate_time_series(preset="random_walk", random_walk=True)
```

**df console**

```python
df["Sales"] = 100 + np.cumsum(np.random.normal(0, 2, 96))
```

#### 4.7.7. Simulate Custom Series

**Menu path:** Data Analytics > Time Series Analysis > Data Simulations > Simulate Custom Series

- **Inputs:** All time-variable and data-variable options in one auto-populated dialog.
- **Process:** Generates custom date, numeric-index, or period-label series with selected trend, seasonality, and noise.
- **Expected outputs:** New grid and simulation report.

**App console**

```python
app.simulate_time_series(time_mode="numeric", trend_type="quadratic", seasonal_model="multiplicative", noise_type="uniform")
```

**df console**

```python
df = app.simulate_time_series(apply=False)
```

## 5. Machine Learning

### 5.1. Supervised Learning

#### 5.1.1. Logistic Regression

**Menu path:** Machine Learning > Supervised Learning > Logistic Regression

- **Inputs:** Target column and feature columns, plus a train ratio.
- **Process:** Splits data, fits the classifier, predicts the test set, and builds a full report.
- **Expected outputs:** Test-set predictions, margin-labeled confusion matrix, metrics, technique details, and technique-specific plots.

**App console**

```python
app.classify("logistic", "Target", ["Feature_1", "Feature_2"], show=True)
```

**df console**

```python
df = app.generate_ml_demo("logistic", apply=False); df.head()
```

#### 5.1.2. Decision Tree

**Menu path:** Machine Learning > Supervised Learning > Decision Tree

- **Inputs:** Target column and feature columns, plus a train ratio.
- **Process:** Splits data, fits the classifier, predicts the test set, and builds a full report.
- **Expected outputs:** Test-set predictions, margin-labeled confusion matrix, metrics, technique details, and technique-specific plots.

**App console**

```python
app.classify("decision_tree", "Target", ["Feature_1", "Feature_2"], show=True)
```

**df console**

```python
df = app.generate_ml_demo("decision_tree", apply=False); df.head()
```

#### 5.1.3. Random Forest

**Menu path:** Machine Learning > Supervised Learning > Random Forest

- **Inputs:** Target column and feature columns, plus a train ratio.
- **Process:** Splits data, fits the classifier, predicts the test set, and builds a full report.
- **Expected outputs:** Test-set predictions, margin-labeled confusion matrix, metrics, technique details, and technique-specific plots.

**App console**

```python
app.classify("random_forest", "Target", ["Feature_1", "Feature_2"], show=True)
```

**df console**

```python
df = app.generate_ml_demo("random_forest", apply=False); df.head()
```

#### 5.1.4. Naive Bayes

**Menu path:** Machine Learning > Supervised Learning > Naive Bayes

- **Inputs:** Target column and feature columns, plus a train ratio.
- **Process:** Splits data, fits the classifier, predicts the test set, and builds a full report.
- **Expected outputs:** Test-set predictions, margin-labeled confusion matrix, metrics, technique details, and technique-specific plots.

**App console**

```python
app.classify("naive_bayes", "Target", ["Feature_1", "Feature_2"], show=True)
```

**df console**

```python
df = app.generate_ml_demo("naive_bayes", apply=False); df.head()
```

#### 5.1.5. K-Nearest Neighbors

**Menu path:** Machine Learning > Supervised Learning > K-Nearest Neighbors

- **Inputs:** Target column and feature columns, plus a train ratio.
- **Process:** Splits data, fits the classifier, predicts the test set, and builds a full report.
- **Expected outputs:** Test-set predictions, margin-labeled confusion matrix, metrics, technique details, and technique-specific plots.

**App console**

```python
app.classify("knn", "Target", ["Feature_1", "Feature_2"], show=True)
```

**df console**

```python
df = app.generate_ml_demo("knn", apply=False); df.head()
```

#### 5.1.6. Neural Network

**Menu path:** Machine Learning > Supervised Learning > Neural Network

- **Inputs:** Target column and feature columns, plus a train ratio.
- **Process:** Splits data, fits the classifier, predicts the test set, and builds a full report.
- **Expected outputs:** Test-set predictions, margin-labeled confusion matrix, metrics, technique details, and technique-specific plots.

**App console**

```python
app.classify("neural_network", "Target", ["Feature_1", "Feature_2"], show=True)
```

**df console**

```python
df = app.generate_ml_demo("neural_network", apply=False); df.head()
```

#### 5.1.7. Support Vector Machine

**Menu path:** Machine Learning > Supervised Learning > Support Vector Machine

- **Inputs:** Target column and feature columns, plus a train ratio.
- **Process:** Splits data, fits the classifier, predicts the test set, and builds a full report.
- **Expected outputs:** Test-set predictions, margin-labeled confusion matrix, metrics, technique details, and technique-specific plots.

**App console**

```python
app.classify("svm", "Target", ["Feature_1", "Feature_2"], show=True)
```

**df console**

```python
df = app.generate_ml_demo("svm", apply=False); df.head()
```

### 5.2. Unsupervised Learning

#### 5.2.1. K-Means Clustering

**Menu path:** Machine Learning > Unsupervised Learning > K-Means Clustering

- **Inputs:** Numeric feature columns and algorithm parameters.
- **Process:** Fits the unsupervised model and attaches labels or components when requested.
- **Expected outputs:** Cluster or component tables, counts, scores, and algorithm-specific plots.

**App console**

```python
app.kmeans(["Feature_1", "Feature_2"], show=True)
```

**df console**

```python
features = df.select_dtypes(include=[np.number])
```

#### 5.2.2. Hierarchical Clustering

**Menu path:** Machine Learning > Unsupervised Learning > Hierarchical Clustering

- **Inputs:** Numeric feature columns and algorithm parameters.
- **Process:** Fits the unsupervised model and attaches labels or components when requested.
- **Expected outputs:** Cluster or component tables, counts, scores, and algorithm-specific plots.

**App console**

```python
app.hierarchical(["Feature_1", "Feature_2"], show=True)
```

**df console**

```python
features = df.select_dtypes(include=[np.number])
```

#### 5.2.3. DBSCAN Clustering

**Menu path:** Machine Learning > Unsupervised Learning > DBSCAN Clustering

- **Inputs:** Numeric feature columns and algorithm parameters.
- **Process:** Fits the unsupervised model and attaches labels or components when requested.
- **Expected outputs:** Cluster or component tables, counts, scores, and algorithm-specific plots.

**App console**

```python
app.dbscan(["Feature_1", "Feature_2"], show=True)
```

**df console**

```python
features = df.select_dtypes(include=[np.number])
```

#### 5.2.4. Gaussian Mixture

**Menu path:** Machine Learning > Unsupervised Learning > Gaussian Mixture

- **Inputs:** Numeric feature columns and algorithm parameters.
- **Process:** Fits the unsupervised model and attaches labels or components when requested.
- **Expected outputs:** Cluster or component tables, counts, scores, and algorithm-specific plots.

**App console**

```python
app.gaussian_mixture(["Feature_1", "Feature_2"], show=True)
```

**df console**

```python
features = df.select_dtypes(include=[np.number])
```

#### 5.2.5. Principal Component Analysis

**Menu path:** Machine Learning > Unsupervised Learning > Principal Component Analysis

- **Inputs:** Numeric feature columns and algorithm parameters.
- **Process:** Fits the unsupervised model and attaches labels or components when requested.
- **Expected outputs:** Cluster or component tables, counts, scores, and algorithm-specific plots.

**App console**

```python
app.pca(["Feature_1", "Feature_2"], show=True)
```

**df console**

```python
features = df.select_dtypes(include=[np.number])
```

### 5.3. Simulation Data

#### 5.3.1. Logistic Regression Demo

**Menu path:** Machine Learning > Simulation Data > Logistic Regression Demo

- **Inputs:** None; the button uses inferred defaults.
- **Process:** Generates a technique-appropriate synthetic dataset with a fixed seed.
- **Expected outputs:** A new data grid ready for the matching technique.

**App console**

```python
app.generate_ml_demo("logistic", apply=True)
```

**df console**

```python
df = app.generate_ml_demo("logistic", apply=False)
```

#### 5.3.2. Decision Tree Demo

**Menu path:** Machine Learning > Simulation Data > Decision Tree Demo

- **Inputs:** None; the button uses inferred defaults.
- **Process:** Generates a technique-appropriate synthetic dataset with a fixed seed.
- **Expected outputs:** A new data grid ready for the matching technique.

**App console**

```python
app.generate_ml_demo("decision_tree", apply=True)
```

**df console**

```python
df = app.generate_ml_demo("decision_tree", apply=False)
```

#### 5.3.3. Random Forest Demo

**Menu path:** Machine Learning > Simulation Data > Random Forest Demo

- **Inputs:** None; the button uses inferred defaults.
- **Process:** Generates a technique-appropriate synthetic dataset with a fixed seed.
- **Expected outputs:** A new data grid ready for the matching technique.

**App console**

```python
app.generate_ml_demo("random_forest", apply=True)
```

**df console**

```python
df = app.generate_ml_demo("random_forest", apply=False)
```

#### 5.3.4. Naive Bayes Demo

**Menu path:** Machine Learning > Simulation Data > Naive Bayes Demo

- **Inputs:** None; the button uses inferred defaults.
- **Process:** Generates a technique-appropriate synthetic dataset with a fixed seed.
- **Expected outputs:** A new data grid ready for the matching technique.

**App console**

```python
app.generate_ml_demo("naive_bayes", apply=True)
```

**df console**

```python
df = app.generate_ml_demo("naive_bayes", apply=False)
```

#### 5.3.5. K-Nearest Neighbors Demo

**Menu path:** Machine Learning > Simulation Data > K-Nearest Neighbors Demo

- **Inputs:** None; the button uses inferred defaults.
- **Process:** Generates a technique-appropriate synthetic dataset with a fixed seed.
- **Expected outputs:** A new data grid ready for the matching technique.

**App console**

```python
app.generate_ml_demo("knn", apply=True)
```

**df console**

```python
df = app.generate_ml_demo("knn", apply=False)
```

#### 5.3.6. Support Vector Machine Demo

**Menu path:** Machine Learning > Simulation Data > Support Vector Machine Demo

- **Inputs:** None; the button uses inferred defaults.
- **Process:** Generates a technique-appropriate synthetic dataset with a fixed seed.
- **Expected outputs:** A new data grid ready for the matching technique.

**App console**

```python
app.generate_ml_demo("svm", apply=True)
```

**df console**

```python
df = app.generate_ml_demo("svm", apply=False)
```

#### 5.3.7. Neural Network Demo

**Menu path:** Machine Learning > Simulation Data > Neural Network Demo

- **Inputs:** None; the button uses inferred defaults.
- **Process:** Generates a technique-appropriate synthetic dataset with a fixed seed.
- **Expected outputs:** A new data grid ready for the matching technique.

**App console**

```python
app.generate_ml_demo("neural_network", apply=True)
```

**df console**

```python
df = app.generate_ml_demo("neural_network", apply=False)
```

#### 5.3.8. K-Means Demo

**Menu path:** Machine Learning > Simulation Data > K-Means Demo

- **Inputs:** None; the button uses inferred defaults.
- **Process:** Generates a technique-appropriate synthetic dataset with a fixed seed.
- **Expected outputs:** A new data grid ready for the matching technique.

**App console**

```python
app.generate_ml_demo("kmeans", apply=True)
```

**df console**

```python
df = app.generate_ml_demo("kmeans", apply=False)
```

#### 5.3.9. Hierarchical Clustering Demo

**Menu path:** Machine Learning > Simulation Data > Hierarchical Clustering Demo

- **Inputs:** None; the button uses inferred defaults.
- **Process:** Generates a technique-appropriate synthetic dataset with a fixed seed.
- **Expected outputs:** A new data grid ready for the matching technique.

**App console**

```python
app.generate_ml_demo("hierarchical", apply=True)
```

**df console**

```python
df = app.generate_ml_demo("hierarchical", apply=False)
```

#### 5.3.10. DBSCAN Demo

**Menu path:** Machine Learning > Simulation Data > DBSCAN Demo

- **Inputs:** None; the button uses inferred defaults.
- **Process:** Generates a technique-appropriate synthetic dataset with a fixed seed.
- **Expected outputs:** A new data grid ready for the matching technique.

**App console**

```python
app.generate_ml_demo("dbscan", apply=True)
```

**df console**

```python
df = app.generate_ml_demo("dbscan", apply=False)
```

#### 5.3.11. Gaussian Mixture Demo

**Menu path:** Machine Learning > Simulation Data > Gaussian Mixture Demo

- **Inputs:** None; the button uses inferred defaults.
- **Process:** Generates a technique-appropriate synthetic dataset with a fixed seed.
- **Expected outputs:** A new data grid ready for the matching technique.

**App console**

```python
app.generate_ml_demo("gaussian_mixture", apply=True)
```

**df console**

```python
df = app.generate_ml_demo("gaussian_mixture", apply=False)
```

#### 5.3.12. PCA Demo

**Menu path:** Machine Learning > Simulation Data > PCA Demo

- **Inputs:** None; the button uses inferred defaults.
- **Process:** Generates a technique-appropriate synthetic dataset with a fixed seed.
- **Expected outputs:** A new data grid ready for the matching technique.

**App console**

```python
app.generate_ml_demo("pca", apply=True)
```

**df console**

```python
df = app.generate_ml_demo("pca", apply=False)
```

## 6. Transactional Analytics

### 6.1. Generate Demo Data

#### 6.1.1. Basket Matrix

**Menu path:** Transactional Analytics > Generate Demo Data > Basket Matrix

- **Inputs:** Optional confirmation to replace the current grid.
- **Process:** Creates a wide transaction basket matrix.
- **Expected outputs:** Rows as baskets and columns as items.

**App console**

```python
app.generate_basket(transactions=200, item_count=10)
```

**df console**

```python
df = pd.read_csv('basket.csv')
```

#### 6.1.2. Customer Transaction Log

**Menu path:** Transactional Analytics > Generate Demo Data > Customer Transaction Log

- **Inputs:** Optional confirmation to replace the current grid.
- **Process:** Creates long-format customer basket records.
- **Expected outputs:** Customer id, transaction id, and item columns.

**App console**

```python
app.generate_long_transactions(customers=40, transactions_per_customer=4)
```

**df console**

```python
df.groupby('CustomerID').size()
```

#### 6.1.3. Sequence Log

**Menu path:** Transactional Analytics > Generate Demo Data > Sequence Log

- **Inputs:** Optional confirmation to replace the current grid.
- **Process:** Creates customer purchase sequences.
- **Expected outputs:** Customer id and ordered item-sequence columns.

**App console**

```python
app.generate_sequence_log(customers=40, sequence_length=6)
```

**df console**

```python
df.sort_values(['CustomerID', 'SequenceID'])
```

#### 6.1.4. Profit / Utility Basket

**Menu path:** Transactional Analytics > Generate Demo Data > Profit / Utility Basket

- **Inputs:** Optional confirmation to replace the current grid.
- **Process:** Creates baskets with item profit weights.
- **Expected outputs:** Basket matrix plus profit columns.

**App console**

```python
app.generate_profit_basket(transactions=200, item_count=10)
```

**df console**

```python
df.filter(like='Profit').mean()
```

#### 6.1.5. Clear Data

**Menu path:** Transactional Analytics > Generate Demo Data > Clear Data

- **Inputs:** Confirmation to clear the grid.
- **Process:** Removes all transaction data.
- **Expected outputs:** Empty data grid.

**App console**

```python
app.set_df(pd.DataFrame())
```

**df console**

```python
df = df.iloc[0:0]
```

### 6.2. Frequent Patterns

#### 6.2.1. Apriori

**Menu path:** Transactional Analytics > Frequent Patterns > Apriori

- **Inputs:** Basket columns and a minimum support value.
- **Process:** Runs Apriori to discover frequent itemsets.
- **Expected outputs:** Frequent itemsets with support and chart.

**App console**

```python
app.frequent_itemsets(min_support=0.1, algorithm="Apriori")
```

**df console**

```python
basket = df.drop(columns=["CustomerID"], errors="ignore").astype(bool)
```

#### 6.2.2. FP-Growth

**Menu path:** Transactional Analytics > Frequent Patterns > FP-Growth

- **Inputs:** Basket columns and a minimum support value.
- **Process:** Runs FP-Growth to discover frequent itemsets.
- **Expected outputs:** Frequent itemsets with support and chart.

**App console**

```python
app.frequent_itemsets(min_support=0.1, algorithm="FP-Growth")
```

**df console**

```python
basket = df.drop(columns=["CustomerID"], errors="ignore").astype(bool)
```

#### 6.2.3. Eclat

**Menu path:** Transactional Analytics > Frequent Patterns > Eclat

- **Inputs:** Basket columns and a minimum support value.
- **Process:** Runs Eclat to discover frequent itemsets.
- **Expected outputs:** Frequent itemsets with support and chart.

**App console**

```python
app.frequent_itemsets(min_support=0.1, algorithm="Eclat")
```

**df console**

```python
basket = df.drop(columns=["CustomerID"], errors="ignore").astype(bool)
```

#### 6.2.4. Closed Itemsets

**Menu path:** Transactional Analytics > Frequent Patterns > Closed Itemsets

- **Inputs:** Basket columns and minimum support.
- **Process:** Finds closed frequent itemsets locally.
- **Expected outputs:** Closed itemsets with support.

**App console**

```python
app.closed_itemsets(min_support=0.1)
```

**df console**

```python
df.drop(columns=['CustomerID'], errors='ignore').astype(bool)
```

#### 6.2.5. Maximal Itemsets

**Menu path:** Transactional Analytics > Frequent Patterns > Maximal Itemsets

- **Inputs:** Basket columns and minimum support.
- **Process:** Finds maximal frequent itemsets locally.
- **Expected outputs:** Maximal itemsets with support.

**App console**

```python
app.maximal_itemsets(min_support=0.1)
```

**df console**

```python
df.drop(columns=['CustomerID'], errors='ignore').astype(bool)
```

### 6.3. Business Applications

#### 6.3.1. Market Basket Report

**Menu path:** Transactional Analytics > Business Applications > Market Basket Report

- **Inputs:** Minimum support and confidence.
- **Process:** Combines frequent itemsets and association rules.
- **Expected outputs:** Itemset table, rules table, and rule plot.

**App console**

```python
app.market_basket(min_support=0.1, min_confidence=0.5)
```

**df console**

```python
df.drop(columns=['CustomerID'], errors='ignore').astype(bool)
```

#### 6.3.2. RFM Segmentation

**Menu path:** Transactional Analytics > Business Applications > RFM Segmentation

- **Inputs:** Customer id, date, and spend columns.
- **Process:** Computes recency, frequency, monetary scores and segments.
- **Expected outputs:** RFM scores, segment summary, and segment chart.

**App console**

```python
app.rfm()
```

**df console**

```python
df.groupby('CustomerID')['Spend'].agg(['count', 'sum'])
```

#### 6.3.3. Sequential Purchase Patterns

**Menu path:** Transactional Analytics > Business Applications > Sequential Purchase Patterns

- **Inputs:** Customer id, sequence id, and item columns.
- **Process:** Runs a prefix-span style sequence miner.
- **Expected outputs:** Sequential patterns with support and bar chart.

**App console**

```python
app.sequential_patterns(min_support=0.2)
```

**df console**

```python
df.sort_values(['CustomerID', 'SequenceID'])
```

#### 6.3.4. Profit-Weighted Rules

**Menu path:** Transactional Analytics > Business Applications > Profit-Weighted Rules

- **Inputs:** Basket items, profit columns, support, and confidence.
- **Process:** Generates rules ranked by utility-weighted lift.
- **Expected outputs:** Profit-weighted rule report.

**App console**

```python
app.profit_weighted_rules(min_support=0.05, min_confidence=0.3)
```

**df console**

```python
df.filter(like='Profit').mean()
```

#### 6.3.5. Recommendation / Affinity Report

**Menu path:** Transactional Analytics > Business Applications > Recommendation / Affinity Report

- **Inputs:** Target item and rule thresholds.
- **Process:** Finds items that frequently co-occur with the target.
- **Expected outputs:** Recommendation table and affinity plot.

**App console**

```python
app.recommendations(min_support=0.05, min_confidence=0.3)
```

**df console**

```python
df.corr().sort_values('Item_A', ascending=False)
```

## 7. Plots

### 7.1. Main Menu

#### 7.1.1. Plot Builder

**Menu path:** Plots > Plot Builder

- **Inputs:** Plot type, variable roles, and styling options.
- **Process:** Opens one dialog with variable dropdowns, options, and a live report-style output.
- **Expected outputs:** Selected plot in a downloadable report window.

**App console**

```python
app.visualize('scatter', x='sales', y='profit', hue='region')
```

**df console**

```python
df.plot.scatter(x='sales', y='profit')
```

#### 7.1.2. Quick Plot (Auto)

**Menu path:** Plots > Quick Plot (Auto)

- **Inputs:** One, two, or three columns.
- **Process:** Infers histogram, bar, box, or scatter from column types.
- **Expected outputs:** Automatically chosen plot.

**App console**

```python
app.plot(['sales', 'profit'])
```

**df console**

```python
df[['sales', 'profit']].plot()
```

#### 7.1.3. Plot Help

**Menu path:** Plots > Plot Help

- **Inputs:** None.
- **Process:** Shows quick instructions and console examples.
- **Expected outputs:** Help dialog.

**App console**

```python
app.show_plot_help()
```

**df console**

```python
df.head()
```

### 7.2. Distribution

#### 7.2.1. Histogram

**Menu path:** Plots > Distribution > Histogram

- **Inputs:** A numeric value column and optional grouping column.
- **Process:** Builds the histogram with configurable bins, grouping, and palette.
- **Expected outputs:** Distribution plot in a report window.

**App console**

```python
app.visualize("histogram", x="x")
```

**df console**

```python
df["x"].plot(kind="hist")
```

#### 7.2.2. KDE / Density

**Menu path:** Plots > Distribution > KDE / Density

- **Inputs:** A numeric value column and optional grouping column.
- **Process:** Builds the kde / density with configurable bins, grouping, and palette.
- **Expected outputs:** Distribution plot in a report window.

**App console**

```python
app.visualize("kde", x="x")
```

**df console**

```python
sns.displot(df['x'], kde=True)
```

#### 7.2.3. Box Plot

**Menu path:** Plots > Distribution > Box Plot

- **Inputs:** A numeric value column and optional grouping column.
- **Process:** Builds the box plot with configurable bins, grouping, and palette.
- **Expected outputs:** Distribution plot in a report window.

**App console**

```python
app.visualize("box", x="x")
```

**df console**

```python
sns.displot(df['x'], kde=True)
```

#### 7.2.4. Violin Plot

**Menu path:** Plots > Distribution > Violin Plot

- **Inputs:** A numeric value column and optional grouping column.
- **Process:** Builds the violin plot with configurable bins, grouping, and palette.
- **Expected outputs:** Distribution plot in a report window.

**App console**

```python
app.visualize("violin", x="x")
```

**df console**

```python
sns.displot(df['x'], kde=True)
```

#### 7.2.5. Strip Plot

**Menu path:** Plots > Distribution > Strip Plot

- **Inputs:** A numeric value column and optional grouping column.
- **Process:** Builds the strip plot with configurable bins, grouping, and palette.
- **Expected outputs:** Distribution plot in a report window.

**App console**

```python
app.visualize("strip", x="x")
```

**df console**

```python
sns.displot(df['x'], kde=True)
```

### 7.3. Categorical

#### 7.3.1. Bar / Count Chart

**Menu path:** Plots > Categorical > Bar / Count Chart

- **Inputs:** Category column, optional value column, and optional grouping column.
- **Process:** Aggregates values and draws a bar / count chart.
- **Expected outputs:** Categorical comparison plot.

**App console**

```python
app.visualize("bar", x="category", y="value")
```

**df console**

```python
df.groupby("category")["value"].sum().plot(kind="bar")
```

#### 7.3.2. Horizontal Bar

**Menu path:** Plots > Categorical > Horizontal Bar

- **Inputs:** Category column, optional value column, and optional grouping column.
- **Process:** Aggregates values and draws a horizontal bar.
- **Expected outputs:** Categorical comparison plot.

**App console**

```python
app.visualize("horizontal_bar", x="category", y="value")
```

**df console**

```python
df.groupby("category")["value"].sum().plot(kind="bar")
```

#### 7.3.3. Pie Chart

**Menu path:** Plots > Categorical > Pie Chart

- **Inputs:** Category column, optional value column, and optional grouping column.
- **Process:** Aggregates values and draws a pie chart.
- **Expected outputs:** Categorical comparison plot.

**App console**

```python
app.visualize("pie", x="category", y="value")
```

**df console**

```python
df.groupby("category")["value"].sum().plot(kind="bar")
```

#### 7.3.4. Grouped Bar

**Menu path:** Plots > Categorical > Grouped Bar

- **Inputs:** Category column, optional value column, and optional grouping column.
- **Process:** Aggregates values and draws a grouped bar.
- **Expected outputs:** Categorical comparison plot.

**App console**

```python
app.visualize("grouped_bar", x="category", y="value"), hue="group"
```

**df console**

```python
df.groupby("category")["value"].sum().plot(kind="bar")
```

#### 7.3.5. Stacked Bar

**Menu path:** Plots > Categorical > Stacked Bar

- **Inputs:** Category column, optional value column, and optional grouping column.
- **Process:** Aggregates values and draws a stacked bar.
- **Expected outputs:** Categorical comparison plot.

**App console**

```python
app.visualize("stacked_bar", x="category", y="value"), hue="group"
```

**df console**

```python
df.groupby("category")["value"].sum().plot(kind="bar")
```

### 7.4. Relationships

#### 7.4.1. Scatter Plot

**Menu path:** Plots > Relationships > Scatter Plot

- **Inputs:** Numeric columns and optional size or group columns.
- **Process:** Draws the scatter plot to inspect relationships.
- **Expected outputs:** Relationship plot in a report window.

**App console**

```python
app.visualize("scatter", x="x", y="y")
```

**df console**

```python
df.plot.scatter(x="x", y="y")
```

#### 7.4.2. Bubble Chart

**Menu path:** Plots > Relationships > Bubble Chart

- **Inputs:** Numeric columns and optional size or group columns.
- **Process:** Draws the bubble chart to inspect relationships.
- **Expected outputs:** Relationship plot in a report window.

**App console**

```python
app.visualize("bubble", x="x", y="y")
```

**df console**

```python
df.plot.scatter(x="x", y="y")
```

#### 7.4.3. Line Plot

**Menu path:** Plots > Relationships > Line Plot

- **Inputs:** Numeric columns and optional size or group columns.
- **Process:** Draws the line plot to inspect relationships.
- **Expected outputs:** Relationship plot in a report window.

**App console**

```python
app.visualize("line", x="x", y="y")
```

**df console**

```python
df.plot.scatter(x="x", y="y")
```

#### 7.4.4. Area Plot

**Menu path:** Plots > Relationships > Area Plot

- **Inputs:** Numeric columns and optional size or group columns.
- **Process:** Draws the area plot to inspect relationships.
- **Expected outputs:** Relationship plot in a report window.

**App console**

```python
app.visualize("area", x="x", y="y")
```

**df console**

```python
df.plot.scatter(x="x", y="y")
```

#### 7.4.5. Correlation Heatmap

**Menu path:** Plots > Relationships > Correlation Heatmap

- **Inputs:** Numeric columns and optional size or group columns.
- **Process:** Draws the correlation heatmap to inspect relationships.
- **Expected outputs:** Relationship plot in a report window.

**App console**

```python
app.visualize("heatmap", columns=["x", "y"])
```

**df console**

```python
df.plot.scatter(x="x", y="y")
```

#### 7.4.6. Pair Plot

**Menu path:** Plots > Relationships > Pair Plot

- **Inputs:** Numeric columns and optional size or group columns.
- **Process:** Draws the pair plot to inspect relationships.
- **Expected outputs:** Relationship plot in a report window.

**App console**

```python
app.visualize("pairplot", columns=["x", "y"])
```

**df console**

```python
df.plot.scatter(x="x", y="y")
```

### 7.5. Business Analytics

#### 7.5.1. Pareto Chart

**Menu path:** Plots > Business Analytics > Pareto Chart

- **Inputs:** Category or time column plus a value column.
- **Process:** Builds a pareto chart with cumulative or rolling behavior.
- **Expected outputs:** Business analytics plot.

**App console**

```python
app.visualize("pareto", x="period", y="value")
```

**df console**

```python
df["value"].rolling(3).mean().plot()
```

#### 7.5.2. Time Series Line

**Menu path:** Plots > Business Analytics > Time Series Line

- **Inputs:** Category or time column plus a value column.
- **Process:** Builds a time series line with cumulative or rolling behavior.
- **Expected outputs:** Business analytics plot.

**App console**

```python
app.visualize("time_series", x="period", y="value")
```

**df console**

```python
df["value"].rolling(3).mean().plot()
```

#### 7.5.3. Moving Average

**Menu path:** Plots > Business Analytics > Moving Average

- **Inputs:** Category or time column plus a value column.
- **Process:** Builds a moving average with cumulative or rolling behavior.
- **Expected outputs:** Business analytics plot.

**App console**

```python
app.visualize("moving_average", x="period", y="value")
```

**df console**

```python
df["value"].rolling(3).mean().plot()
```

## 8. Python Console

### 8.1. Main Menu

#### 8.1.1. Open Console

**Menu path:** Python Console > Open Console

- **Inputs:** None.
- **Process:** Opens a persistent Python shell with app, df, pandas, numpy, matplotlib, seaborn, scipy, and sklearn loaded.
- **Expected outputs:** Interactive console window.

**App console**

```python
app.open_console()
```

**df console**

```python
df.head()
```

#### 8.1.2. df Object

**Menu path:** Python Console > df Object

- **Inputs:** The active DataFrame.
- **Process:** Represents the current data grid; assigning a new DataFrame syncs it back to the app after execution.
- **Expected outputs:** DataFrame-backed grid data.

**App console**

```python
app.get_df()
```

**df console**

```python
df.describe()
```

#### 8.1.3. app Object

**Menu path:** Python Console > app Object

- **Inputs:** The running SADSA application instance.
- **Process:** Exposes all data, analytics, simulation, reporting, and plotting methods.
- **Expected outputs:** Programmatic control of the app.

**App console**

```python
app.help_api()
```

**df console**

```python
app.set_df(df)
```

#### 8.1.4. New / Open Script

**Menu path:** Python Console > New / Open Script

- **Inputs:** A script path or new empty editor.
- **Process:** Creates or opens a script in the built-in editor.
- **Expected outputs:** Script editor tab.

**App console**

```python
app.open_console().open_script()
```

**df console**

```python
script = 'print(df.shape)'
```

#### 8.1.5. Save / Save As

**Menu path:** Python Console > Save / Save As

- **Inputs:** Script text and target path.
- **Process:** Writes the editor contents to a .py file.
- **Expected outputs:** Saved script file.

**App console**

```python
app.open_console().save_script()
```

**df console**

```python
Path('analysis.py').write_text('print(1)')
```

#### 8.1.6. Run Script

**Menu path:** Python Console > Run Script

- **Inputs:** Script text or selected lines.
- **Process:** Executes code with stdout and stderr captured in the console.
- **Expected outputs:** Console output and updated app state.

**App console**

```python
app.open_console().run_script()
```

**df console**

```python
exec('print(df.shape)')
```

## 9. Help

### 9.1. Main Menu

#### 9.1.1. Documentation

**Menu path:** Help > Documentation

- **Inputs:** None.
- **Process:** Opens this documentation page and offers TXT, Markdown, DOCX, and PDF downloads.
- **Expected outputs:** Readable documentation with app/author/company/license heading.

**App console**

```python
app.open_documentation()
```

**df console**

```python
print(app.documentation_metadata())
```

#### 9.1.2. Contact

**Menu path:** Help > Contact

- **Inputs:** None.
- **Process:** Shows company contact information.
- **Expected outputs:** Contact dialog.

**App console**

```python
app.show_message("Contact", "Amchik Solutions, India")
```

**df console**

```python
company = 'Amchik Solutions, India'
```

#### 9.1.3. Author

**Menu path:** Help > Author

- **Inputs:** None.
- **Process:** Shows author name, email, and profile link.
- **Expected outputs:** Author dialog.

**App console**

```python
app.show_message("Author", "Dr. Kamakshaiah Musunuru")
```

**df console**

```python
author = 'Dr. Kamakshaiah Musunuru'
```

#### 9.1.4. Version

**Menu path:** Help > Version

- **Inputs:** None.
- **Process:** Shows the application version.
- **Expected outputs:** Version dialog: Version 2.

**App console**

```python
app.show_message("Version", "Version 2")
```

**df console**

```python
version = "Version 2"
```

#### 9.1.5. About

**Menu path:** Help > About

- **Inputs:** None.
- **Process:** Shows a short description of SADSA.
- **Expected outputs:** About dialog.

**App console**

```python
app.show_message("About", "SADSA by Amchik Solutions, India")
```

**df console**

```python
about = 'SADSA'
```


## License and contact

License terms are shown in `File > View License`. Trial mode is available immediately; full-mode features require a valid key.

For support and licensing, contact dr.m.kamakshaiah@gmail.com. Author profile: https://github.com/Kamakshaiah.
