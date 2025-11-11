# sadsa-os
# SADSA — Software Application for Data Science and Analytics

**Version:** V-02.25.0.0.1  
**Author:** Dr. Kamakshaiah Musunuru  
**Contact:** contact@codingfigs.com  
**GitHub:** [SADSA-OS](https://github.com/codingfigs/sadsa-os)  

---

## 🧠 Overview

**SADSA (Software Application for Data Science and Analytics)** is a Python-based desktop application designed to simplify statistical analysis, machine learning, and data visualization for students, researchers, and data professionals.  

Built using **Python** for the GUI, SADSA provides a menu-driven interface for handling datasets, applying transformations, running advanced statistical tests, machine learning algorithms, and generating insightful plots — all without writing code.

---

## 💡 Features

### 📂 **File Operations**
- Open and import CSV datasets.
- Save processed data.
- Clear data from the workspace.
- Exit with ease.

### ✏️ **Data Editing**
- Rename columns.
- Compute new variables.
- Recode variables.
- Handle missing data.
- Set fixed values.

### 🔁 **Transformations**
- **Data Simulations** and synthetic data generation.
- **Multivariate Normal Distribution** sampling.
- **Data Decomposition**: Cholesky, QR, SVD, Eigen.
- **Data Standardization**: Min-Max, Z-Score, Decimal Scaling, Log & Log-Normal transformations.

### 📊 **Data Analytics**
- **Descriptive Statistics**: Frequency Tables, Summary Stats.
- **Inferential Statistics**: T-Test, Chi-Square, Normality Tests, ANOVA, MANOVA.
- **Factor Analysis**: Exploratory (EFA) & Confirmatory (CFA).
- **Correlation & Regression Analysis**.
- **Cluster Analysis**: K-Means, Hierarchical.
- **Time Series Analysis**: Stationarity, Decomposition, Holt-Winters, Moving Averages.

### 🤖 **Machine Learning Models**
- Logistic Regression.
- Decision Tree.
- Random Forest.
- Naive Bayes.
- K-Nearest Neighbors.
- Neural Network.
- Support Vector Machine (SVM).

### 📈 **Visualization**
- Easy access to plot generation with customizable options.

### 💻 **Python Console**
- Interactive Python shell integrated into the application.
- Direct access to loaded DataFrame via `df` variable.
- Execute custom Python code without leaving the application.
- Full access to all data analysis libraries (pandas, numpy, scipy, scikit-learn, etc.).
- Multi-line code support with Ctrl+Return execution.
- Call any app function directly from the console.
- Perfect for exploratory data analysis and custom operations.

### Console Methods & Features

#### Core Methods
- **`__init__(parent, app_instance, title="Python Console")`** - Initializes the Python console window with parent window reference and application instance.
- **`setup_ui()`** - Creates the console user interface with output area, input area, buttons, and information panel.
- **`setup_console_environment()`** - Sets up the Python environment with pre-imported libraries and displays welcome message with available objects and examples.

#### Code Execution Methods
- **`execute_code(event=None)`** - Executes Python code from input area with error handling and output capture via Ctrl+Return or Execute button.
- **`_import_module(module_name)`** - Imports a module by name and handles import errors gracefully.

#### Output & Display Methods
- **`print_output(text, tag="info")`** - Prints text to the output area with color coding (info=blue, error=red, success=green).
- **`clear_output()`** - Clears all text from the output display area.
- **`clear_input()`** - Clears all text from the code input area.
- **`show_help()`** - Displays a comprehensive help window with available objects, commands, and usage examples.

#### Pre-imported Libraries & Objects
- **`df`** - Current DataFrame from your loaded data, accessible for analysis and manipulation.
- **`app`** - Reference to the SADSA application instance for accessing app methods and data.
- **`pd`** - Pandas library for data manipulation and analysis.
- **`np`** - NumPy library for numerical computations and array operations.
- **`plt`** - Matplotlib.pyplot library for creating plots and visualizations.
- **`sns`** - Seaborn library for statistical data visualization.
- **`stats`** - SciPy.stats library for statistical functions and distributions.

### ℹ️ **Help & About**
- Contact Information.
- Author Bio.
- Versioning.
- Application Overview.

---

## ⚙️ Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/Kamakshaiah/SADSA.git
    ```

2. Navigate to the project folder:
    ```bash
    cd SADSA
    ```

3. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Run the application:
    ```bash
    python sadsa.py
    ```
5. Downoload Windows EXE file from https://drive.google.com/file/d/1l1mySjtGB6x7Q9CitPrU3xzhTj8ahRhe/view?usp=sharing
---

## � Python Console

SADSA includes an **interactive Python console** for advanced users who want to perform custom data analysis, write custom code, or explore data interactively.

### How to Access
- Navigate to **Help** → **Python Console** from the menu bar
- A new window will open with the Python shell

### Key Features
- **Full Python Environment**: Execute any Python code directly
- **DataFrame Access**: Access your loaded data via the `df` variable
- **App Integration**: Reference the application via `app` to call any app function
- **Pre-imported Libraries**: 
  - `pandas` (pd)
  - `numpy` (np)
  - `scipy`, `matplotlib`, `scikit-learn`
  - `statsmodels`, `networkx` and more

### Common Operations
```python
# View data
df.head()
df.describe()
df.info()

# Data manipulation
df['new_col'] = df['col1'] + df['col2']
df_filtered = df[df['col'] > 100]
df_grouped = df.groupby('category').sum()

# Statistical analysis
correlation = df.corr()
mean_val = df['col'].mean()
std_val = df['col'].std()

# Call app functions from console
app.perform_correlation_analysis()  # Performs correlation analysis on the dataset
app.data  # Access current data from app

# File Operations
app.open_csv()                       # Opens a file dialog to load CSV data
app.save_data()                      # Saves current data in memory
app.save_data_file()                 # Saves data to a CSV file
app.clear_data()                     # Clears all loaded data
app.display_data()                   # Refreshes the data display in the grid

# Edit Menu Operations
app.rename_columns()                 # Renames one or more columns in the dataset
app.compute_variable()               # Creates new computed variables from existing columns
app.recode_variable()                # Recodes/transforms existing variables
app.missing_data_treatment()         # Handles missing values using various methods
app.set_values()                     # Sets fixed values in the dataset

# Transformations Menu
app.data_simulations()               # Generates simulated/synthetic data
app.generate_multivariate_normal()   # Generates multivariate normal distribution data
app.perform_decomposition("cholesky") # Performs Cholesky decomposition on numeric data
app.perform_decomposition("qr")      # Performs QR decomposition on numeric data
app.perform_decomposition("svd")     # Performs Singular Value Decomposition (SVD)
app.perform_decomposition("eig")     # Performs Eigenvalue decomposition
app.perform_standardization("Min-Max") # Applies Min-Max scaling to data
app.perform_standardization("Z-Score") # Applies Z-Score standardization to data
app.perform_standardization("Decimal Scaling") # Applies Decimal Scaling transformation
app.perform_standardization("Log")   # Applies Log transformation to data
app.perform_standardization("Log-Normal") # Applies Log-Normal transformation

# Data Analytics - Descriptive Statistics
app.show_frequencies()               # Displays frequency tables for categorical variables
app.show_summary_statistics()        # Shows summary statistics (mean, std, min, max, etc.)
app.show_about_dataset()             # Displays dataset metadata (rows, columns, types, missing values)

# Data Analytics - Inferential Statistics
app.perform_ttest()                  # Performs t-test (one-sample, two-sample, paired)
app.perform_chisquare()              # Performs Chi-Square test for independence
app.perform_normality_tests()        # Tests for data normality (Shapiro-Wilk, Kolmogorov-Smirnov)
app.perform_anova()                  # Performs ANOVA (Analysis of Variance) test
app.perform_manova()                 # Performs MANOVA (Multivariate ANOVA) test

# Data Analytics - Factor Analysis
app.perform_efa()                    # Performs Exploratory Factor Analysis
app.perform_cfa()                    # Performs Confirmatory Factor Analysis

# Data Analytics - Correlation & Regression
app.perform_regression_analysis()    # Performs linear and multiple regression analysis

# Data Analytics - Cluster Analysis
app.perform_kmeans()                 # Performs K-Means clustering on the data
app.perform_hierarchical()           # Performs Hierarchical clustering on the data

# Data Analytics - Time Series Analysis
app.perform_stationarity_tests()     # Tests for stationarity in time series data
app.perform_seasonal_decomposition() # Decomposes time series into trend, seasonal, and residual
app.perform_holt_winters()           # Applies Holt-Winters exponential smoothing
app.perform_moving_averages()        # Calculates moving averages for time series smoothing

# Machine Learning Models
app.perform_logistic_regression()    # Builds logistic regression classification model
app.perform_decision_tree()          # Builds decision tree classification model
app.perform_random_forest()          # Builds random forest classification/regression model
app.perform_naive_bayes()            # Builds Naive Bayes classification model
app.perform_knn()                    # Builds K-Nearest Neighbors classification model
app.perform_svm()                    # Builds Support Vector Machine classification model
app.perform_neural_network()         # Builds neural network classification/regression model

# Visualization & Display
app.generate_plot()                  # Opens plot generation interface with various plot types
app.show_report_window(data, title)  # Displays data/results in a report window
app.show_message(title, message)     # Shows a message dialog with custom title and content

# Console & Help
app.open_python_console()            # Opens the interactive Python console window
```

### Multi-line Code
- Write multiple lines of code
- Press **Ctrl+Return** to execute
- Use proper Python indentation for blocks

---

## 📌 Dependencies
- `tkinter` — GUI Framework.
- `pandas` — Data manipulation.
- `numpy` — Numerical computations.
- `scipy` — Statistical functions.
- `scikit-learn` — Machine Learning.
- `matplotlib` — Plotting & Visualization.

---

## 🏆 About

SADSA is developed by **Amchik Solutions, India** as a comprehensive, intuitive, and accessible tool for data science education and applied research.

---

## 💬 Contact

For feedback, collaboration, or support, please reach out:  
**Email:** contact@codingfigs.com
