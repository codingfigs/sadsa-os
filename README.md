# Software Application for Data Science and Analytics (SADSA)

- **Version:** 2.1
- **Author:** Dr. Kamakshaiah Musunuru
- **Firm:** Amchik Solutions, India
- **Contact:** dr.m.kamakshaiah@gmail.com
- **Profile:** https://github.com/codingfigs

SADSA is a desktop application for data science and business analytics. It combines an editable spreadsheet-style data grid, descriptive and inferential statistics, machine learning, visualization, marketing analytics, domain-specific business simulations, advanced missing-data treatment, a Python console, and downloadable in-app documentation.

## Contents

- [Downloads](#downloads)
- [Features](#features)
- [Quick start](#quick-start)
- [Python console](#python-console)
- [Installation](#installation)
- [Running from source](#running-from-source)
- [Tests](#tests)
- [Project layout](#project-layout)
- [Documentation and release notes](#documentation-and-release-notes)
- [Support](#support)

## Downloads

| Asset | Description |
| --- | --- |
| [SADSA_v2.1_portable.zip](SADSA_v2.1_portable.zip) | Portable Windows bundle with EXE, README, license files, and sample data. |
| [SADSA_v2.1.exe](dist/SADSA_v2.1.exe) | Standalone Windows executable with embedded version metadata `2.1.0.0`. |
| [SADSA_v2.1_Setup.exe](installer_output/SADSA_v2.1_Setup.exe) | Windows installer with Start menu and optional desktop shortcut. |
| [install_linux.sh](install_linux.sh) | Linux setup script that creates a virtual environment and launches the app. |
| [requirements.txt](requirements.txt) | Pinned Python dependencies for running from source. |

All download filenames and version metadata are aligned with **Version 2.1**. See the [v2.1 release page](release_page/README.md) for release notes, checksums, and validation details.

## Features

### Data preparation

- Open, save, and clear CSV data.
- Edit the data grid through right-click actions or the `Edit` menu.
- Insert, edit, rename, copy, and delete rows or columns.
- Add blank columns with name-only prompts.
- Compute variables with pandas-style expressions and common math functions.
- Inspect a full dataset properties report with row/column counts, dimensions, missing and duplicate counts, column types, and all column names.
- Treat missing data with regression, random forest, KNN, iterative imputation, MCAR/MAR/MNAR simulation, and missingness significance tests.

### Analytics

- Descriptive statistics, inferential statistics, factor analysis, correlation, regression, and cluster analysis.
- Time-series simulations for stationary, non-stationary, seasonal, Holt-Winters, moving-average, random-walk, and custom series.
- Supervised learning with detailed prediction, confusion-matrix, classification-report, and metric outputs.
- Unsupervised learning with K-Means, hierarchical clustering, DBSCAN, Gaussian Mixture, and PCA.
- Seeded simulation generators for machine-learning and domain workflows.

### Marketing

The `Marketing` menu is organized into three practical workflows:

- **Transactional Analysis:** transaction matrices, Apriori/FP-Growth/Eclat frequent itemsets, closed and maximal itemsets, association rules, market basket reports, and sequential purchase patterns.
- **Customer Analytics:** RFM segmentation, customer lifetime value, and churn prediction.
- **Campaign & Pricing Analytics:** campaign response uplift, price elasticity, profit-weighted rules, and recommendation/affinity reports.

Each marketing technique has a matching demo generator and documented required columns.

### Domain analytics

SADSA includes 22 GUI-independent analytics techniques across:

- **Finance:** credit risk, fraud detection, Value at Risk, demand forecasting, fair lending, and more.
- **HRM:** attrition, pay equity, workforce forecasting, and people analytics.
- **IT & Operations:** predictive maintenance, incident priority, SLA breach risk, capacity forecasting, inventory reorder, supplier risk, and process cycle analysis.
- **International Business:** country segmentation, export forecasting, FX risk, trade anomaly detection, gravity models, and tariff impact analysis.

### Visualization and reporting

- 20 plot types through `Plots > Plot Builder` and `app.visualize(...)`.
- Chart windows with export support for generated reports and plots.
- Report windows with scrolling for long output, tables, and logs.
- Documentation exports in TXT, Markdown, DOCX, and PDF.

### Python console

- Interactive console with `app`, `df`, pandas, NumPy, Matplotlib, seaborn, SciPy, scikit-learn, and common path helpers preloaded.
- Script editor with open, save, and run actions.
- Programmatic access to the same analytics used by the GUI.

### Licensing

- The application can be used in trial mode.
- A valid full license unlocks report and plot download/export features.
- License information is available from `File > View License`.

## Quick start

1. Install from the setup file or extract the portable ZIP.
2. Open a CSV with `File > Open CSV File`, or generate demo data from `Marketing`, `Finance`, `HRM`, `IT & Operations`, `International Business`, `Machine Learning > Simulation Data`, or `Data Analytics > Time Series Analysis > Data Simulations`.
3. Edit data with the grid context menu or the `Edit` and `Transformations` menus.
4. Run an analysis from `Data Analytics`, `Machine Learning`, `Marketing`, or one of the domain menus.
5. Build a chart with `Plots > Plot Builder`.
6. Open `Help > Documentation` to view or download the complete manual.

## Python console

Open the console from the Python Console menu and automate the same workflows:

```python
app.head()
app.describe()
app.missing()
app.dataset_properties(show=True)

app.simulate_time_series(preset="holt_winters", rows=96)
app.visualize("scatter", x="sales", y="profit", hue="region")

app.generate_marketing_log(customers=300, transactions_per_customer=4)
app.transaction_matrix()
app.frequent_itemsets(algorithm="Apriori", min_support=0.2, show=True)
app.customer_lifetime_value(show=True)
app.churn_prediction(show=True)
app.campaign_response(show=True)
app.price_elasticity(show=True)

app.finance_demo("credit_risk", rows=200, apply=True)
app.credit_risk(show=True)
app.hrm_demo("attrition", rows=200, apply=True)
app.attrition(model="logistic", show=True)

app.open_documentation()
```

## Installation

### Windows portable

1. Extract `SADSA_v2.1_portable.zip`, or copy `SADSA_v2.1.exe` together with `license.json`, `license.txt`, and the `Data` folder.
2. Double-click `SADSA_v2.1.exe`.
3. Enter a valid license key when prompted, or continue in trial mode.

### Windows setup

1. Run `SADSA_v2.1_Setup.exe`.
2. Accept the default installation options for a standard setup.
3. Launch SADSA from the Start menu or the optional desktop shortcut.

The installer bundles the Python runtime, application files, icon, sample data, license files, and this README.

### Linux

```bash
chmod +x install_linux.sh
./install_linux.sh
```

Use `./install_linux.sh --install-only` to install dependencies without launching the app. Linux requires a desktop environment with Tkinter available. If Tkinter is missing, install the distribution package such as `python3-tk` on Debian/Ubuntu.

## Running from source

```bash
python -m pip install -r requirements.txt
python pyfda.py
```

Python 3.10+ with Tkinter is recommended. The pinned dependency versions are listed in `requirements.txt`.

## Tests

```bash
python -m py_compile pyfda.py domain_analytics.py tests\test_domain_analytics.py
python -m unittest discover -s tests
```

## Project layout

| Path | Purpose |
| --- | --- |
| `pyfda.py` | Complete Tkinter application source and public API. |
| `domain_analytics.py` | GUI-independent domain simulation and analysis engine. |
| `tests/test_domain_analytics.py` | Unit tests for domain analytics. |
| `pyfda.spec` | PyInstaller specification for the portable Windows build. |
| `file_version_info.txt` | Windows version resource for the portable EXE. |
| `SADSA_v2_setup.iss` | Inno Setup script for the Windows installer. |
| `install_linux.sh` | Linux setup and launcher. |
| `requirements.txt` | Python dependencies. |
| `Data/` | Sample CSV and Excel files. |
| `license.json`, `license.txt` | License metadata and agreement. |
| `release_page/README.md` | Version 2.1 release notes and asset verification. |

## Documentation and release notes

- Use `Help > Documentation` in the app to view or download the complete manual as TXT, Markdown, DOCX, or PDF.
- Read the [Version 2.1 release page](release_page/README.md) for release highlights, download assets, checksums, and build validation.

## Support

For support, licensing, and updates, contact dr.m.kamakshaiah@gmail.com or visit https://github.com/codingfigs.
