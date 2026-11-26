# Software Application for Data Science and Analytics (SADSA)

- **Version:** 2.1
- **Author:** Dr. Kamakshaiah Musunuru
- **Firm:** Amchik Solutions, India
- **Contact:** dr.m.kamakshaiah@gmail.com
- **Profile:** https://github.com/codingfigs

SADSA is a desktop application for data science and business analytics. It combines an editable data grid, descriptive and inferential statistics, machine learning, visualization, marketing analytics, domain-specific business simulations, advanced missing-data treatment, a Python console, and downloadable in-app documentation.

This `main` branch contains the source and support files needed to run SADSA on Linux. Windows installers, portable archives, PyInstaller files, and Inno Setup scripts are not tracked here.

## Contents

- [Repository files](#repository-files)
- [Linux installation](#linux-installation)
- [Running from source](#running-from-source)
- [Quick start](#quick-start)
- [Python console](#python-console)
- [Features](#features)
- [Documentation and license](#documentation-and-license)
- [Support](#support)

## Repository files

| Path | Purpose |
| --- | --- |
| `pyfda.py` | Complete Tkinter application source and public API. |
| `domain_analytics.py` | GUI-independent analytics engine used by `pyfda.py`. |
| `install_linux.sh` | Linux setup script; creates `.venv`, installs dependencies, and launches the app. |
| `requirements.txt` | Pinned Python dependencies. |
| `icon.ico` | Application icon used by the Linux desktop entry. |
| `README.md` | Repository guide. |

## Linux installation

```bash
chmod +x install_linux.sh
./install_linux.sh
```

The script:

1. Checks for Python 3 and Tkinter.
2. Creates a virtual environment in `.venv`.
3. Installs the pinned packages from `requirements.txt`.
4. Creates a user desktop entry when `~/.local/share/applications` exists.
5. Launches SADSA.

Install dependencies without launching the app:

```bash
./install_linux.sh --install-only
```

Linux requires a desktop environment with Tkinter available. If Tkinter is missing, install the distribution package such as `python3-tk` on Debian/Ubuntu.

## Running from source

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python pyfda.py
```

## Quick start

1. Launch the app with `./install_linux.sh` or `python pyfda.py`.
2. Open a CSV with `File > Open CSV File`, or generate demo data from `Marketing`, `Finance`, `HRM`, `IT & Operations`, `International Business`, `Machine Learning > Simulation Data`, or `Data Analytics > Time Series Analysis > Data Simulations`.
3. Edit data with the grid context menu or the `Edit` and `Transformations` menus.
4. Run analyses from `Data Analytics`, `Machine Learning`, `Marketing`, or one of the domain menus.
5. Build charts with `Plots > Plot Builder`.
6. Open `Help > Documentation` to view or download the complete manual.

## Python console

The built-in Python Console preloads `app`, `df`, pandas, NumPy, Matplotlib, seaborn, SciPy, scikit-learn, and common path helpers.

```python
app.head()
app.describe()
app.dataset_properties(show=True)

app.generate_marketing_log(customers=300, transactions_per_customer=4)
app.transaction_matrix()
app.frequent_itemsets(algorithm="Apriori", min_support=0.2, show=True)
app.customer_lifetime_value(show=True)
app.churn_prediction(show=True)
app.campaign_response(show=True)
app.price_elasticity(show=True)

app.simulate_time_series(preset="holt_winters", rows=96)
app.visualize("scatter", x="sales", y="profit", hue="region")
app.open_documentation()
```

## Features

- Editable CSV workflows with row/column insert, edit, rename, copy, and delete actions.
- Dataset properties report with row/column counts, dimensions, missing and duplicate counts, column types, and all column names.
- Descriptive and inferential statistics, factor analysis, correlation, regression, and cluster analysis.
- Time-series simulations and forecasting workflows.
- Supervised and unsupervised machine learning with seeded demo data.
- Marketing analytics for transactional patterns, customer value, churn, uplift, elasticity, RFM, utility rules, and recommendations.
- 22 domain analytics techniques across Finance, HRM, IT & Operations, and International Business.
- Advanced missing-data treatment with regression, random forest, KNN, iterative imputation, MCAR/MAR/MNAR simulation, and missingness significance tests.
- 20 configurable plot types through `Plots > Plot Builder` and `app.visualize(...)`.
- Downloadable documentation in TXT, Markdown, DOCX, and PDF.

## Documentation and license

Use `Help > Documentation` in the app to view or download the complete manual. The manual is generated from the current application menu structure, so paths and technique coverage stay aligned with the application.

SADSA can be used in trial mode. A valid full license unlocks report and plot download/export features. License information is available from `File > View License`. For source installs, activation keys can be supplied with the `SADSA_FULL_KEYS` environment variable; the local `license.json` file is created at runtime and is not tracked in this repository.

## Support

For support, licensing, and updates, contact dr.m.kamakshaiah@gmail.com or visit https://github.com/codingfigs.
