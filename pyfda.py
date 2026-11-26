import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, Toplevel
from tkinter import ttk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Ellipse
import statsmodels.api as sm
from statsmodels.multivariate.manova import MANOVA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller, kpss
from sklearn.cross_decomposition import CCA
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification, make_blobs, make_moons
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from factor_analyzer import FactorAnalyzer, calculate_bartlett_sphericity, calculate_kmo
from semopy import Model
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import json
from datetime import datetime, timedelta
import datetime as dt
import os
import ast
import codeop
import contextlib
import io
import math
import random
import re
import statistics
import sys
import textwrap
import traceback
from pathlib import Path
import domain_analytics

# Predefined keys for trial and full version
TRIAL_KEYS = {item.strip() for item in os.environ.get("SADSA_TRIAL_KEYS", "").split(",") if item.strip()}
FULL_KEYS = {item.strip() for item in os.environ.get("SADSA_FULL_KEYS", "").split(",") if item.strip()}

LICENSE_FILE = "license.json"
APP_NAME = "Software Application for Data Science and Analytics (SADSA)"
APP_SHORT_NAME = "SADSA"
APP_VERSION = "Version 2.1"
APP_COMPANY = "Amchik Solutions, India"
APP_AUTHOR = "Dr. Kamakshaiah Musunuru"
APP_EMAIL = "dr.m.kamakshaiah@gmail.com"
APP_PROFILE = "https://github.com/codingfigs"

PLOT_TYPES = {
    "auto": "Auto",
    "histogram": "Histogram",
    "kde": "KDE / Density",
    "box": "Box Plot",
    "violin": "Violin Plot",
    "strip": "Strip Plot",
    "bar": "Bar / Count Chart",
    "horizontal_bar": "Horizontal Bar",
    "pie": "Pie Chart",
    "grouped_bar": "Grouped Bar",
    "stacked_bar": "Stacked Bar",
    "scatter": "Scatter Plot",
    "bubble": "Bubble Chart",
    "line": "Line Plot",
    "area": "Area Plot",
    "heatmap": "Correlation Heatmap",
    "pairplot": "Pair Plot",
    "pareto": "Pareto Chart",
    "time_series": "Time Series Line",
    "moving_average": "Moving Average"
}

LICENSE_TEXT = """
LICENSE AGREEMENT FOR SOFTWARE APPLICATION FOR DATA SCIENCE AND ANALYTICS (SADSA)
Version: Version 2.1
Date: {activation_date}
Developer: Amchik Solutions, India

1. License Types
   - Trial Version: Limited features, no expiration.
   - Full Version: One-year validity from activation date, requires renewal after expiration.

2. License Activation & Expiration
   - License verification occurs at every startup.
   - Full licenses expire after one year and require renewal.

3. Restrictions
   - Users may not modify, distribute, or reverse-engineer the software.
   - License is non-transferable.

4. Disclaimer
   - Provided "as is" without warranties.
   - Developer is not liable for data loss, security vulnerabilities, or software issues.

5. Contact
   - Email: dr.m.kamakshaiah@gmail.com
   - GitHub: https://github.com/codingfigs
"""



class EditableCSVApp:
    def __init__(self, root):
        self.root = root
        self.root.title(f"{APP_SHORT_NAME} - {APP_VERSION}")
        self.root.state("zoomed")  # Maximize window on launch

        self.data = None
        self.file_path = None
        self.license_type = "TRIAL"  # Default to trial version
        self.console = None

        # Ask for license key
        self.check_license()


        # Create a menu bar
        self.menu_bar = tk.Menu(self.root)
        self.root.config(menu=self.menu_bar)

        # Add 'File' menu
        self.file_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="File", menu=self.file_menu)
        self.file_menu.add_command(label="Open CSV File", command=self.open_csv)
        self.file_menu.add_command(label="Save Data", command=self.save_data)
        self.file_menu.add_command(label="Save Data File", command=self.save_data_file)
        self.file_menu.add_command(label="Clear Data", command=self.clear_data)
        self.file_menu.add_command(label="View License", command=self.show_license)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit", command=self.exit_app)
        

        # Add 'Edit' menu
        self.edit_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Edit", menu=self.edit_menu)
        self.edit_menu.add_command(label="Rename Columns", command=self.rename_columns)
        self.edit_menu.add_command(label="Rename Column...", command=self.rename_selected_column)
        self.edit_menu.add_command(label="Edit Cell...", command=self.edit_selected_cell)
        self.edit_menu.add_command(label="Edit Row...", command=lambda: self.edit_row_dialog())
        self.edit_menu.add_command(label="Edit Column...", command=lambda: self.edit_column_dialog())
        self.insert_menu = tk.Menu(self.edit_menu, tearoff=0)
        self.edit_menu.add_cascade(label="Insert", menu=self.insert_menu)
        self.insert_menu.add_command(label="Row Above", command=lambda: self.insert_row_dialog(self._selected_row_index()))
        self.insert_menu.add_command(label="Row Below", command=lambda: self.insert_row_dialog(self._selected_row_index() + 1 if self._selected_row_index() is not None else None))
        self.insert_menu.add_command(label="Column Before", command=lambda: self.insert_column_dialog(self._active_column_index))
        self.insert_menu.add_command(label="Column After", command=lambda: self.insert_column_dialog(self._active_column_index + 1 if self._active_column_index is not None else None))
        self.delete_menu = tk.Menu(self.edit_menu, tearoff=0)
        self.edit_menu.add_cascade(label="Delete", menu=self.delete_menu)
        self.delete_menu.add_command(label="Selected Row", command=lambda: self.delete_row_dialog(self._selected_row_index()))
        self.delete_menu.add_command(label="Selected Column", command=lambda: self.delete_column_dialog(self._active_column_index))
        self.edit_menu.add_command(label="Copy Column Names", command=self.copy_column_names_dialog)
        self.edit_menu.add_separator()
        self.edit_menu.add_command(label="Compute Variable", command=self.compute_variable)
        self.edit_menu.add_command(label="Recode Variable", command=self.recode_variable)
        self.edit_menu.add_command(label="Set Values", command=self.set_values)        

        # Create Treeview for data grid
        self.tree_frame = ttk.Frame(self.root)
        self.tree_frame.pack(expand=True, fill="both")
        self.tree_frame.columnconfigure(0, weight=1)
        self.tree_frame.rowconfigure(0, weight=1)
        self.tree = ttk.Treeview(self.tree_frame, show="headings")
        self.tree_scroll_y = ttk.Scrollbar(self.tree_frame, orient="vertical", command=self.tree.yview)
        self.tree_scroll_x = ttk.Scrollbar(self.tree_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=self.tree_scroll_y.set, xscrollcommand=self.tree_scroll_x.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        self.tree_scroll_y.grid(row=0, column=1, sticky="ns")
        self.tree_scroll_x.grid(row=1, column=0, sticky="ew")
        self._active_column_index = 0
        self.tree.bind("<Button-1>", self._remember_tree_selection)
        self.tree.bind("<Double-1>", self.on_double_click)
        self.tree.bind("<Button-3>", self.show_right_click_menu)
        
        # Add 'Transformations' menu
        self.transformations_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Transformations", menu=self.transformations_menu)
        self.transformations_menu.add_command(label="Data Simulations", command=self.data_simulations)
        self.transformations_menu.add_command(label="Multivariate Normal Distribution", command=self.generate_multivariate_normal)
        self.missing_data_menu = tk.Menu(self.transformations_menu, tearoff=0)
        self.transformations_menu.add_cascade(label="Missing Data Treatment", menu=self.missing_data_menu)
        self.missing_data_menu.add_command(label="Basic Treatment...", command=self.missing_data_treatment)
        self.missing_data_menu.add_command(label="Advanced Imputation...", command=self.open_advanced_imputation)
        self.missing_data_menu.add_separator()
        self.missing_data_menu.add_command(label="Simulate Missing Data...", command=self.open_missing_data_simulation)
        self.missing_data_menu.add_command(label="Missingness Significance Test...", command=self.open_missingness_test)
        self.missing_data_menu.add_command(label="Missingness Report", command=self.missingness_report)

        # Add 'Data Decomposition' submenu
        self.decomposition_menu = tk.Menu(self.transformations_menu, tearoff=0)
        self.transformations_menu.add_cascade(label="Data Decomposition", menu=self.decomposition_menu)
        self.decomposition_menu.add_command(label="Cholesky Decomposition", command=lambda: self.perform_decomposition("cholesky"))
        self.decomposition_menu.add_command(label="QR Decomposition", command=lambda: self.perform_decomposition("qr"))
        self.decomposition_menu.add_command(label="SVD Decomposition", command=lambda: self.perform_decomposition("svd"))
        self.decomposition_menu.add_command(label="Eigen Decomposition", command=lambda: self.perform_decomposition("eig"))

        # Add 'Data Standardization' submenu
        self.standardization_menu = tk.Menu(self.transformations_menu, tearoff=0)
        self.transformations_menu.add_cascade(label="Data Standardization", menu=self.standardization_menu)
        self.standardization_menu.add_command(label="Min-Max Scaling", command=lambda: self.perform_standardization("Min-Max"))
        self.standardization_menu.add_command(label="Z-Score Standardization", command=lambda: self.perform_standardization("Z-Score"))
        self.standardization_menu.add_command(label="Decimal Scaling", command=lambda: self.perform_standardization("Decimal Scaling"))
        self.standardization_menu.add_command(label="Log Transformation", command=lambda: self.perform_standardization("Log"))
        self.standardization_menu.add_command(label="Log-Normal Transformation", command=lambda: self.perform_standardization("Log-Normal"))

        # Add 'Data Analytics' menu
        self.analytics_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Data Analytics", menu=self.analytics_menu)
        
        # Descriptive Statistics submenu
        self.descriptive_menu = tk.Menu(self.analytics_menu, tearoff=0)
        self.analytics_menu.add_cascade(label="Descriptive Statistics", menu=self.descriptive_menu)
        self.descriptive_menu.add_command(label="Frequency Tables", command=self.show_frequencies)
        self.descriptive_menu.add_command(label="Summary Statistics", command=self.show_summary_statistics)
        self.descriptive_menu.add_command(label="Dataset Properties", command=self.show_dataset_properties)

        # Inferential Statistics submenu
        self.inferential_menu = tk.Menu(self.analytics_menu, tearoff=0)
        self.analytics_menu.add_cascade(label="Inferential Statistics", menu=self.inferential_menu)
        self.inferential_menu.add_command(label="T-Test", command=self.perform_ttest)
        self.inferential_menu.add_command(label="Chi-Square Test", command=self.perform_chisquare)
        self.inferential_menu.add_command(label="Normality Tests", command=self.perform_normality_tests)
        self.inferential_menu.add_command(label="ANOVA", command=self.perform_anova)
        self.inferential_menu.add_command(label="MANOVA", command=self.perform_manova)

        # Add 'Factor Analysis' menu
        self.factor_analysis_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.analytics_menu.add_cascade(label="Factor Analysis", menu=self.factor_analysis_menu)
        self.factor_analysis_menu.add_command(label="Exploratory Factor Analysis", command=self.perform_efa)
        self.factor_analysis_menu.add_command(label="Confirmatory Factor Analysis", command=self.perform_cfa)

        # Correlation and Regression submenu
        self.corr_reg_menu = tk.Menu(self.analytics_menu, tearoff=0)
        self.analytics_menu.add_cascade(label="Correlation & Regression", menu=self.corr_reg_menu)
        self.corr_reg_menu.add_command(label="Correlation Analysis", command=self.perform_correlation_analysis)
        self.corr_reg_menu.add_command(label="Regression Analysis", command=self.perform_regression_analysis)

        # Cluster Analysis submenu
        self.cluster_menu = tk.Menu(self.analytics_menu, tearoff=0)
        self.analytics_menu.add_cascade(label="Cluster Analysis", menu=self.cluster_menu)
        self.cluster_menu.add_command(label="K-Means Clustering", command=self.perform_kmeans)
        self.cluster_menu.add_command(label="Hierarchical Clustering", command=self.perform_hierarchical)

        # Time Series Analysis submenu
        self.time_series_menu = tk.Menu(self.analytics_menu, tearoff=0)
        self.analytics_menu.add_cascade(label="Time Series Analysis", menu=self.time_series_menu)
        self.time_series_menu.add_command(label="Stationarity Tests", command=self.perform_stationarity_tests)
        self.time_series_menu.add_command(label="Seasonal Decomposition", command=self.perform_seasonal_decomposition)
        self.time_series_menu.add_command(label="Holt-Winters Method", command=self.perform_holt_winters)
        self.time_series_menu.add_command(label="Moving Averages", command=self.perform_moving_averages)
        self.time_series_menu.add_separator()
        self.time_series_simulation_menu = tk.Menu(self.time_series_menu, tearoff=0)
        self.time_series_menu.add_cascade(label="Data Simulations", menu=self.time_series_simulation_menu)
        for simulation_label, simulation_preset in (
            ("Simulate Stationary Series...", "stationary"),
            ("Simulate Non-Stationary Series...", "non_stationary"),
            ("Simulate Seasonal Series...", "seasonal"),
            ("Simulate Trend + Seasonal Series...", "holt_winters"),
            ("Simulate Moving-Average Series...", "moving_average"),
            ("Simulate Custom Series...", "custom")
        ):
            self.time_series_simulation_menu.add_command(label=simulation_label, command=lambda simulation_preset=simulation_preset: self.open_time_series_simulation(simulation_preset))

        self.ml_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Machine Learning", menu=self.ml_menu)
        self.ml_supervised_menu = tk.Menu(self.ml_menu, tearoff=0)
        self.ml_menu.add_cascade(label="Supervised Learning", menu=self.ml_supervised_menu)
        self.ml_supervised_menu.add_command(label="Logistic Regression", command=self.perform_logistic_regression)
        self.ml_supervised_menu.add_command(label="Decision Tree", command=self.perform_decision_tree)
        self.ml_supervised_menu.add_command(label="Random Forest", command=self.perform_random_forest)
        self.ml_supervised_menu.add_command(label="Naive Bayes", command=self.perform_naive_bayes)
        self.ml_supervised_menu.add_command(label="K-Nearest Neighbors", command=self.perform_knn)
        self.ml_supervised_menu.add_command(label="Neural Network", command=self.perform_neural_network)
        self.ml_supervised_menu.add_command(label="Support Vector Machine", command=self.perform_svm)
        self.ml_unsupervised_menu = tk.Menu(self.ml_menu, tearoff=0)
        self.ml_menu.add_cascade(label="Unsupervised Learning", menu=self.ml_unsupervised_menu)
        self.ml_unsupervised_menu.add_command(label="K-Means Clustering", command=self.perform_kmeans)
        self.ml_unsupervised_menu.add_command(label="Hierarchical Clustering", command=self.perform_hierarchical)
        self.ml_unsupervised_menu.add_command(label="DBSCAN Clustering", command=self.perform_dbscan)
        self.ml_unsupervised_menu.add_command(label="Gaussian Mixture", command=self.perform_gaussian_mixture)
        self.ml_unsupervised_menu.add_command(label="Principal Component Analysis", command=self.perform_pca)
        self.ml_simulation_menu = tk.Menu(self.ml_menu, tearoff=0)
        self.ml_menu.add_cascade(label="Simulation Data", menu=self.ml_simulation_menu)
        for technique_label, technique_key in self._ml_simulation_options():
            self.ml_simulation_menu.add_command(label=technique_label, command=lambda key=technique_key: self.generate_ml_demo(key))

        self.marketing_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Marketing", menu=self.marketing_menu)
        self.generation_menu = tk.Menu(self.marketing_menu, tearoff=0)
        self.marketing_menu.add_cascade(label="Generate Demo Data", menu=self.generation_menu)
        self.demo_transactional_menu = tk.Menu(self.generation_menu, tearoff=0)
        self.generation_menu.add_cascade(label="Transactional Data", menu=self.demo_transactional_menu)
        self.demo_transactional_menu.add_command(label="Basket Matrix", command=lambda: self.generate_basket_data("wide"))
        self.demo_transactional_menu.add_command(label="Customer Transaction Log", command=lambda: self.generate_basket_data("long"))
        self.demo_transactional_menu.add_command(label="Sequence Log", command=lambda: self.generate_basket_data("sequence"))
        self.demo_transactional_menu.add_command(label="Profit / Utility Basket", command=lambda: self.generate_basket_data("profit"))
        self.demo_customer_menu = tk.Menu(self.generation_menu, tearoff=0)
        self.generation_menu.add_cascade(label="Customer & Campaign Data", menu=self.demo_customer_menu)
        self.demo_customer_menu.add_command(label="Marketing Campaign Log", command=lambda: self.generate_basket_data("marketing"))
        self.generation_menu.add_separator()
        self.generation_menu.add_command(label="Clear Data", command=self.clear_data)
        self.transactional_analysis_menu = tk.Menu(self.marketing_menu, tearoff=0)
        self.marketing_menu.add_cascade(label="Transactional Analysis", menu=self.transactional_analysis_menu)
        self.transactional_analysis_menu.add_command(label="Transaction Matrix", command=self.perform_transaction_matrix)
        self.frequent_itemsets_menu = tk.Menu(self.transactional_analysis_menu, tearoff=0)
        self.transactional_analysis_menu.add_cascade(label="Frequent Itemsets", menu=self.frequent_itemsets_menu)
        self.frequent_itemsets_menu.add_command(label="Apriori", command=lambda: self.perform_association_mining("Apriori"))
        self.frequent_itemsets_menu.add_command(label="FP-Growth", command=lambda: self.perform_association_mining("FP-Growth"))
        self.frequent_itemsets_menu.add_command(label="Eclat", command=lambda: self.perform_association_mining("Eclat"))
        self.transactional_analysis_menu.add_separator()
        self.transactional_analysis_menu.add_command(label="Closed Itemsets", command=lambda: self.perform_itemset_variant("Closed"))
        self.transactional_analysis_menu.add_command(label="Maximal Itemsets", command=lambda: self.perform_itemset_variant("Maximal"))
        self.transactional_analysis_menu.add_command(label="Association Rules", command=self.perform_association_rules)
        self.transactional_analysis_menu.add_command(label="Market Basket Report", command=self.show_market_basket_report)
        self.transactional_analysis_menu.add_command(label="Sequential Purchase Patterns", command=self.perform_sequential_patterns)
        self.customer_analytics_menu = tk.Menu(self.marketing_menu, tearoff=0)
        self.marketing_menu.add_cascade(label="Customer Analytics", menu=self.customer_analytics_menu)
        self.customer_analytics_menu.add_command(label="RFM Segmentation", command=self.perform_rfm_segmentation)
        self.customer_analytics_menu.add_command(label="Customer Lifetime Value (CLV)", command=self.perform_clv_analysis)
        self.customer_analytics_menu.add_command(label="Churn Prediction", command=self.perform_churn_prediction)
        self.campaign_analytics_menu = tk.Menu(self.marketing_menu, tearoff=0)
        self.marketing_menu.add_cascade(label="Campaign & Pricing Analytics", menu=self.campaign_analytics_menu)
        self.campaign_analytics_menu.add_command(label="Campaign Response Uplift", command=self.perform_campaign_response_analysis)
        self.campaign_analytics_menu.add_command(label="Price Elasticity Analysis", command=self.perform_price_elasticity_analysis)
        self.campaign_analytics_menu.add_command(label="Profit-Weighted Rules", command=self.perform_profit_weighted_rules)
        self.campaign_analytics_menu.add_command(label="Recommendation / Affinity Report", command=self.perform_affinity_recommendations)
        self.pattern_menu = self.transactional_analysis_menu
        self.business_menu = self.campaign_analytics_menu

        self.finance_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Finance", menu=self.finance_menu)
        self.finance_demo_menu = tk.Menu(self.finance_menu, tearoff=0)
        self.finance_menu.add_cascade(label="Generate Demo Data", menu=self.finance_demo_menu)
        for label, preset in domain_analytics.DOMAIN_SIMULATIONS["finance"]:
            self.finance_demo_menu.add_command(label=label, command=lambda preset=preset: self.generate_domain_demo("finance", preset))
        self.finance_analysis_menu = tk.Menu(self.finance_menu, tearoff=0)
        self.finance_menu.add_cascade(label="Business Analytics", menu=self.finance_analysis_menu)
        for label, technique in domain_analytics.DOMAIN_ANALYSES["finance"]:
            self.finance_analysis_menu.add_command(label=label, command=lambda technique=technique: self.perform_domain_analysis("finance", technique))

        self.hrm_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="HRM", menu=self.hrm_menu)
        self.hrm_demo_menu = tk.Menu(self.hrm_menu, tearoff=0)
        self.hrm_menu.add_cascade(label="Generate Demo Data", menu=self.hrm_demo_menu)
        for label, preset in domain_analytics.DOMAIN_SIMULATIONS["hrm"]:
            self.hrm_demo_menu.add_command(label=label, command=lambda preset=preset: self.generate_domain_demo("hrm", preset))
        self.hrm_analysis_menu = tk.Menu(self.hrm_menu, tearoff=0)
        self.hrm_menu.add_cascade(label="People Analytics", menu=self.hrm_analysis_menu)
        for label, technique in domain_analytics.DOMAIN_ANALYSES["hrm"]:
            self.hrm_analysis_menu.add_command(label=label, command=lambda technique=technique: self.perform_domain_analysis("hrm", technique))

        self.it_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="IT & Operations", menu=self.it_menu)
        self.it_demo_menu = tk.Menu(self.it_menu, tearoff=0)
        self.it_menu.add_cascade(label="Generate Demo Data", menu=self.it_demo_menu)
        for label, preset in domain_analytics.DOMAIN_SIMULATIONS["it_operations"]:
            self.it_demo_menu.add_command(label=label, command=lambda preset=preset: self.generate_domain_demo("it_operations", preset))
        self.it_analysis_menu = tk.Menu(self.it_menu, tearoff=0)
        self.it_menu.add_cascade(label="Operations Analytics", menu=self.it_analysis_menu)
        for label, technique in domain_analytics.DOMAIN_ANALYSES["it_operations"]:
            self.it_analysis_menu.add_command(label=label, command=lambda technique=technique: self.perform_domain_analysis("it_operations", technique))

        self.international_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="International Business", menu=self.international_menu)
        self.international_demo_menu = tk.Menu(self.international_menu, tearoff=0)
        self.international_menu.add_cascade(label="Generate Demo Data", menu=self.international_demo_menu)
        for label, preset in domain_analytics.DOMAIN_SIMULATIONS["international"]:
            self.international_demo_menu.add_command(label=label, command=lambda preset=preset: self.generate_domain_demo("international", preset))
        self.international_analysis_menu = tk.Menu(self.international_menu, tearoff=0)
        self.international_menu.add_cascade(label="International Analytics", menu=self.international_analysis_menu)
        for label, technique in domain_analytics.DOMAIN_ANALYSES["international"]:
            self.international_analysis_menu.add_command(label=label, command=lambda technique=technique: self.perform_domain_analysis("international", technique))

        self.plots_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Plots", menu=self.plots_menu)
        self.plots_menu.add_command(label="Plot Builder...", command=self.open_plot_builder)
        self.plots_menu.add_command(label="Quick Plot (Auto)...", command=self.generate_plot)
        self.plots_menu.add_separator()
        self.distribution_menu = tk.Menu(self.plots_menu, tearoff=0)
        self.plots_menu.add_cascade(label="Distribution", menu=self.distribution_menu)
        self.distribution_menu.add_command(label="Histogram", command=lambda: self.open_plot_builder("histogram"))
        self.distribution_menu.add_command(label="KDE / Density", command=lambda: self.open_plot_builder("kde"))
        self.distribution_menu.add_command(label="Box Plot", command=lambda: self.open_plot_builder("box"))
        self.distribution_menu.add_command(label="Violin Plot", command=lambda: self.open_plot_builder("violin"))
        self.distribution_menu.add_command(label="Strip Plot", command=lambda: self.open_plot_builder("strip"))
        self.categorical_menu = tk.Menu(self.plots_menu, tearoff=0)
        self.plots_menu.add_cascade(label="Categorical", menu=self.categorical_menu)
        self.categorical_menu.add_command(label="Bar / Count Chart", command=lambda: self.open_plot_builder("bar"))
        self.categorical_menu.add_command(label="Horizontal Bar", command=lambda: self.open_plot_builder("horizontal_bar"))
        self.categorical_menu.add_command(label="Pie Chart", command=lambda: self.open_plot_builder("pie"))
        self.categorical_menu.add_command(label="Grouped Bar", command=lambda: self.open_plot_builder("grouped_bar"))
        self.categorical_menu.add_command(label="Stacked Bar", command=lambda: self.open_plot_builder("stacked_bar"))
        self.relationship_menu = tk.Menu(self.plots_menu, tearoff=0)
        self.plots_menu.add_cascade(label="Relationships", menu=self.relationship_menu)
        self.relationship_menu.add_command(label="Scatter Plot", command=lambda: self.open_plot_builder("scatter"))
        self.relationship_menu.add_command(label="Bubble Chart", command=lambda: self.open_plot_builder("bubble"))
        self.relationship_menu.add_command(label="Line Plot", command=lambda: self.open_plot_builder("line"))
        self.relationship_menu.add_command(label="Area Plot", command=lambda: self.open_plot_builder("area"))
        self.relationship_menu.add_command(label="Correlation Heatmap", command=lambda: self.open_plot_builder("heatmap"))
        self.relationship_menu.add_command(label="Pair Plot", command=lambda: self.open_plot_builder("pairplot"))
        self.business_menu_plots = tk.Menu(self.plots_menu, tearoff=0)
        self.plots_menu.add_cascade(label="Business Analytics", menu=self.business_menu_plots)
        self.business_menu_plots.add_command(label="Pareto Chart", command=lambda: self.open_plot_builder("pareto"))
        self.business_menu_plots.add_command(label="Time Series Line", command=lambda: self.open_plot_builder("time_series"))
        self.business_menu_plots.add_command(label="Moving Average", command=lambda: self.open_plot_builder("moving_average"))
        self.plots_menu.add_separator()
        self.plots_menu.add_command(label="Plot Help", command=self.show_plot_help)

        self.console_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Python Console", menu=self.console_menu)
        self.console_menu.add_command(label="Open Console", command=self.open_console)
        self.console_menu.add_separator()
        self.console_menu.add_command(label="New Script", command=lambda: self.open_console().new_script())
        self.console_menu.add_command(label="Open Script...", command=lambda: self.open_console().open_script())
        self.console_menu.add_command(label="Save Script", command=lambda: self.open_console().save_script())
        self.console_menu.add_command(label="Save Script As...", command=lambda: self.open_console().save_script_as())
        self.console_menu.add_separator()
        self.console_menu.add_command(label="Run Script", command=lambda: self.open_console().run_script())

        # Add 'Help' menu
        self.help_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Help", menu=self.help_menu)
        self.documentation_menu = tk.Menu(self.help_menu, tearoff=0)
        self.help_menu.add_cascade(label="Documentation", menu=self.documentation_menu)
        self.documentation_menu.add_command(label="Open Documentation", command=self.open_documentation)
        self.documentation_menu.add_separator()
        self.documentation_menu.add_command(label="Download TXT", command=lambda: self.download_documentation("txt"))
        self.documentation_menu.add_command(label="Download Markdown", command=lambda: self.download_documentation("md"))
        self.documentation_menu.add_command(label="Download DOCX", command=lambda: self.download_documentation("docx"))
        self.documentation_menu.add_command(label="Download PDF", command=lambda: self.download_documentation("pdf"))
        self.help_menu.add_separator()
        self.help_menu.add_command(label="Contact", command=lambda: self.show_message("Contact", f"{APP_COMPANY}\n{APP_EMAIL}\n{APP_PROFILE}"))
        self.help_menu.add_command(label="Author", command=lambda: self.show_message("Author", f"{APP_AUTHOR}\n{APP_EMAIL}\n{APP_PROFILE}"))
        self.help_menu.add_command(label="Version", command=lambda: self.show_message("Version", APP_VERSION))
        self.help_menu.add_command(label="About", command=lambda: self.show_message("About", f"{APP_NAME} {APP_VERSION} from {APP_COMPANY}."))

        # Create a frame for labels and button
        top_frame = tk.Frame(self.root)
        top_frame.pack(pady=10, fill='x')

        # Left-side label
        left_label = tk.Label(top_frame, text=f"{APP_NAME}\n{APP_COMPANY}\n{APP_VERSION}", font=("Arial", 10))
        left_label.pack(side="right", padx=20)

        # Open CSV Button in the center
        self.open_button = tk.Button(top_frame, text="Open CSV File", command=self.open_csv)
        self.open_button.pack(side="left", padx=20)
        self.generate_button = tk.Button(top_frame, text="Generate Demo Data", command=self.generate_basket_data)
        self.generate_button.pack(side="left", padx=20)
        self.plot_button = tk.Button(top_frame, text="Plots", command=self.open_plot_builder)
        self.plot_button.pack(side="left", padx=20)
        self.ml_simulation_button = tk.Button(top_frame, text="ML Simulations", command=self.open_ml_simulations)
        self.ml_simulation_button.pack(side="left", padx=20)
        self.marketing_button = tk.Button(top_frame, text="Marketing Report", command=self.show_market_basket_report)
        self.marketing_button.pack(side="left", padx=20)
        self.console_button = tk.Button(top_frame, text="Python Console", command=self.open_console)
        self.console_button.pack(side="left", padx=20)

        # Status label in the top-right corner of the main window
        self.status_label = tk.Label(self.root, text=self.get_status_message(), font=("Arial", 8), fg="red" if self.license_type == "TRIAL" else "blue")
        self.status_label.place(relx=0.999, rely=0.991, anchor='se')

    def show_license(self):
        license_window = Toplevel(self.root)
        license_window.title("License Agreement")
        license_window.geometry("600x400")
        
        text_widget = tk.Text(license_window, wrap=tk.WORD)
        text_widget.insert(tk.END, LICENSE_TEXT)
        text_widget.pack(expand=True, fill='both')    
        
    def get_status_message(self):
        return "Running on full mode." if self.license_type == "FULL" else "Running on trial mode."

    def _license_path(self):
        if getattr(sys, "frozen", False):
            return os.path.join(os.path.dirname(sys.executable), LICENSE_FILE)
        return LICENSE_FILE

    def check_license(self):
        license_path = self._license_path()
        if os.path.exists(license_path):
            try:
                with open(license_path, "r") as f:
                    license_data = json.load(f)
                key = license_data.get("key", "")
                activation_date = license_data.get("activation_date", "")

                if key in FULL_KEYS and activation_date:
                    activation_date = datetime.strptime(activation_date, "%Y-%m-%d")
                    expiration_date = activation_date + timedelta(days=365)
                    if datetime.now() > expiration_date:
                        messagebox.showwarning("License Expired", "Your full version license has expired. Please enter a new key.")
                        self.request_new_license()
                    else:
                        self.license_type = "FULL"
                else:
                    self.request_new_license()
            except (json.JSONDecodeError, KeyError):
                messagebox.showerror("Error", "License file is corrupted. Please enter a new key.")
                self.request_new_license()
        else:
            self.request_new_license()

    def request_new_license(self):
        key = simpledialog.askstring("License Key", "Enter your license key:")
        if key in FULL_KEYS:
            self.license_type = "FULL"
            with open(self._license_path(), "w") as f:
                json.dump({"key": key, "activation_date": datetime.now().strftime("%Y-%m-%d")}, f)
        else:
            self.license_type = "TRIAL"
            
    def open_csv(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not self.file_path:
            return
        
        self.data = pd.read_csv(self.file_path)
        self.display_data()
    
    def display_data(self):
        self.tree.delete(*self.tree.get_children())
        self.tree["columns"] = list(self.data.columns)
        self.tree["show"] = "headings"
        
        for col in self.data.columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100, anchor="center")
        
        for _, row in self.data.iterrows():
            self.tree.insert("", tk.END, values=list(row))
        self._sync_console_df()

    def clear_data(self):
        self.data = None
        self.tree.delete(*self.tree.get_children())
        self.tree["columns"] = []
        messagebox.showinfo("Success", "Data grid cleared successfully!")
    
    def _remember_tree_selection(self, event):
        col_id = self.tree.identify_column(event.x)
        if not col_id or col_id == "#0" or self.data is None:
            return
        index = int(col_id[1:]) - 1
        if 0 <= index < len(self.data.columns):
            self._active_column_index = index

    def _column_name_at(self, index):
        if self.data is None or index is None or not (0 <= int(index) < len(self.data.columns)):
            return None
        return self.data.columns[int(index)]

    def _selected_row_index(self):
        item = self.tree.focus()
        if not item:
            return None
        return self.tree.index(item)

    def on_double_click(self, event):
        if self.data is None:
            return
        row_id = self.tree.identify_row(event.y)
        col_id = self.tree.identify_column(event.x)
        if not row_id or not col_id or col_id == "#0":
            return
        col_index = int(col_id[1:]) - 1
        column = self._column_name_at(col_index)
        if column is None:
            return
        self._active_column_index = col_index
        row_index = self.tree.index(row_id)
        entry = ttk.Entry(self.tree)
        entry.place(x=event.x, y=event.y)
        current = self.tree.item(row_id, "values")[col_index]
        entry.insert(0, "" if current is None else str(current))
        entry.focus()

        def close_entry():
            if entry.winfo_exists():
                entry.destroy()

        def save_edit(_event=None):
            value = entry.get()
            close_entry()
            self.edit_cell(row_index, column, value)

        entry.bind("<Return>", save_edit)
        entry.bind("<FocusOut>", lambda _event: close_entry())

    def edit_cell_dialog(self, row_index, column):
        if self.data is None:
            messagebox.showerror("Error", "No data loaded!")
            return
        if row_index is None or column not in self.data.columns:
            messagebox.showerror("Error", "Select a cell first!")
            return
        current = self.data.iloc[int(row_index)][column]
        try:
            initial = "" if pd.isna(current) else str(current)
        except (TypeError, ValueError):
            initial = str(current)
        value = simpledialog.askstring("Edit Cell", f"New value for '{column}' (row {int(row_index) + 1}):", initialvalue=initial, allowblank=True)
        if value is None:
            return
        self.edit_cell(int(row_index), column, value)

    def insert_row_dialog(self, position=None):
        if self.data is None:
            messagebox.showerror("Error", "No data loaded!")
            return
        count = simpledialog.askinteger("Insert Row", "Number of rows to insert:", initialvalue=1, minvalue=1, maxvalue=1000)
        if count is None:
            return
        self.insert_row(position, count=count)

    def insert_column_dialog(self, position=None):
        if self.data is None:
            messagebox.showerror("Error", "No data loaded!")
            return
        default_name = self._unique_column_name(self.data.columns, f"Column_{len(self.data.columns) + 1}")
        name = simpledialog.askstring("Insert Column", "New column name:", initialvalue=default_name)
        if name is None or not str(name).strip():
            return
        inserted = self.insert_column(position, name=str(name).strip())
        created = inserted.columns[position if position is not None else len(inserted.columns) - 1]
        messagebox.showinfo("Success", f"Column '{created}' inserted successfully!")

    def edit_row_dialog(self, row_index=None):
        if self.data is None:
            messagebox.showerror("Error", "No data loaded!")
            return
        if row_index is None:
            row_index = self._selected_row_index()
        if row_index is None or not (0 <= int(row_index) < len(self.data)):
            messagebox.showinfo("Edit Row", "Select a row first!")
            return
        row_index = int(row_index)
        window = Toplevel(self.root)
        window.title(f"Edit Row {row_index + 1}")
        window.geometry("560x420")
        container = tk.Frame(window)
        container.pack(fill="both", expand=True, padx=10, pady=10)
        canvas = tk.Canvas(container)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        fields = tk.Frame(canvas)
        fields.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=fields, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        entries = {}
        for offset, column in enumerate(self.data.columns):
            value = self.data.iloc[row_index][column]
            try:
                initial = "" if pd.isna(value) else str(value)
            except (TypeError, ValueError):
                initial = str(value)
            tk.Label(fields, text=str(column), width=24, anchor="w").grid(row=offset, column=0, padx=6, pady=4, sticky="w")
            entry = tk.Entry(fields, width=28)
            entry.insert(0, initial)
            entry.grid(row=offset, column=1, padx=6, pady=4, sticky="ew")
            entries[column] = entry
        fields.columnconfigure(1, weight=1)

        def apply_changes():
            for column, entry in entries.items():
                converted = self._coerce_cell_value(entry.get(), self.data[column].dtype)
                self.data.at[self.data.index[row_index], column] = converted
            self.display_data()
            messagebox.showinfo("Success", f"Row {row_index + 1} updated successfully!")
            window.destroy()

        button_frame = tk.Frame(window)
        button_frame.pack(pady=8)
        tk.Button(button_frame, text="Apply", command=apply_changes).pack(side="left", padx=6)
        tk.Button(button_frame, text="Cancel", command=window.destroy).pack(side="left", padx=6)

    def edit_column_dialog(self, index=None):
        if self.data is None:
            messagebox.showerror("Error", "No data loaded!")
            return
        if index is None:
            index = self._active_column_index
        column = self._column_name_at(index)
        if column is None:
            messagebox.showinfo("Edit Column", "Select a column first!")
            return
        window = Toplevel(self.root)
        window.title(f"Edit Column '{column}'")
        window.geometry("520x460")
        tk.Label(window, text="Column name:").pack(anchor="w", padx=10, pady=(10, 2))
        name_entry = tk.Entry(window, width=48)
        name_entry.insert(0, str(column))
        name_entry.pack(padx=10, fill="x")
        tk.Label(window, text="Values (one per row):").pack(anchor="w", padx=10, pady=(10, 2))
        text_frame = tk.Frame(window)
        text_frame.pack(fill="both", expand=True, padx=10)
        text = tk.Text(text_frame, height=16, width=48)
        text_scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=text_scrollbar.set)
        text.pack(side="left", fill="both", expand=True)
        text_scrollbar.pack(side="right", fill="y")
        initial_values = []
        for value in self.data[column]:
            try:
                initial_values.append("" if pd.isna(value) else str(value))
            except (TypeError, ValueError):
                initial_values.append(str(value))
        text.insert("1.0", "\n".join(initial_values))

        def apply_changes():
            new_name = name_entry.get().strip() or str(column)
            lines = text.get("1.0", "end").splitlines()
            if len(lines) != len(self.data):
                messagebox.showerror("Edit Column", f"Provide exactly {len(self.data)} values, one per row.")
                return
            dtype = self.data[column].dtype
            self.data[column] = [self._coerce_cell_value(line, dtype) for line in lines]
            if new_name != str(column):
                self.data = self.data.rename(columns={column: new_name})
            self.display_data()
            messagebox.showinfo("Success", f"Column '{new_name}' updated successfully!")
            window.destroy()

        button_frame = tk.Frame(window)
        button_frame.pack(pady=8)
        tk.Button(button_frame, text="Apply", command=apply_changes).pack(side="left", padx=6)
        tk.Button(button_frame, text="Cancel", command=window.destroy).pack(side="left", padx=6)

    def edit_selected_cell(self):
        if self.data is None:
            messagebox.showerror("Error", "No data loaded!")
            return
        row_index = self._selected_row_index()
        if row_index is None:
            messagebox.showinfo("Edit Cell", "Select a row first!")
            return
        self.edit_cell_dialog(row_index, self._column_name_at(self._active_column_index))

    def rename_selected_column(self):
        if self.data is None:
            messagebox.showerror("Error", "No data loaded!")
            return
        old_name = self._column_name_at(self._active_column_index)
        if old_name is None:
            messagebox.showerror("Error", "Select a column first!")
            return
        new_name = simpledialog.askstring("Rename Column", f"Rename '{old_name}' to:", initialvalue=str(old_name))
        if new_name is None or not str(new_name).strip() or str(new_name) == str(old_name):
            return
        self.rename_column(old_name, str(new_name).strip())

    def delete_row_dialog(self, index):
        if index is None:
            messagebox.showinfo("Delete Row", "Select a row first!")
            return
        self.delete_row_at(int(index))
        messagebox.showinfo("Success", "Row deleted successfully!")

    def delete_column_dialog(self, index):
        name = self._column_name_at(index)
        if name is None:
            messagebox.showinfo("Delete Column", "Select a column first!")
            return
        self.delete_column(name)
        messagebox.showinfo("Success", f"Column '{name}' deleted successfully!")

    def copy_column_names_dialog(self):
        text = self.copy_column_names()
        messagebox.showinfo("Column Names", text)

    def delete_column(self, column_name):
        if self.data is None:
            raise ValueError("No data loaded.")
        if column_name in self.data.columns:
            self.data = self.data.drop(columns=[column_name])
            self.display_data()
        return self.data

    def delete_row(self, row_id):
        if self.data is None or not row_id:
            return self.data
        return self.delete_row_at(self.tree.index(row_id))

    def delete_row_at(self, index):
        if self.data is None:
            raise ValueError("No data loaded.")
        position = int(index)
        if not (0 <= position < len(self.data)):
            return self.data
        self.data = self.data.drop(self.data.index[position]).reset_index(drop=True)
        self.display_data()
        return self.data

    def show_right_click_menu(self, event):
        if self.data is None:
            return None
        item = self.tree.identify_row(event.y)
        col_id = self.tree.identify_column(event.x)
        if hasattr(self.tree, "identify_region"):
            region = self.tree.identify_region(event.x, event.y)
        else:
            region = "cell" if item else ("heading" if col_id else "")
        col_index = None
        if col_id and col_id != "#0":
            candidate = int(col_id[1:]) - 1
            if 0 <= candidate < len(self.data.columns):
                col_index = candidate
                self._active_column_index = candidate
        column = self._column_name_at(col_index)

        menu = tk.Menu(self.root, tearoff=0)
        if region in ("cell", "row") and item:
            row_index = self.tree.index(item)
            menu.add_command(label="Edit Cell...", command=lambda: self.edit_cell_dialog(row_index, column))
            menu.add_command(label="Edit Row...", command=lambda: self.edit_row_dialog(row_index))
            menu.add_separator()
            menu.add_command(label="Insert Row Above", command=lambda: self.insert_row_dialog(row_index))
            menu.add_command(label="Insert Row Below", command=lambda: self.insert_row_dialog(row_index + 1))
            menu.add_command(label="Delete Row", command=lambda: self.delete_row_dialog(row_index))
            if column is not None:
                menu.add_separator()
                menu.add_command(label="Insert Column Before", command=lambda: self.insert_column_dialog(col_index))
                menu.add_command(label="Insert Column After", command=lambda: self.insert_column_dialog(col_index + 1))
                menu.add_command(label="Rename Column...", command=lambda: self.rename_selected_column())
                menu.add_command(label="Edit Column...", command=lambda: self.edit_column_dialog(col_index))
                menu.add_command(label="Delete Column", command=lambda: self.delete_column_dialog(col_index))
        elif region == "heading" and column is not None:
            menu.add_command(label="Rename Column...", command=self.rename_selected_column)
            menu.add_command(label="Edit Column...", command=lambda: self.edit_column_dialog(col_index))
            menu.add_command(label="Insert Column Before", command=lambda: self.insert_column_dialog(col_index))
            menu.add_command(label="Insert Column After", command=lambda: self.insert_column_dialog(col_index + 1))
            menu.add_command(label="Delete Column", command=lambda: self.delete_column_dialog(col_index))
            menu.add_separator()
            menu.add_command(label="Copy Column Names", command=self.copy_column_names_dialog)
        else:
            menu.add_command(label="Append Row", command=lambda: self.insert_row_dialog(len(self.data)))
            menu.add_command(label="Append Column", command=lambda: self.insert_column_dialog(len(self.data.columns)))
            menu.add_command(label="Copy Column Names", command=self.copy_column_names_dialog)
        menu.post(event.x_root, event.y_root)
        return menu

    def rename_columns(self):
        if self.data is None:
            messagebox.showerror("Error", "No data loaded!")
            return

        rename_window = Toplevel(self.root)
        rename_window.title("Rename/Delete Columns")
        rename_window.geometry("500x500")

        frame = tk.Frame(rename_window)
        frame.pack(pady=10, padx=10, fill='both', expand=True)

        max_columns_per_row = 3  # Arrange items in 3 columns
        entries = {}

        for i, col in enumerate(self.data.columns):
            row_idx = i // max_columns_per_row
            col_idx = (i % max_columns_per_row) * 2

            tk.Label(frame, text=col).grid(row=row_idx, column=col_idx, padx=5, pady=5, sticky="w")
            entry = tk.Entry(frame)
            entry.grid(row=row_idx, column=col_idx + 1, padx=5, pady=5)
            entry.insert(0, col)
            entries[col] = entry

        def apply_changes():
            new_column_names = {col: entry.get() if entry.get().strip() else col for col, entry in entries.items()}
            self.data.rename(columns=new_column_names, inplace=True)
            self.display_data()
            messagebox.showinfo("Success", "Column names updated successfully!")
            rename_window.destroy()

        apply_button = tk.Button(rename_window, text="Apply", command=apply_changes)
        apply_button.pack(pady=10)
        
    def set_values(self):
        if self.data is None:
            messagebox.showerror("Error", "No data loaded!")
            return

        set_values_window = Toplevel(self.root)
        set_values_window.title("Set Values")
        set_values_window.geometry("400x400")

        tk.Label(set_values_window, text="Select Variable:").pack()
        column_var = tk.StringVar(set_values_window)
        column_dropdown = ttk.Combobox(set_values_window, textvariable=column_var, values=list(self.data.columns))
        column_dropdown.pack()

        def process_set_values():
            selected_col = column_var.get()
            if selected_col not in self.data.columns:
                messagebox.showerror("Error", "Invalid column selection!")
                return
            
            unique_values = self.data[selected_col].dropna().unique()
            value_map = {}
            
            for value in unique_values:
                new_label = simpledialog.askstring("Set Values", f"Enter name for {value}:")
                value_map[value] = new_label
            
            new_col_name = selected_col + "_Labeled"
            self.data[new_col_name] = self.data[selected_col].map(value_map)
            
            self.display_data()
            messagebox.showinfo("Success", "Values set successfully!")
            set_values_window.destroy()
        
        apply_button = tk.Button(set_values_window, text="Apply", command=process_set_values)
        apply_button.pack()

    def save_data(self):
        if self.data is None:
            messagebox.showerror("Error", "No data loaded!")
            return
        
        new_data = []
        for row_id in self.tree.get_children():
            new_data.append(self.tree.item(row_id, "values"))
        
        self.data = pd.DataFrame(new_data, columns=self.data.columns)
        messagebox.showinfo("Success", "Data saved successfully in the grid!")
    
    def save_data_file(self):
        if self.data is None:
            messagebox.showerror("Error", "No data to save!")
            return
        
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
        if file_path:
            self.data.to_csv(file_path, index=False)
            messagebox.showinfo("Success", "Data saved successfully!")
    
    def exit_app(self):
        self.root.quit()

    def compute_variable(self):
        if self.data is None:
            messagebox.showerror("Error", "No data loaded!")
            return
        window = Toplevel(self.root)
        window.title("Compute Variable")
        window.geometry("620x260")
        tk.Label(window, text="Target column (existing name to overwrite, or a new name):").pack(anchor="w", padx=12, pady=(12, 2))
        target_var = tk.StringVar(window)
        active = self._active_column_index
        if active is not None and 0 <= active < len(self.data.columns):
            target_var.set(str(self.data.columns[active]))
        target_box = ttk.Combobox(window, textvariable=target_var, values=[str(column) for column in self.data.columns], width=70)
        target_box.pack(padx=12, fill="x")
        tk.Label(window, text="Expression (e.g., Age * 2, log(Income), Price * Qty):").pack(anchor="w", padx=12, pady=(12, 2))
        expression_entry = tk.Entry(window, width=72)
        expression_entry.pack(padx=12, fill="x")
        tk.Label(window, text="Available functions: log, log10, log2, sqrt, exp, abs, round, floor, ceil, sin, cos, tan, mean, median, min, max, sum").pack(anchor="w", padx=12, pady=(10, 2))

        def apply_changes():
            target = target_var.get().strip()
            expression = expression_entry.get().strip()
            if not target:
                messagebox.showerror("Error", "Enter a target column name!")
                return
            if target not in [str(column) for column in self.data.columns] and not messagebox.askyesno(
                "Create Column", f"Column '{target}' does not exist. Create it with the computed values?"
            ):
                return
            try:
                self.compute_column(target, expression)
            except Exception as error:
                messagebox.showerror("Error", str(error))
                return
            messagebox.showinfo("Success", f"Computed values assigned to '{target}' successfully!")
            window.destroy()

        button_frame = tk.Frame(window)
        button_frame.pack(pady=12)
        tk.Button(button_frame, text="Apply", command=apply_changes).pack(side="left", padx=8)
        tk.Button(button_frame, text="Cancel", command=window.destroy).pack(side="left", padx=8)

    def recode_variable(self):
        if self.data is None:
            messagebox.showerror("Error", "No data loaded!")
            return

        recode_window = Toplevel(self.root)
        recode_window.title("Recode Variable")
        recode_window.geometry("400x400")

        column_label = tk.Label(recode_window, text="Select Variable:")
        column_label.pack()
        column_var = tk.StringVar(recode_window)
        column_dropdown = ttk.Combobox(recode_window, textvariable=column_var, values=list(self.data.columns))
        column_dropdown.pack()

        def process_recode():
            selected_col = column_var.get()
            if selected_col not in self.data.columns:
                messagebox.showerror("Error", "Invalid column selection!")
                return
            
            if pd.api.types.is_numeric_dtype(self.data[selected_col]):
                min_value = float(simpledialog.askstring("Numeric Recode", "Enter minimum value:"))
                max_value = float(simpledialog.askstring("Numeric Recode", "Enter maximum value:"))
                num_categories = int(simpledialog.askstring("Numeric Recode", "Enter number of categories:"))
                
                intervals = np.linspace(min_value, max_value, num_categories + 1)
                category_names = [simpledialog.askstring("Category Name", f"Enter name for category {i+1} ({intervals[i]:.2f} - {intervals[i+1]:.2f}):") 
                                  for i in range(num_categories)]
                
                self.data[selected_col + "_Recoded"] = pd.cut(self.data[selected_col], bins=intervals, labels=category_names, include_lowest=True)
            else:
                unique_values = self.data[selected_col].unique()
                recode_dict = {}
                for value in unique_values:
                    new_value = simpledialog.askstring("Recode Variable", f"Enter numeric value for '{value}':")
                    recode_dict[value] = new_value
                
                self.data[selected_col + "_Recoded"] = self.data[selected_col].replace(recode_dict)
            
            self.display_data()
            messagebox.showinfo("Success", "Variable recoded successfully!")
            recode_window.destroy()
        
        apply_btn = tk.Button(recode_window, text="Apply Recode", command=process_recode)
        apply_btn.pack()

    
    def missing_data_treatment(self):
        if self.data is None:
            messagebox.showerror("Error", "No data loaded!")
            return

        missing_window = Toplevel(self.root)
        missing_window.title("Missing Data Treatment")
        missing_window.geometry("400x300")

        tk.Label(missing_window, text="Select Variable:").pack()
        column_var = tk.StringVar(missing_window)
        column_dropdown = ttk.Combobox(missing_window, textvariable=column_var, values=list(self.data.columns))
        column_dropdown.pack()

        tk.Label(missing_window, text="Select Method:").pack()
        method_var = tk.StringVar(missing_window)
        method_dropdown = ttk.Combobox(missing_window, textvariable=method_var, 
                                       values=["Mean", "Median", "Mode", "Drop Rows", "Drop Column"])
        method_dropdown.pack()

        def apply_treatment():
            selected_col = column_var.get()
            method = method_var.get()
            
            if selected_col not in self.data.columns:
                messagebox.showerror("Error", "Invalid column selection!")
                return

            if method == "Mean":
                self.data[selected_col].fillna(self.data[selected_col].mean(), inplace=True)
            elif method == "Median":
                self.data[selected_col].fillna(self.data[selected_col].median(), inplace=True)
            elif method == "Mode":
                self.data[selected_col].fillna(self.data[selected_col].mode().iloc[0], inplace=True)
            elif method == "Drop Rows":
                self.data.dropna(subset=[selected_col], inplace=True)
            elif method == "Drop Column":
                self.data.drop(columns=[selected_col], inplace=True)
            else:
                messagebox.showerror("Error", "Invalid method selected!")
                return
            
            self.display_data()
            messagebox.showinfo("Success", "Missing data treated successfully!")
            missing_window.destroy()
        
        tk.Button(missing_window, text="Apply", command=apply_treatment).pack()

    def _unique_simulation_name(self, base, used):
        candidate = base
        suffix = 1
        while candidate in used:
            candidate = f"{base}_{suffix}"
            suffix += 1
        used.add(candidate)
        return candidate

    def _infer_data_simulation_defaults(self):
        used_names = set(self.data.columns) if self.data is not None else set()
        rows = len(self.data) if self.data is not None and len(self.data) else 100
        specifications = []
        if self.data is not None:
            for column in self.data.columns:
                if len(specifications) >= 3:
                    break
                series = self.data[column].dropna()
                if series.empty:
                    continue
                name = self._unique_simulation_name(f"{column}_Sim", used_names)
                if pd.api.types.is_numeric_dtype(series):
                    if pd.api.types.is_integer_dtype(series):
                        low = int(series.min())
                        high = int(series.max()) + 1
                        if high <= low:
                            high = low + 1
                        specifications.append(f"{name} | int | {low}, {high}")
                    else:
                        low = float(series.min())
                        high = float(series.max())
                        if high <= low:
                            high = low + 1
                        specifications.append(f"{name} | float | {low:.3f}, {high:.3f}")
                else:
                    categories = list(pd.unique(series.astype(str)))[:5]
                    if not categories:
                        categories = ["A", "B", "C"]
                    specifications.append(f"{name} | category | {', '.join(categories)}")
        fallback = [
            ("int", "1, 100"),
            ("float", "0.000, 1.000"),
            ("category", "A, B, C")
        ]
        for index, (kind, detail) in enumerate(fallback):
            if len(specifications) >= 3:
                break
            name = self._unique_simulation_name(f"Sim_{kind.title()}", used_names)
            specifications.append(f"{name} | {kind} | {detail}")
        return {"rows": rows, "seed": 42, "specifications": "\n".join(specifications[:3])}

    def _parse_covariance_text(self, text, dimension):
        rows = [row for row in re.split(r"[;\n]+", str(text).strip()) if row.strip()]
        if len(rows) == 1 and dimension > 1:
            values = [value for value in re.split(r"[,\s]+", rows[0].strip()) if value]
            if len(values) == dimension ** 2:
                rows = [", ".join(values[index * dimension:(index + 1) * dimension]) for index in range(dimension)]
        matrix_rows = []
        for row in rows:
            values = [value for value in re.split(r"[,\s]+", row.strip()) if value]
            matrix_rows.append([float(value) for value in values])
        matrix = np.asarray(matrix_rows, dtype=float)
        if matrix.shape != (dimension, dimension):
            raise ValueError(f"Covariance matrix must be {dimension}x{dimension}.")
        return matrix

    def _infer_multivariate_normal_defaults(self):
        samples = len(self.data) if self.data is not None and len(self.data) else 100
        used_names = set(self.data.columns) if self.data is not None else set()
        numeric = self.data.select_dtypes(include=[np.number]) if self.data is not None else pd.DataFrame()
        if numeric.shape[1] > 0:
            selected = list(numeric.columns[:3])
            values = numeric[selected].dropna()
            names = [self._unique_simulation_name(f"{column}_MVN", used_names) for column in selected]
            means = values.mean().to_numpy(dtype=float)
            covariance = np.cov(values, rowvar=False, bias=True)
            if covariance.ndim == 0:
                covariance = np.asarray([[float(covariance)]])
            if len(selected) == 1:
                covariance = np.asarray([[float(values[selected[0]].var(ddof=0))]])
        else:
            names = [self._unique_simulation_name(f"MVN_{index + 1}", used_names) for index in range(3)]
            means = np.zeros(3)
            covariance = np.eye(3)
        covariance_text = "; ".join(", ".join(f"{value:.3f}" for value in row) for row in covariance)
        return {
            "names": ", ".join(names),
            "means": ", ".join(f"{value:.3f}" for value in means),
            "covariance": covariance_text,
            "samples": samples,
            "seed": 42
        }

    def data_simulations(self):
        defaults = self._infer_data_simulation_defaults()
        dialog = TransformationSimulationDialog(self.root, self, "data", defaults)
        dialog.lift()
        return dialog

    def generate_multivariate_normal(self):
        defaults = self._infer_multivariate_normal_defaults()
        dialog = TransformationSimulationDialog(self.root, self, "mvn", defaults)
        dialog.lift()
        return dialog

    def _ml_simulation_options(self):
        return [
            ("Logistic Regression Demo", "logistic"),
            ("Decision Tree Demo", "decision_tree"),
            ("Random Forest Demo", "random_forest"),
            ("Naive Bayes Demo", "naive_bayes"),
            ("K-Nearest Neighbors Demo", "knn"),
            ("Support Vector Machine Demo", "svm"),
            ("Neural Network Demo", "neural_network"),
            ("K-Means Demo", "kmeans"),
            ("Hierarchical Clustering Demo", "hierarchical"),
            ("DBSCAN Demo", "dbscan"),
            ("Gaussian Mixture Demo", "gaussian_mixture"),
            ("PCA Demo", "pca")
        ]

    def generate_ml_demo(self, technique="logistic", rows=160, seed=42, apply=True):
        key = str(technique).strip().lower().replace(" ", "_").replace("-", "_")
        aliases = {
            "logisticregression": "logistic", "lr": "logistic", "dt": "decision_tree",
            "randomforest": "random_forest", "rf": "random_forest", "nb": "naive_bayes",
            "knearestneighbors": "knn", "support_vector_machine": "svm", "svc": "svm",
            "neuralnetwork": "neural_network", "mlp": "neural_network", "kmeans": "kmeans",
            "hierarchicalclustering": "hierarchical", "gaussianmixture": "gaussian_mixture",
            "principal_component_analysis": "pca"
        }
        key = aliases.get(key, key)
        valid_keys = {option[1] for option in self._ml_simulation_options()}
        if key not in valid_keys:
            raise ValueError(f"Unknown ML simulation '{technique}'. Use one of: {', '.join(sorted(valid_keys))}")
        rows = max(20, int(rows))
        random_state = int(seed)
        target = None
        if key == "logistic":
            values, labels = make_classification(n_samples=rows, n_features=3, n_informative=2, n_redundant=0, class_sep=1.8, flip_y=0.02, random_state=random_state)
        elif key == "decision_tree":
            values, labels = make_classification(n_samples=rows, n_features=4, n_informative=3, n_redundant=0, n_clusters_per_class=2, flip_y=0.04, random_state=random_state)
        elif key == "random_forest":
            values, labels = make_classification(n_samples=rows, n_features=4, n_informative=4, n_redundant=0, flip_y=0.06, random_state=random_state)
        elif key == "naive_bayes":
            values, labels = make_blobs(n_samples=rows, centers=2, n_features=3, cluster_std=1.6, random_state=random_state)
        elif key == "knn":
            values, labels = make_blobs(n_samples=rows, centers=3, n_features=3, cluster_std=1.1, random_state=random_state)
        elif key == "svm":
            values, labels = make_classification(n_samples=rows, n_features=2, n_informative=2, n_redundant=0, class_sep=2.0, flip_y=0.02, random_state=random_state)
        elif key == "neural_network":
            values, labels = make_moons(n_samples=rows, noise=0.18, random_state=random_state)
        elif key == "dbscan":
            values, labels = make_moons(n_samples=rows, noise=0.06, random_state=random_state)
            target = labels
        elif key == "pca":
            rng = np.random.default_rng(random_state)
            covariance = np.array([[2.5, 1.8, 1.2, 0.4], [1.8, 2.0, 0.9, 0.3], [1.2, 0.9, 1.6, 0.2], [0.4, 0.3, 0.2, 1.0]])
            values = rng.multivariate_normal([0, 1, -1, 2], covariance, rows)
        else:
            centers = 3 if key in ("kmeans", "hierarchical") else 4
            values, labels = make_blobs(n_samples=rows, centers=centers, n_features=3, cluster_std=[0.7, 0.9, 1.1, 1.3][:centers], random_state=random_state)
            target = labels
        frame = pd.DataFrame(values, columns=[f"Feature_{index + 1}" for index in range(values.shape[1])])
        if key not in ("pca",):
            frame.insert(0, "Target", [f"Class_{label}" for label in (labels if target is None else target)])
        frame.attrs["ML Simulation"] = key
        if apply:
            self.set_df(frame)
            messagebox.showinfo("Success", f"{key.replace('_', ' ').title()} data generated with {len(frame)} rows.")
        return frame

    def time_series_simulation_defaults(self, preset="custom"):
        frame = self.data
        columns = list(frame.columns) if frame is not None else []
        numeric_columns = [column for column in columns if pd.api.types.is_numeric_dtype(frame[column])]
        time_keywords = ("date", "time", "period", "month", "year", "quarter", "week")
        value_keywords = ("sales", "value", "revenue", "amount", "demand", "price", "units")
        time_candidates = [column for column in columns if any(keyword in str(column).lower() for keyword in time_keywords) or pd.api.types.is_datetime64_any_dtype(frame[column])]
        time_column = time_candidates[0] if time_candidates else (columns[0] if columns else "Date")
        value_candidates = [column for column in numeric_columns if column != time_column]
        preferred_values = [column for column in value_candidates if any(keyword in str(column).lower() for keyword in value_keywords)]
        value_column = preferred_values[0] if preferred_values else (value_candidates[0] if value_candidates else (columns[-1] if columns else "Sales"))
        rows = len(frame) if frame is not None and len(frame) > 0 else 96
        rows = max(rows, 8)
        period = 12 if rows >= 24 else max(2, rows // 2)
        start = "2020-01-01"
        frequency = "M"
        time_mode = "date"
        base = 100.0
        standard_deviation = 1.0
        slope = 0.0
        if time_column in columns:
            series = frame[time_column].dropna()
            if not series.empty:
                first_value = series.iloc[0]
                if pd.api.types.is_datetime64_any_dtype(series) or isinstance(first_value, (pd.Timestamp, dt.datetime, dt.date)):
                    start = pd.to_datetime(first_value).strftime("%Y-%m-%d")
                    parsed = pd.to_datetime(series, errors="coerce").dropna().sort_values()
                    if len(parsed) > 1:
                        gap_days = float(parsed.diff().dropna().dt.total_seconds().abs().median() / 86400)
                        if gap_days < 1:
                            frequency = "H"
                        elif gap_days <= 7:
                            frequency = "W"
                        elif gap_days <= 31:
                            frequency = "M"
                        elif gap_days <= 90:
                            frequency = "Q"
                        else:
                            frequency = "Y"
                elif pd.api.types.is_numeric_dtype(series):
                    start = str(first_value)
                    time_mode = "numeric"
        if value_column in numeric_columns and frame is not None:
            values = pd.to_numeric(frame[value_column], errors="coerce").dropna()
            if not values.empty:
                base = float(values.mean())
                standard_deviation = float(values.std()) if len(values) > 1 else 1.0
                if not np.isfinite(standard_deviation) or standard_deviation <= 0:
                    standard_deviation = max(abs(base) * 0.05, 1.0)
                if len(values) > 2:
                    slope = float(np.polyfit(np.arange(len(values)), values.to_numpy(dtype=float), 1)[0])
        defaults = {
            "preset": "custom",
            "time_column": time_column,
            "value_column": value_column,
            "rows": rows,
            "start": start,
            "frequency": frequency,
            "time_mode": time_mode,
            "time_step": 1,
            "base": round(base, 4),
            "trend_type": "linear",
            "trend": round(slope, 4),
            "seasonal_model": "additive",
            "seasonality": round(standard_deviation * 0.5, 4),
            "period": period,
            "noise_type": "normal",
            "noise": round(standard_deviation * 0.25, 4),
            "random_walk": False,
            "seed": 42,
            "apply": True,
            "show": True
        }
        preset_key = str(preset).strip().lower().replace(" ", "_")
        if preset_key == "stationary":
            defaults.update({"preset": preset_key, "trend_type": "none", "trend": 0, "seasonality": 0, "noise": round(max(standard_deviation * 0.2, 0.5), 4), "random_walk": False})
        elif preset_key in ("non_stationary", "trend", "nonstationary"):
            defaults.update({"preset": preset_key, "trend_type": "linear", "trend": round(abs(slope) + max(standard_deviation * 0.05, 0.25), 4), "seasonality": 0, "random_walk": False})
        elif preset_key == "seasonal":
            defaults.update({"preset": preset_key, "trend_type": "linear", "trend": round(abs(slope) * 0.25, 4), "seasonality": round(standard_deviation * 0.9, 4), "period": period, "random_walk": False})
        elif preset_key in ("holt_winters", "trend_seasonal"):
            defaults.update({"preset": "holt_winters", "trend_type": "linear", "trend": round(abs(slope) + max(standard_deviation * 0.03, 0.2), 4), "seasonality": round(standard_deviation * 0.8, 4), "period": period, "random_walk": False})
        elif preset_key == "moving_average":
            defaults.update({"preset": preset_key, "trend_type": "linear", "trend": round(abs(slope) * 0.5, 4), "seasonality": 0, "noise": round(standard_deviation * 0.5, 4), "random_walk": False})
        elif preset_key == "random_walk":
            defaults.update({"preset": preset_key, "trend_type": "linear", "trend": 0, "seasonality": 0, "noise": round(standard_deviation * 0.5, 4), "random_walk": True})
        return defaults

    def open_time_series_simulation(self, preset="custom"):
        dialog = TimeSeriesSimulationDialog(self.root, self, preset)
        dialog.lift()
        return dialog

    def simulate_time_series(self, time_column=None, value_column=None, rows=None, start=None, frequency=None, time_mode=None, time_step=1, base=None, trend_type=None, trend=None, seasonal_model=None, seasonality=None, period=None, noise_type=None, noise=None, random_walk=False, seed=42, apply=True, show=True, preset="custom"):
        defaults = self.time_series_simulation_defaults(preset)
        time_column = time_column or defaults["time_column"]
        value_column = value_column or defaults["value_column"]
        rows = int(rows if rows is not None else defaults["rows"])
        start_was_given = start is not None
        start = start if start_was_given else defaults["start"]
        frequency = frequency if frequency is not None else defaults["frequency"]
        time_mode_key = str(time_mode if time_mode is not None else defaults["time_mode"]).strip().lower().replace(" ", "_")
        time_mode_key = {"date_range": "date", "date": "date", "datetime": "date", "numeric": "numeric", "numeric_index": "numeric", "period": "period", "period_labels": "period"}.get(time_mode_key, "date")
        if time_mode_key == "numeric" and not start_was_given:
            try:
                float(start)
            except (TypeError, ValueError):
                start = "0"
        time_step = float(time_step)
        if time_step == 0:
            raise ValueError("time_step must not be zero.")
        base = float(base if base is not None else defaults["base"])
        trend_type_key = str(trend_type if trend_type is not None else defaults["trend_type"]).strip().lower()
        trend_type_key = {"none": "none", "linear": "linear", "quadratic": "quadratic"}.get(trend_type_key, "linear")
        trend = float(trend if trend is not None else defaults["trend"])
        seasonal_model_key = str(seasonal_model if seasonal_model is not None else defaults["seasonal_model"]).strip().lower()
        seasonal_model_key = {"additive": "additive", "multiplicative": "multiplicative"}.get(seasonal_model_key, "additive")
        seasonality = float(seasonality if seasonality is not None else defaults["seasonality"])
        period = int(period if period is not None else defaults["period"])
        noise_type_key = str(noise_type if noise_type is not None else defaults["noise_type"]).strip().lower()
        noise_type_key = {"normal": "normal", "uniform": "uniform", "none": "none"}.get(noise_type_key, "normal")
        noise = float(noise if noise is not None else defaults["noise"])
        if rows < 2:
            raise ValueError("rows must be at least 2.")
        if period < 2:
            raise ValueError("period must be at least 2.")
        if period > rows // 2:
            period = max(2, rows // 2)
        if str(time_column).strip() == str(value_column).strip():
            raise ValueError("Time variable and data variable must be different.")
        frequency_key = str(frequency).strip().upper()
        frequency_key = {"H": "h", "HOUR": "h", "D": "D", "DAY": "D", "B": "B", "W": "W", "WEEK": "W", "M": "ME", "MONTH": "ME", "MONTHLY": "ME", "Q": "QE", "QUARTER": "QE", "Y": "YE", "YEAR": "YE"}.get(frequency_key, frequency_key)
        if time_mode_key == "date":
            time_values = pd.date_range(start=str(start), periods=rows, freq=frequency_key)
        elif time_mode_key == "numeric":
            start_value = float(start)
            time_values = np.arange(start_value, start_value + time_step * rows, time_step)[:rows]
        else:
            time_values = [f"P{index + 1}" for index in range(rows)]
        rng = np.random.default_rng(int(seed))
        index = np.arange(rows, dtype=float)
        if trend_type_key == "none":
            trend_component = np.zeros(rows, dtype=float)
        elif trend_type_key == "quadratic":
            trend_component = trend * (index ** 2) / max(1, rows - 1)
        else:
            trend_component = trend * index
        if seasonal_model_key == "multiplicative":
            seasonal_component = (base + trend_component) * (seasonality / 100.0) * np.sin(2 * np.pi * index / period)
        else:
            seasonal_component = seasonality * np.sin(2 * np.pi * index / period)
        if noise_type_key == "none":
            noise_component = np.zeros(rows, dtype=float)
        elif noise_type_key == "uniform":
            noise_component = rng.uniform(-noise, noise, rows)
        else:
            noise_component = rng.normal(0, noise, rows)
        if random_walk:
            drift = trend if trend_type_key == "linear" else 0
            values = base + np.cumsum(rng.normal(drift, max(noise, 1e-9), rows)) + seasonal_component
        else:
            values = base + trend_component + seasonal_component + noise_component
        frame = pd.DataFrame({time_column: time_values, value_column: values})
        frame.attrs["Time Series Simulation"] = preset
        if apply:
            self.set_df(frame)
        report = {
            "Preset": preset,
            "Time variable": time_column,
            "Data variable": value_column,
            "Rows": rows,
            "Start": start,
            "Frequency": frequency,
            "Time mode": time_mode_key,
            "Trend type": trend_type_key,
            "Trend": trend,
            "Seasonal model": seasonal_model_key,
            "Seasonal amplitude": seasonality,
            "Seasonal period": period,
            "Noise type": noise_type_key,
            "Noise": noise,
            "Random walk": bool(random_walk),
            "Seed": int(seed),
            "Applied to grid": bool(apply),
            "Preview": frame.head(10),
            "Summary": pd.DataFrame({value_column: values}).describe()
        }
        if show:
            figure, axis = plt.subplots(figsize=(9, 4))
            axis.plot(frame[time_column], frame[value_column], marker="o", markersize=3)
            axis.set_title(f"Simulated {value_column} ({preset})")
            axis.set_xlabel(time_column)
            axis.set_ylabel(value_column)
            axis.grid(alpha=0.3)
            figure.tight_layout()
            self.show_report_window(report, "Time Series Simulation Report", fig=figure)
        return frame

    def open_ml_simulations(self):
        dialog = MLSimulationDialog(self.root, self)
        dialog.lift()
        return dialog

##    def generate_multivariate_normal(self):
##        var_names = simpledialog.askstring("Input", "Enter variable names (comma-separated):")
##        means_input = simpledialog.askstring("Input", "Enter means for each variable (comma-separated):")
##        num_samples = simpledialog.askinteger("Input", "Enter number of samples:", minvalue=1)
##        
##        if not var_names or not means_input or not num_samples:
##            messagebox.showerror("Error", "Invalid input values.")
##            return
##        
##        var_names = [name.strip() for name in var_names.split(',')]
##        means = [float(mean.strip()) for mean in means_input.split(',')]
##        num_variables = len(var_names)
##        
##        if len(means) != num_variables:
##            messagebox.showerror("Error", "Number of means must match number of variables.")
##            return
##        
##        cov_matrix = np.identity(num_variables)  # Default identity covariance matrix
##        
##        simulated_data = np.random.multivariate_normal(means, cov_matrix, num_samples)
##        df_new = pd.DataFrame(simulated_data, columns=var_names)
##        
##        if self.data is None:
##            self.data = df_new
##        else:
##            self.data = pd.concat([self.data, df_new], axis=1)
##        
##        self.display_data()
##        messagebox.showinfo("Success", "Multivariate normal data appended to the data grid.")

    def perform_standardization(self, method):
        if self.data is None:
            messagebox.showerror("Error", "No data loaded!")
            return
        
        selected_col = simpledialog.askstring("Select Column", "Enter the column name for standardization:")
        if selected_col not in self.data.columns:
            messagebox.showerror("Error", "Invalid column name!")
            return
        
        new_col_name = f"{selected_col}_{method}"  # Append transformation name to column
        df = self.data.copy()
        
        if method == "Min-Max":
            scaler = MinMaxScaler()
            df[new_col_name] = scaler.fit_transform(df[[selected_col]])
        elif method == "Z-Score":
            scaler = StandardScaler()
            df[new_col_name] = scaler.fit_transform(df[[selected_col]])
        elif method == "Decimal Scaling":
            max_abs = df[selected_col].abs().max()
            df[new_col_name] = df[selected_col] / (10 ** np.ceil(np.log10(max_abs)))
        elif method == "Log":
            df[new_col_name] = np.log(df[selected_col] + 1)
        elif method == "Log-Normal":
            df[new_col_name] = np.exp(df[selected_col])
        
        self.data = df  # Update main data
        self.display_data()
        messagebox.showinfo("Success", f"{method} applied successfully to {selected_col}. New variable added: {new_col_name}")

##    def perform_decomposition(self, method):
##        if self.data is None:
##            messagebox.showerror("Error", "No data loaded!")
##            return
##        
##        selected_columns = simpledialog.askstring("Select Columns", "Enter column names for decomposition (comma-separated):")
##        if not selected_columns:
##            messagebox.showerror("Error", "No columns selected!")
##            return
##        
##        selected_columns = [col.strip() for col in selected_columns.split(',')]
##        if any(col not in self.data.columns for col in selected_columns):
##            messagebox.showerror("Error", "Invalid column names!")
##            return
##        
##        matrix = self.data[selected_columns].values.astype(float)
##        
##        try:
##            if method == "cholesky":
##                result = np.linalg.cholesky(matrix)
##            elif method == "qr":
##                result, _ = np.linalg.qr(matrix)
##            elif method == "svd":
##                result = np.linalg.svd(matrix, full_matrices=False)[0]
##            elif method == "eig":
##                result = np.linalg.eig(matrix)[1]
##            else:
##                messagebox.showerror("Error", "Invalid decomposition method!")
##                return
##            
##            df_result = pd.DataFrame(result, columns=[f"{method}_comp_{i+1}" for i in range(result.shape[1])])
##            self.data = pd.concat([self.data, df_result], axis=1)
##            self.display_data()
##            messagebox.showinfo("Success", f"{method.capitalize()} decomposition performed successfully! Results appended to the data grid.")
##        except np.linalg.LinAlgError as e:
##            messagebox.showerror("Error", f"Decomposition failed: {e}")

    def perform_decomposition(self, method):
        if self.data is None:
            messagebox.showerror("Error", "No data loaded!")
            return
        
        selected_columns = simpledialog.askstring("Select Columns", "Enter column names for decomposition (comma-separated):")
        if not selected_columns:
            messagebox.showerror("Error", "No columns selected!")
            return
        
        selected_columns = [col.strip() for col in selected_columns.split(',')]
        if any(col not in self.data.columns for col in selected_columns):
            messagebox.showerror("Error", "Invalid column names!")
            return
        
        matrix = self.data[selected_columns].values.astype(float)
        
        try:
            if method == "cholesky":
                matrix = matrix @ matrix.T  # Ensure the matrix is symmetric positive definite
                result = np.linalg.cholesky(matrix)
            elif method == "qr":
                result, _ = np.linalg.qr(matrix)
            elif method == "svd":
                result = np.linalg.svd(matrix, full_matrices=False)[0]
            elif method == "eig":
                result = np.linalg.eig(matrix)[1]
            else:
                messagebox.showerror("Error", "Invalid decomposition method!")
                return
            
            df_result = pd.DataFrame(result, columns=[f"{method}_comp_{i+1}" for i in range(result.shape[1])])
            self.data = pd.concat([self.data, df_result], axis=1)
            self.display_data()
            messagebox.showinfo("Success", f"{method.capitalize()} decomposition performed successfully! Results appended to the data grid.")
        except np.linalg.LinAlgError as e:
            messagebox.showerror("Error", f"Decomposition failed: {e}")

    def _report_value_to_text(self, value):
        if value is None:
            return ""
        if hasattr(value, "to_string"):
            return value.to_string()
        if isinstance(value, dict):
            return "\n".join(f"{key}\n{self._report_value_to_text(item)}" for key, item in value.items())
        if isinstance(value, (list, tuple, set, pd.Index)):
            return "\n".join(self._report_value_to_text(item) for item in value)
        return str(value)

    def _copy_report_text(self, text_widget, button=None):
        content = text_widget.get("1.0", tk.END).rstrip("\n")
        window = text_widget.winfo_toplevel()
        window.clipboard_clear()
        window.clipboard_append(content)
        window.update()
        if button is not None:
            button.configure(text="Copied!")
            text_widget.after(1200, lambda: button.configure(text="Copy Output"))

    def _report_default_stem(self, title):
        stem = re.sub(r"[^A-Za-z0-9]+", "_", str(title or "report")).strip("_").lower()
        return stem or "report"

    def _pdf_safe_text(self, value):
        return str(value).encode("latin-1", "replace").decode("latin-1")

    def _export_report_files(self, title, data=None, figures=None):
        selected_figures = [figure for figure in (figures or []) if figure is not None]
        file_path = filedialog.asksaveasfilename(
            defaultextension=".docx",
            initialfile=self._report_default_stem(title),
            filetypes=[("Word Document", "*.docx"), ("PDF File", "*.pdf")]
        )
        if not file_path:
            return False
        temporary_images = []
        try:
            for index, figure in enumerate(selected_figures):
                image_path = f"temp_report_chart_{os.getpid()}_{index}.png"
                figure.savefig(image_path, bbox_inches="tight")
                temporary_images.append(image_path)
            if file_path.lower().endswith(".docx"):
                document = Document()
                document.add_heading(str(title), level=1)
                if data is not None:
                    if isinstance(data, dict):
                        for key, value in data.items():
                            document.add_heading(str(key), level=2)
                            document.add_paragraph(self._report_value_to_text(value))
                    else:
                        document.add_paragraph(self._report_value_to_text(data))
                for image_path in temporary_images:
                    document.add_picture(image_path, width=Inches(6.2))
                document.save(file_path)
            elif file_path.lower().endswith(".pdf"):
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                pdf.multi_cell(200, 10, self._pdf_safe_text(title) + "\n\n")
                if data is not None:
                    if isinstance(data, dict):
                        for key, value in data.items():
                            pdf.multi_cell(200, 10, self._pdf_safe_text(f"{key}\n{self._report_value_to_text(value)}\n\n"))
                    else:
                        pdf.multi_cell(200, 10, self._pdf_safe_text(self._report_value_to_text(data)))
                for image_path in temporary_images:
                    if pdf.get_y() + 100 > pdf.page_break_trigger:
                        pdf.add_page()
                    pdf.image(image_path, x=10, y=pdf.get_y(), w=180)
                pdf.output(file_path)
            else:
                return False
        except Exception as error:
            messagebox.showerror("Export Failed", str(error))
            return False
        finally:
            for image_path in temporary_images:
                if os.path.exists(image_path):
                    os.remove(image_path)
        return True

    def show_report_window(self, data, title, fig=None, figures=None):
        report_window = Toplevel(self.root)
        report_window.title(title)
        report_window.geometry("900x650")

        frame = tk.Frame(report_window)
        frame.pack(fill="both", expand=True)

        paned_window = tk.PanedWindow(frame, orient="horizontal", sashwidth=6, sashrelief="raised", opaqueresize=True)
        paned_window.pack(fill="both", expand=True)

        output_frame = tk.Frame(paned_window)
        paned_window.add(output_frame, minsize=240, stretch="always")
        output_frame.columnconfigure(0, weight=1)
        output_frame.rowconfigure(0, weight=1)
        text_widget = tk.Text(output_frame, wrap=tk.NONE)
        text_scroll_y = ttk.Scrollbar(output_frame, orient="vertical", command=text_widget.yview)
        text_scroll_x = ttk.Scrollbar(output_frame, orient="horizontal", command=text_widget.xview)
        text_widget.configure(yscrollcommand=text_scroll_y.set, xscrollcommand=text_scroll_x.set)
        text_widget.grid(row=0, column=0, sticky="nsew")
        text_scroll_y.grid(row=0, column=1, sticky="ns")
        text_scroll_x.grid(row=1, column=0, sticky="ew")

        if isinstance(data, dict):
            for key, value in data.items():
                text_widget.insert(tk.END, f"{key}\n{self._report_value_to_text(value)}\n\n")
        else:
            text_widget.insert(tk.END, f"{title}\n\n{self._report_value_to_text(data)}")

        all_figures = []
        if fig is not None:
            all_figures.append(fig)
        if figures:
            all_figures.extend([figure for figure in figures if figure is not None])

        if all_figures:
            plot_frame = tk.Frame(paned_window)
            paned_window.add(plot_frame, minsize=240, stretch="always")
            plot_canvas = tk.Canvas(plot_frame, highlightthickness=0)
            plot_scrollbar = ttk.Scrollbar(plot_frame, orient="vertical", command=plot_canvas.yview)
            plot_inner = ttk.Frame(plot_canvas)
            plot_inner.bind("<Configure>", lambda event: plot_canvas.configure(scrollregion=plot_canvas.bbox("all")))
            plot_window = plot_canvas.create_window((0, 0), window=plot_inner, anchor="nw")
            plot_canvas.configure(yscrollcommand=plot_scrollbar.set)
            plot_canvas.bind("<Configure>", lambda event: plot_canvas.itemconfigure(plot_window, width=event.width))
            plot_scrollbar.pack(side="right", fill="y")
            plot_canvas.pack(side="left", fill="both", expand=True)
            figure_canvases = []
            for figure in all_figures:
                canvas = FigureCanvasTkAgg(figure, master=plot_inner)
                canvas.draw()
                canvas.get_tk_widget().pack(fill="x", padx=5, pady=5)
                figure_canvases.append(canvas)
            report_window.figure_canvases = figure_canvases

        copy_button = tk.Button(report_window, text="Copy Output", command=lambda: self._copy_report_text(text_widget, copy_button))
        copy_button.pack(anchor="nw", padx=10, pady=(5, 0))
        download_button = tk.Button(report_window, text="Download Report", command=lambda: self.download_report(data, title, figures=all_figures))
        download_button.pack(anchor="nw", padx=10, pady=5)
        return report_window

    def download_report(self, data, title, fig=None, figures=None):
        if self.license_type != "FULL":
            messagebox.showinfo("Feature Locked", "Unlock this feature by buying a license. Go to Help menu for more details.")
            return
        all_figures = []
        if fig is not None:
            all_figures.append(fig)
        if figures:
            all_figures.extend([figure for figure in figures if figure is not None])
        if self._export_report_files(title, data, all_figures):
            messagebox.showinfo("Success", "Report saved successfully!")

    def generate_basket_data(self, kind="wide"):
        if self.data is not None:
            if not messagebox.askyesno("Replace Data", "Generating basket data will replace the current data grid. Continue?"):
                return

        if kind == "long":
            customers = simpledialog.askinteger("Generate Customer Transaction Log", "Enter number of customers:", initialvalue=50, minvalue=5, maxvalue=5000)
            if customers is None:
                return
            transactions_per_customer = simpledialog.askinteger("Generate Customer Transaction Log", "Enter transactions per customer:", initialvalue=4, minvalue=1, maxvalue=50)
            if transactions_per_customer is None:
                return
            item_count = simpledialog.askinteger("Generate Customer Transaction Log", "Enter number of items:", initialvalue=10, minvalue=3, maxvalue=52)
            if item_count is None:
                return
            basket_size = simpledialog.askinteger("Generate Customer Transaction Log", "Enter average basket size:", initialvalue=3, minvalue=1, maxvalue=item_count)
            if basket_size is None:
                return
            seed = simpledialog.askinteger("Generate Customer Transaction Log", "Enter random seed:", initialvalue=42, minvalue=0)
            if seed is None:
                return
            generated = self.generate_long_transaction_data(customers, transactions_per_customer, item_count, basket_size, seed)
            success_message = "Customer transaction log generated successfully!"
        elif kind == "sequence":
            customers = simpledialog.askinteger("Generate Sequence Log", "Enter number of customers:", initialvalue=50, minvalue=5, maxvalue=5000)
            if customers is None:
                return
            sequence_length = simpledialog.askinteger("Generate Sequence Log", "Enter events per sequence:", initialvalue=6, minvalue=2, maxvalue=20)
            if sequence_length is None:
                return
            item_count = simpledialog.askinteger("Generate Sequence Log", "Enter number of items:", initialvalue=10, minvalue=3, maxvalue=52)
            if item_count is None:
                return
            seed = simpledialog.askinteger("Generate Sequence Log", "Enter random seed:", initialvalue=42, minvalue=0)
            if seed is None:
                return
            generated = self.generate_sequence_data(customers, sequence_length, item_count, seed)
            success_message = "Sequence log generated successfully!"
        elif kind == "profit":
            transactions = simpledialog.askinteger("Generate Profit / Utility Basket", "Enter number of transactions:", initialvalue=200, minvalue=10, maxvalue=10000)
            if transactions is None:
                return
            item_count = simpledialog.askinteger("Generate Profit / Utility Basket", "Enter number of items:", initialvalue=10, minvalue=3, maxvalue=52)
            if item_count is None:
                return
            basket_size = simpledialog.askinteger("Generate Profit / Utility Basket", "Enter average basket size:", initialvalue=3, minvalue=1, maxvalue=item_count)
            if basket_size is None:
                return
            seed = simpledialog.askinteger("Generate Profit / Utility Basket", "Enter random seed:", initialvalue=42, minvalue=0)
            if seed is None:
                return
            generated = self.generate_profit_basket_data(transactions, item_count, basket_size, seed)
            success_message = "Profit / utility basket data generated successfully!"
        elif kind == "marketing":
            customers = simpledialog.askinteger("Generate Marketing Campaign Log", "Enter number of customers:", initialvalue=200, minvalue=10, maxvalue=10000)
            if customers is None:
                return
            transactions_per_customer = simpledialog.askinteger("Generate Marketing Campaign Log", "Enter purchases per customer:", initialvalue=3, minvalue=1, maxvalue=20)
            if transactions_per_customer is None:
                return
            seed = simpledialog.askinteger("Generate Marketing Campaign Log", "Enter random seed:", initialvalue=42, minvalue=0)
            if seed is None:
                return
            generated = self.generate_marketing_data(customers, transactions_per_customer, seed)
            success_message = "Marketing campaign log generated successfully!"
        else:
            transactions = simpledialog.askinteger("Generate Basket Matrix", "Enter number of transactions:", initialvalue=200, minvalue=10, maxvalue=10000)
            if transactions is None:
                return
            item_count = simpledialog.askinteger("Generate Basket Matrix", "Enter number of items:", initialvalue=10, minvalue=3, maxvalue=52)
            if item_count is None:
                return
            basket_size = simpledialog.askinteger("Generate Basket Matrix", "Enter average basket size:", initialvalue=3, minvalue=1, maxvalue=item_count)
            if basket_size is None:
                return
            strength = simpledialog.askfloat("Generate Basket Matrix", "Enter co-occurrence strength (0 to 1):", initialvalue=0.6, minvalue=0.0, maxvalue=1.0)
            if strength is None:
                return
            seed = simpledialog.askinteger("Generate Basket Matrix", "Enter random seed:", initialvalue=42, minvalue=0)
            if seed is None:
                return
            generated = self.generate_wide_basket_data(transactions, item_count, basket_size, strength, seed)
            success_message = "Basket matrix generated successfully!"

        self.data = generated
        self.file_path = None
        self.display_data()
        messagebox.showinfo("Success", success_message)

    def generate_wide_basket_data(self, transactions, item_count, basket_size, strength, seed):
        rng = np.random.default_rng(seed)
        item_names = [f"Item_{index + 1}" for index in range(item_count)]
        records = []
        for index in range(transactions):
            chosen = set(rng.choice(item_names, size=basket_size, replace=False))
            if item_count >= 2 and rng.random() < strength:
                chosen.update(item_names[:2])
            if item_count >= 5 and rng.random() < strength * 0.7:
                chosen.update(item_names[2:5])
            records.append([1 if item in chosen else 0 for item in item_names])
        frame = pd.DataFrame(records, columns=item_names)
        frame.insert(0, "TransactionID", [f"T{index + 1}" for index in range(transactions)])
        return frame

    def generate_long_transaction_data(self, customers, transactions_per_customer, item_count, basket_size, seed):
        rng = np.random.default_rng(seed)
        item_names = [f"Item_{index + 1}" for index in range(item_count)]
        records = []
        start_date = pd.Timestamp("2025-01-01")
        for customer_index in range(customers):
            customer_id = f"C{customer_index + 1:03d}"
            transaction_count = int(rng.integers(max(1, transactions_per_customer - 1), transactions_per_customer + 1))
            for _ in range(transaction_count):
                chosen = set(rng.choice(item_names, size=basket_size, replace=False))
                if item_count >= 2 and rng.random() < 0.5:
                    chosen.update(item_names[:2])
                items = sorted(chosen)
                day_offset = int(rng.integers(0, 180))
                amount = round(float(rng.uniform(15, 60) * len(items)), 2)
                records.append({
                    "CustomerID": customer_id,
                    "TransactionID": f"T{len(records) + 1:05d}",
                    "Date": (start_date + pd.Timedelta(days=day_offset)).strftime("%Y-%m-%d"),
                    "Amount": amount,
                    "Items": ", ".join(items)
                })
        return pd.DataFrame(records)

    def generate_sequence_data(self, customers, sequence_length, item_count, seed):
        rng = np.random.default_rng(seed)
        item_names = [f"Item_{index + 1}" for index in range(item_count)]
        records = []
        for customer_index in range(customers):
            sequence = list(rng.choice(item_names, size=sequence_length, replace=True))
            if item_count >= 2 and rng.random() < 0.7:
                sequence[0] = item_names[0]
                sequence[1] = item_names[1]
            records.append({
                "CustomerID": f"C{customer_index + 1:03d}",
                "Sequence": " > ".join(sequence)
            })
        return pd.DataFrame(records)

    def generate_profit_basket_data(self, transactions, item_count, basket_size, seed):
        rng = np.random.default_rng(seed)
        frame = self.generate_wide_basket_data(transactions, item_count, basket_size, 0.6, seed)
        profit_values = {}
        for index in range(item_count):
            item = f"Item_{index + 1}"
            value = round(float(rng.uniform(5, 50)), 2)
            profit_values[item] = value
            frame[f"Profit_{item}"] = value
        frame["Amount"] = sum(frame[item] * value for item, value in profit_values.items()).round(2)
        return frame

    def generate_marketing_data(self, customers=200, transactions_per_customer=3, seed=42):
        rng = np.random.default_rng(seed)
        channels = np.array(["Email", "Social", "Search", "Display"])
        channel_effects = {"Email": 0.45, "Social": 0.1, "Search": 0.75, "Display": -0.2}
        records = []
        start_date = pd.Timestamp("2025-01-01")
        for customer_index in range(customers):
            customer_id = f"C{customer_index + 1:04d}"
            age = int(rng.integers(18, 66))
            tenure = int(rng.integers(1, 72))
            support_tickets = int(rng.poisson(0.8))
            channel = str(rng.choice(channels))
            response_score = 1 / (1 + np.exp(-(channel_effects[channel] + rng.normal(0, 0.6))))
            response = int(rng.random() < response_score)
            churn_score = 1 / (1 + np.exp(-(-1.2 + support_tickets * 0.5 - tenure * 0.03)))
            churn = int(rng.random() < churn_score)
            for purchase_index in range(transactions_per_customer):
                price = round(float(rng.uniform(20, 90)), 2)
                quantity = int(max(1, round(14 - 0.09 * price + rng.normal(0, 1.2))))
                amount = round(price * quantity, 2)
                day_offset = int(rng.integers(0, 180))
                records.append({
                    "CustomerID": customer_id,
                    "Date": (start_date + pd.Timedelta(days=day_offset)).strftime("%Y-%m-%d"),
                    "Age": age,
                    "TenureMonths": tenure,
                    "SupportTickets": support_tickets,
                    "CampaignChannel": channel,
                    "Response": response,
                    "Churn": churn,
                    "Price": price,
                    "Quantity": quantity,
                    "Amount": amount,
                })
        return pd.DataFrame(records)

    def _default_item_columns(self):
        if self.data is None:
            return ""
        if "Items" in self.data.columns:
            return "Items"
        if "Sequence" in self.data.columns:
            return "Sequence"
        excluded = {"TransactionID", "CustomerID", "Date", "Amount"}
        return ", ".join(str(column) for column in self.data.columns if column not in excluded and not str(column).startswith("Profit_"))

    def _ask_association_parameters(self, algorithm=None):
        if self.data is None:
            messagebox.showerror("Error", "No data loaded!")
            return None

        params_window = Toplevel(self.root)
        params_window.title("Market Basket Analysis Parameters")
        params_window.geometry("560x400")
        params_window.transient(self.root)
        params_window.grab_set()

        frame = tk.Frame(params_window)
        frame.pack(pady=10, padx=10, fill="both", expand=True)
        frame.columnconfigure(1, weight=1)

        algorithm_var = tk.StringVar(value=algorithm if algorithm in ("Apriori", "FP-Growth", "Eclat") else "Apriori")
        columns_var = tk.StringVar(value=self._default_item_columns())
        min_support_var = tk.StringVar(value="0.10")
        min_confidence_var = tk.StringVar(value="0.60")
        min_lift_var = tk.StringVar(value="1.0")
        max_len_var = tk.StringVar(value="")
        top_n_var = tk.StringVar(value="100")
        amount_var = tk.StringVar(value="Amount" if "Amount" in self.data.columns else "")

        algorithm_entry = ttk.Combobox(frame, textvariable=algorithm_var, values=["Apriori", "FP-Growth", "Eclat"], state="readonly")
        rows = [
            ("Algorithm", algorithm_entry),
            ("Item columns / item-list column", tk.Entry(frame, textvariable=columns_var, width=55)),
            ("Minimum support", tk.Entry(frame, textvariable=min_support_var, width=55)),
            ("Minimum confidence", tk.Entry(frame, textvariable=min_confidence_var, width=55)),
            ("Minimum lift", tk.Entry(frame, textvariable=min_lift_var, width=55)),
            ("Maximum itemset length", tk.Entry(frame, textvariable=max_len_var, width=55)),
            ("Top N results", tk.Entry(frame, textvariable=top_n_var, width=55)),
            ("Amount column for utility", tk.Entry(frame, textvariable=amount_var, width=55))
        ]
        for row, (label, widget) in enumerate(rows):
            tk.Label(frame, text=label).grid(row=row, column=0, padx=5, pady=6, sticky="w")
            widget.grid(row=row, column=1, padx=5, pady=6, sticky="ew")

        result = {}

        def apply_parameters():
            try:
                min_support = float(min_support_var.get())
                min_confidence = float(min_confidence_var.get())
                min_lift = float(min_lift_var.get())
                max_len_text = max_len_var.get().strip()
                max_len = int(max_len_text) if max_len_text else None
                top_n = int(top_n_var.get())
            except ValueError:
                messagebox.showerror("Error", "Thresholds and limits must be numeric.", parent=params_window)
                return
            if not 0 < min_support <= 1:
                messagebox.showerror("Error", "Minimum support must be greater than 0 and at most 1.", parent=params_window)
                return
            if not 0 <= min_confidence <= 1:
                messagebox.showerror("Error", "Minimum confidence must be between 0 and 1.", parent=params_window)
                return
            if min_lift < 0:
                messagebox.showerror("Error", "Minimum lift cannot be negative.", parent=params_window)
                return
            if max_len is not None and max_len < 2:
                messagebox.showerror("Error", "Maximum itemset length must be at least 2.", parent=params_window)
                return
            if top_n < 1:
                messagebox.showerror("Error", "Top N results must be at least 1.", parent=params_window)
                return
            result.update({
                "algorithm": algorithm_var.get(),
                "columns": columns_var.get(),
                "min_support": min_support,
                "min_confidence": min_confidence,
                "min_lift": min_lift,
                "max_len": max_len,
                "top_n": top_n,
                "amount_column": amount_var.get().strip()
            })
            params_window.destroy()

        button_frame = tk.Frame(params_window)
        button_frame.pack(pady=10)
        tk.Button(button_frame, text="Run", command=apply_parameters).pack(side="left", padx=5)
        tk.Button(button_frame, text="Cancel", command=params_window.destroy).pack(side="left", padx=5)

        self.root.wait_window(params_window)
        return result or None

    def _build_transaction_matrix(self, columns):
        if not columns:
            raise ValueError("No item columns were selected.")

        if len(columns) == 1:
            series = self.data[columns[0]]
            if columns[0] == "Items" or series.dropna().astype(str).str.contains(",").any():
                item_lists = series.fillna("").astype(str).apply(lambda value: [item.strip() for item in value.split(",") if item.strip()])
                item_names = sorted({item for items in item_lists for item in items})
                matrix = pd.DataFrame(False, index=self.data.index, columns=item_names)
                for row_index, items in item_lists.items():
                    if items:
                        matrix.loc[row_index, items] = True
                matrix = matrix.loc[matrix.any(axis=1)]
                if matrix.empty:
                    raise ValueError("The item-list column did not produce any transactions.")
                return matrix

        pieces = []
        for column in columns:
            series = self.data[column]
            if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
                values = set(series.dropna().unique())
                if values.issubset({0, 1, False, True, 0.0, 1.0}):
                    pieces.append(series.fillna(False).astype(bool).rename(column))
                    continue
            pieces.append(pd.get_dummies(series.astype(str), prefix=column, prefix_sep="=", dtype=bool))
        matrix = pd.concat(pieces, axis=1).fillna(False).astype(bool)
        matrix = matrix.loc[matrix.any(axis=1)]
        if matrix.empty:
            raise ValueError("The selected columns did not produce any transactions.")
        return matrix

    def perform_transaction_matrix(self):
        fields = (("columns", "Item columns / item-list column", tk.StringVar(value=self._default_item_columns())),)
        params = self._ask_field_dialog("Transaction Matrix", fields)
        if not params:
            return
        try:
            columns = [column.strip() for column in params["columns"].split(",") if column.strip()]
            matrix = self.transaction_matrix(columns)
            self.show_report_window({"Transaction Matrix": matrix.astype(int)}, "Transaction Matrix")
        except Exception as error:
            messagebox.showerror("Transaction Matrix Failed", str(error))

    def perform_association_rules(self):
        params = self._ask_association_parameters()
        if not params:
            return
        try:
            columns = [column.strip() for column in params["columns"].split(",") if column.strip()]
            if not columns:
                messagebox.showerror("Error", "Enter at least one item column or item-list column.")
                return
            missing_columns = [column for column in columns if column not in self.data.columns]
            if missing_columns:
                messagebox.showerror("Error", f"Columns not found: {', '.join(missing_columns)}")
                return
            result = self.association_rules(
                columns=columns,
                min_support=params["min_support"],
                min_confidence=params["min_confidence"],
                min_lift=params["min_lift"],
                algorithm=params["algorithm"],
                max_len=params["max_len"],
                top_n=params["top_n"],
                show=False
            )
            if result.empty:
                messagebox.showinfo("No Results", "No association rules found at the selected thresholds.")
                return
            self.show_report_window({"Association Rules": result}, "Association Rules")
        except Exception as error:
            messagebox.showerror("Association Rules Failed", str(error))

    def _eclat_frequent_itemsets(self, matrix, min_support, max_len=None):
        total = len(matrix)
        tidsets = {}
        for column in matrix.columns:
            tidset = frozenset(matrix.index[matrix[column]].tolist())
            if len(tidset) / total >= min_support:
                tidsets[frozenset([column])] = tidset

        results = []
        seen = set()

        def extend(prefix, tidset):
            results.append({"support": len(tidset) / total, "itemsets": frozenset(prefix)})
            if max_len is not None and len(prefix) >= max_len:
                return
            for item, item_tidset in tidsets.items():
                if item.issubset(prefix):
                    continue
                next_itemset = prefix | item
                if next_itemset in seen:
                    continue
                intersection = tidset & item_tidset
                if len(intersection) / total < min_support:
                    continue
                seen.add(next_itemset)
                extend(next_itemset, intersection)

        for itemset, tidset in tidsets.items():
            if itemset in seen:
                continue
            seen.add(itemset)
            extend(itemset, tidset)

        if not results:
            return pd.DataFrame(columns=["support", "itemsets"])
        frame = pd.DataFrame(results)
        frame["_key"] = frame["itemsets"].apply(lambda value: "|".join(sorted(str(item) for item in value)))
        frame = frame.sort_values(["support", "_key"], ascending=[False, True]).drop(columns="_key").reset_index(drop=True)
        return frame

    def _mine_frequent_itemsets(self, algorithm, matrix, min_support, max_len=None):
        if algorithm == "Eclat":
            return self._eclat_frequent_itemsets(matrix, min_support, max_len)
        try:
            from mlxtend.frequent_patterns import apriori, fpgrowth
        except ImportError:
            messagebox.showerror("Missing Dependency", "mlxtend is required for Apriori and FP-Growth.")
            return None
        if algorithm == "Apriori":
            return apriori(matrix, min_support=min_support, use_colnames=True, max_len=max_len)
        return fpgrowth(matrix, min_support=min_support, use_colnames=True, max_len=max_len)

    def _format_itemsets(self, frame, column):
        formatted = frame.copy()
        formatted[column] = formatted[column].apply(lambda value: " AND ".join(sorted(str(item) for item in value)))
        return formatted

    def _format_rules(self, rules):
        formatted = rules.copy()
        for column in ("antecedents", "consequents"):
            formatted[column] = formatted[column].apply(lambda value: " AND ".join(sorted(str(item) for item in value)))
        metric_columns = [column for column in ("antecedent support", "consequent support", "support", "confidence", "lift", "leverage", "conviction") if column in formatted.columns]
        utility_columns = [column for column in ("total_utility", "avg_utility", "utility_lift") if column in formatted.columns]
        for column in metric_columns + utility_columns:
            formatted[column] = pd.to_numeric(formatted[column], errors="coerce").round(4)
        return formatted[["antecedents", "consequents"] + metric_columns + utility_columns]

    def _build_rules_plot(self, rules):
        figure, axes = plt.subplots(figsize=(5, 4))
        lift = pd.to_numeric(rules["lift"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if lift.notna().any():
            scatter = axes.scatter(rules["support"], rules["confidence"], c=lift, cmap="viridis", s=40)
            figure.colorbar(scatter, ax=axes, label="Lift")
        else:
            axes.scatter(rules["support"], rules["confidence"], color="blue", s=40)
        axes.set_xlabel("Support")
        axes.set_ylabel("Confidence")
        axes.set_title("Association Rules")
        figure.tight_layout()
        return figure

    def perform_association_mining(self, algorithm=None):
        params = self._ask_association_parameters(algorithm)
        if not params:
            return

        try:
            from mlxtend.frequent_patterns import association_rules
        except ImportError:
            messagebox.showerror("Missing Dependency", "mlxtend is required to generate association rules.")
            return

        try:
            columns = [column.strip() for column in params["columns"].split(",") if column.strip()]
            missing_columns = [column for column in columns if column not in self.data.columns]
            if not columns:
                messagebox.showerror("Error", "Enter at least one item column or item-list column.")
                return
            if missing_columns:
                messagebox.showerror("Error", f"Columns not found: {', '.join(missing_columns)}")
                return

            matrix = self._build_transaction_matrix(columns)
            if matrix.shape[1] < 2:
                messagebox.showerror("Error", "At least two distinct items are required to generate rules.")
                return

            frequent = self._mine_frequent_itemsets(params["algorithm"], matrix, params["min_support"], params["max_len"])
            if frequent is None:
                return
            if frequent.empty:
                messagebox.showinfo("No Results", "No frequent itemsets found at the selected minimum support.")
                return

            rules = association_rules(frequent, num_itemsets=len(matrix), metric="confidence", min_threshold=params["min_confidence"])
            if rules.empty:
                messagebox.showinfo("No Results", "No association rules found at the selected minimum confidence.")
            else:
                rules = rules[rules["lift"] >= params["min_lift"]].copy()
                rules = rules.sort_values(["lift", "confidence", "support"], ascending=False).head(params["top_n"])

            itemsets_display = self._format_itemsets(frequent.head(params["top_n"]), "itemsets")
            rules_display = self._format_rules(rules)
            figure = self._build_rules_plot(rules) if not rules.empty else None
            title = f"Market Basket Analysis - {params['algorithm']}"
            self.show_report_window({"Frequent Itemsets": itemsets_display, "Association Rules": rules_display}, title, figure)
        except Exception as error:
            messagebox.showerror("Market Basket Analysis Failed", str(error))

    def show_market_basket_report(self):
        self.perform_association_mining()

    def _ask_field_dialog(self, title, fields):
        if self.data is None:
            messagebox.showerror("Error", "No data loaded!")
            return None

        dialog = Toplevel(self.root)
        dialog.title(title)
        dialog.geometry(f"520x{130 + len(fields) * 32}")
        dialog.transient(self.root)
        dialog.grab_set()

        frame = tk.Frame(dialog)
        frame.pack(pady=10, padx=10, fill="both", expand=True)
        frame.columnconfigure(1, weight=1)

        for row, (key, label, variable) in enumerate(fields):
            tk.Label(frame, text=label).grid(row=row, column=0, padx=5, pady=6, sticky="w")
            tk.Entry(frame, textvariable=variable, width=55).grid(row=row, column=1, padx=5, pady=6, sticky="ew")

        values = {}

        def apply_values():
            values.update({key: variable.get().strip() for key, label, variable in fields})
            dialog.destroy()

        button_frame = tk.Frame(dialog)
        button_frame.pack(pady=10)
        tk.Button(button_frame, text="Run", command=apply_values).pack(side="left", padx=5)
        tk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side="left", padx=5)

        self.root.wait_window(dialog)
        return values or None

    def _closed_frequent_itemsets(self, frequent):
        records = list(zip(frequent["itemsets"], frequent["support"]))
        rows = []
        for itemset, support in records:
            closed = True
            for other, other_support in records:
                if len(other) > len(itemset) and set(itemset).issubset(set(other)) and abs(float(other_support) - float(support)) < 1e-12:
                    closed = False
                    break
            if closed:
                rows.append({"support": support, "itemsets": itemset})
        if not rows:
            return pd.DataFrame(columns=["support", "itemsets"])
        return pd.DataFrame(rows).sort_values("support", ascending=False).reset_index(drop=True)

    def perform_itemset_variant(self, variant):
        params = self._ask_association_parameters()
        if not params:
            return

        try:
            columns = [column.strip() for column in params["columns"].split(",") if column.strip()]
            missing_columns = [column for column in columns if column not in self.data.columns]
            if not columns or missing_columns:
                messagebox.showerror("Error", f"Invalid item columns: {', '.join(missing_columns) if missing_columns else 'none selected'}")
                return

            matrix = self._build_transaction_matrix(columns)
            if matrix.shape[1] < 2:
                messagebox.showerror("Error", "At least two distinct items are required.")
                return

            if variant == "Maximal":
                try:
                    from mlxtend.frequent_patterns import fpmax
                except ImportError:
                    messagebox.showerror("Missing Dependency", "mlxtend is required for maximal itemsets.")
                    return
                result = fpmax(matrix, min_support=params["min_support"], use_colnames=True, max_len=params["max_len"])
            else:
                result = self._mine_frequent_itemsets(params["algorithm"], matrix, params["min_support"], params["max_len"])
                if result is None:
                    return
                result = self._closed_frequent_itemsets(result)

            if result.empty:
                messagebox.showinfo("No Results", f"No {variant.lower()} itemsets found at the selected minimum support.")
                return
            display = self._format_itemsets(result.head(params["top_n"]), "itemsets")
            self.show_report_window({"Itemsets": display}, f"{variant} Frequent Itemsets", None)
        except Exception as error:
            messagebox.showerror("Itemset Mining Failed", str(error))

    def _build_sequence_database(self, sequence_column=None):
        if sequence_column and sequence_column in self.data.columns:
            values = self.data[sequence_column]
        elif "Sequence" in self.data.columns:
            values = self.data["Sequence"]
        elif "CustomerID" in self.data.columns and "Item" in self.data.columns:
            order_columns = [column for column in ("CustomerID", "EventOrder") if column in self.data.columns]
            ordered = self.data.sort_values(order_columns)
            values = ordered.groupby("CustomerID")["Item"].apply(lambda items: " > ".join(str(item) for item in items))
        else:
            raise ValueError("Provide a sequence column or a CustomerID/Item transaction log.")

        database = []
        for value in values:
            if pd.isna(value):
                continue
            text = str(value).strip()
            if ">" in text:
                events = [part.strip() for part in text.split(">") if part.strip()]
            else:
                events = [part.strip() for part in text.split(",") if part.strip()]
            if events:
                database.append(events)
        if not database:
            raise ValueError("The sequence column did not produce any sequences.")
        return database

    def _contains_subsequence(self, sequence, pattern):
        iterator = iter(sequence)
        return all(any(event == item for event in iterator) for item in pattern)

    def _prefixspan_frequent_sequences(self, database, min_support, max_len=None):
        total = len(database)
        counts = {}
        for sequence in database:
            for item in set(sequence):
                counts[item] = counts.get(item, 0) + 1

        singletons = [tuple([item]) for item, count in counts.items() if count / total >= min_support]
        results = [(sequence, counts[sequence[0]] / total) for sequence in singletons]
        current = set(singletons)
        length = 1

        while current and singletons and (max_len is None or length < max_len):
            candidates = set()
            for sequence in current:
                for item in singletons:
                    candidates.add(sequence + item)
            next_current = set()
            for candidate in sorted(candidates):
                support = sum(1 for sequence in database if self._contains_subsequence(sequence, candidate)) / total
                if support >= min_support:
                    next_current.add(candidate)
                    results.append((candidate, support))
            current = next_current
            length += 1

        rows = [{"sequence": " > ".join(sequence), "support": round(support, 4), "length": len(sequence)} for sequence, support in results]
        if not rows:
            return pd.DataFrame(columns=["sequence", "support", "length"])
        return pd.DataFrame(rows).sort_values(["support", "length", "sequence"], ascending=[False, True, True]).reset_index(drop=True)

    def _ask_sequence_parameters(self):
        if self.data is None:
            messagebox.showerror("Error", "No data loaded!")
            return None
        sequence_default = "Sequence" if "Sequence" in self.data.columns else self._default_item_columns()
        fields = [
            ("sequence_column", "Sequence column", tk.StringVar(value=sequence_default)),
            ("min_support", "Minimum support", tk.StringVar(value="0.20")),
            ("max_len", "Maximum pattern length", tk.StringVar(value="5")),
            ("top_n", "Top N results", tk.StringVar(value="100"))
        ]
        values = self._ask_field_dialog("Sequential Pattern Parameters", fields)
        if not values:
            return None
        try:
            min_support = float(values["min_support"])
            max_len = int(values["max_len"])
            top_n = int(values["top_n"])
        except ValueError:
            messagebox.showerror("Error", "Thresholds and limits must be numeric.")
            return None
        if not 0 < min_support <= 1:
            messagebox.showerror("Error", "Minimum support must be greater than 0 and at most 1.")
            return None
        if max_len < 2:
            messagebox.showerror("Error", "Maximum pattern length must be at least 2.")
            return None
        if top_n < 1:
            messagebox.showerror("Error", "Top N results must be at least 1.")
            return None
        return {"sequence_column": values["sequence_column"], "min_support": min_support, "max_len": max_len, "top_n": top_n}

    def perform_sequential_patterns(self):
        params = self._ask_sequence_parameters()
        if not params:
            return
        try:
            database = self._build_sequence_database(params["sequence_column"])
            patterns = self._prefixspan_frequent_sequences(database, params["min_support"], params["max_len"])
            if patterns.empty:
                messagebox.showinfo("No Results", "No frequent sequential patterns found.")
                return

            top = patterns.head(params["top_n"])
            figure, axes = plt.subplots(figsize=(5, 4))
            chart_data = top.head(10).sort_values("support")
            axes.barh(chart_data["sequence"], chart_data["support"], color="steelblue")
            axes.set_xlabel("Support")
            axes.set_title("Top Sequential Patterns")
            figure.tight_layout()
            self.show_report_window({"Sequential Patterns": top}, "Sequential Purchase Patterns", figure)
        except Exception as error:
            messagebox.showerror("Sequential Pattern Mining Failed", str(error))

    def _ask_rfm_parameters(self):
        if self.data is None:
            messagebox.showerror("Error", "No data loaded!")
            return None
        fields = [
            ("customer_column", "Customer ID column", tk.StringVar(value="CustomerID" if "CustomerID" in self.data.columns else "")),
            ("date_column", "Date column", tk.StringVar(value="Date" if "Date" in self.data.columns else "")),
            ("amount_column", "Amount column", tk.StringVar(value="Amount" if "Amount" in self.data.columns else "")),
            ("snapshot_date", "Snapshot date (YYYY-MM-DD, optional)", tk.StringVar(value=""))
        ]
        values = self._ask_field_dialog("RFM Segmentation Parameters", fields)
        if not values:
            return None
        missing = [values[key] for key in ("customer_column", "date_column", "amount_column") if values[key] not in self.data.columns]
        if missing:
            messagebox.showerror("Error", f"Columns not found: {', '.join(missing)}")
            return None
        return values

    def _compute_rfm(self, working, customer_column, date_column, amount_column, snapshot=None):
        working = working[[customer_column, date_column, amount_column]].copy()
        working[date_column] = pd.to_datetime(working[date_column], errors="coerce")
        working[amount_column] = pd.to_numeric(working[amount_column], errors="coerce")
        working = working.dropna(subset=[date_column, amount_column])
        if working.empty:
            raise ValueError("No valid dates and amounts were found.")
        if snapshot is None:
            snapshot = working[date_column].max() + pd.Timedelta(days=1)

        rfm = working.groupby(customer_column).agg(
            Recency=(date_column, lambda dates: (snapshot - dates.max()).days),
            Frequency=(date_column, "count"),
            Monetary=(amount_column, "sum")
        )

        def score(series, ascending):
            ranked = series.rank(method="first", ascending=ascending)
            try:
                return pd.qcut(ranked, 5, labels=[5, 4, 3, 2, 1], duplicates="drop").astype(int)
            except ValueError:
                return pd.Series(3, index=series.index)

        rfm["R_Score"] = score(rfm["Recency"], True)
        rfm["F_Score"] = score(rfm["Frequency"], False)
        rfm["M_Score"] = score(rfm["Monetary"], False)
        rfm["RFM_Score"] = rfm["R_Score"] + rfm["F_Score"] + rfm["M_Score"]

        def segment(row):
            if row["R_Score"] >= 4 and row["F_Score"] >= 4:
                return "Champions"
            if row["R_Score"] >= 3 and row["F_Score"] >= 3:
                return "Loyal"
            if row["R_Score"] >= 3:
                return "New / Potential"
            if row["F_Score"] >= 3:
                return "At Risk"
            return "Hibernating"

        rfm["Segment"] = rfm.apply(segment, axis=1)
        summary = rfm["Segment"].value_counts().rename_axis("Segment").reset_index(name="Customers")
        return rfm, summary

    def perform_rfm_segmentation(self):
        params = self._ask_rfm_parameters()
        if not params:
            return
        try:
            snapshot = None
            if params["snapshot_date"]:
                snapshot = pd.Timestamp(params["snapshot_date"])
                if pd.isna(snapshot):
                    messagebox.showerror("Error", "Snapshot date is invalid.")
                    return

            rfm, summary = self._compute_rfm(
                self.data,
                params["customer_column"],
                params["date_column"],
                params["amount_column"],
                snapshot
            )

            figure, axes = plt.subplots(figsize=(5, 4))
            axes.bar(summary["Segment"], summary["Customers"], color="darkorange")
            axes.set_ylabel("Customers")
            axes.set_title("RFM Segments")
            axes.tick_params(axis="x", rotation=25)
            figure.tight_layout()

            self.show_report_window({"RFM Scores": rfm.reset_index(), "Segment Summary": summary}, "RFM Segmentation", figure)
        except Exception as error:
            messagebox.showerror("RFM Segmentation Failed", str(error))

    def _compute_profit_weighted_rules(self, matrix, amounts, algorithm, min_support, min_confidence, min_lift, max_len, top_n):
        from mlxtend.frequent_patterns import association_rules

        average_amount = float(amounts.mean()) if len(amounts) else 0.0
        frequent = self._mine_frequent_itemsets(algorithm, matrix, min_support, max_len)
        if frequent is None or frequent.empty:
            return pd.DataFrame()

        rules = association_rules(frequent, num_itemsets=len(matrix), metric="confidence", min_threshold=min_confidence)
        rules = rules[rules["lift"] >= min_lift].copy()
        if rules.empty:
            return pd.DataFrame()

        utilities = []
        for _, rule in rules.iterrows():
            union_items = list(set(rule["antecedents"]) | set(rule["consequents"]))
            mask = matrix[union_items].all(axis=1) if union_items else pd.Series(False, index=matrix.index)
            subset_amounts = amounts[mask]
            total_utility = float(subset_amounts.sum())
            avg_utility = float(subset_amounts.mean()) if not subset_amounts.empty else 0.0
            utilities.append({
                "total_utility": round(total_utility, 2),
                "avg_utility": round(avg_utility, 2),
                "utility_lift": round(avg_utility / average_amount, 4) if average_amount else 0.0
            })

        weighted = rules.reset_index(drop=True).join(pd.DataFrame(utilities))
        return weighted.sort_values(["total_utility", "lift"], ascending=False).head(top_n)

    def perform_profit_weighted_rules(self):
        params = self._ask_association_parameters()
        if not params:
            return
        amount_column = params.get("amount_column") or "Amount"
        if amount_column not in self.data.columns:
            messagebox.showerror("Error", f"Amount column '{amount_column}' was not found.")
            return

        try:
            columns = [column.strip() for column in params["columns"].split(",") if column.strip()]
            missing_columns = [column for column in columns if column not in self.data.columns]
            if not columns or missing_columns:
                messagebox.showerror("Error", f"Invalid item columns: {', '.join(missing_columns) if missing_columns else 'none selected'}")
                return

            matrix = self._build_transaction_matrix(columns)
            amounts = pd.to_numeric(self.data.loc[matrix.index, amount_column], errors="coerce").fillna(0)
            weighted = self._compute_profit_weighted_rules(
                matrix,
                amounts,
                params["algorithm"],
                params["min_support"],
                params["min_confidence"],
                params["min_lift"],
                params["max_len"],
                params["top_n"]
            )
            if weighted.empty:
                messagebox.showinfo("No Results", "No profit-weighted rules found at the selected thresholds.")
                return
            display = self._format_rules(weighted)
            figure = self._build_rules_plot(weighted)
            self.show_report_window({"Profit-Weighted Rules": display}, "Profit-Weighted Association Rules", figure)
        except ImportError:
            messagebox.showerror("Missing Dependency", "mlxtend is required to generate association rules.")
        except Exception as error:
            messagebox.showerror("Profit-Weighted Rule Mining Failed", str(error))

    def _ask_recommendation_parameters(self):
        fields = [
            ("columns", "Item columns / item-list column", tk.StringVar(value=self._default_item_columns())),
            ("min_support", "Minimum pair support", tk.StringVar(value="0.05")),
            ("min_confidence", "Minimum confidence", tk.StringVar(value="0.30")),
            ("top_n", "Recommendations per item", tk.StringVar(value="5"))
        ]
        values = self._ask_field_dialog("Recommendation / Affinity Parameters", fields)
        if not values:
            return None
        try:
            min_support = float(values["min_support"])
            min_confidence = float(values["min_confidence"])
            top_n = int(values["top_n"])
        except ValueError:
            messagebox.showerror("Error", "Thresholds and limits must be numeric.")
            return None
        if not 0 < min_support <= 1 or not 0 <= min_confidence <= 1 or top_n < 1:
            messagebox.showerror("Error", "Support and confidence must be in [0, 1]; Top N must be at least 1.")
            return None
        return {"columns": values["columns"], "min_support": min_support, "min_confidence": min_confidence, "top_n": top_n}

    def _build_affinity_plot(self, frame):
        top = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["lift"]).sort_values("lift", ascending=False).head(10)
        figure, axes = plt.subplots(figsize=(5, 4))
        axes.barh(top["item"] + " -> " + top["recommended_item"], top["lift"], color="seagreen")
        axes.invert_yaxis()
        axes.set_xlabel("Lift")
        axes.set_title("Top Affinity Recommendations")
        figure.tight_layout()
        return figure

    def _compute_affinity_recommendations(self, matrix, min_support, min_confidence, top_n):
        binary = matrix.astype(int)
        cooccurrence = binary.T.dot(binary) / len(binary)
        item_support = matrix.mean()
        records = []
        for first in matrix.columns:
            for second in matrix.columns:
                if first == second:
                    continue
                pair_support = float(cooccurrence.loc[first, second])
                if pair_support < min_support:
                    continue
                confidence = pair_support / item_support[first] if item_support[first] else 0.0
                if confidence < min_confidence:
                    continue
                denominator = item_support[first] + item_support[second] - pair_support
                jaccard = pair_support / denominator if denominator else 0.0
                lift = pair_support / (item_support[first] * item_support[second]) if item_support[first] * item_support[second] else 0.0
                records.append({
                    "item": str(first),
                    "recommended_item": str(second),
                    "support": round(pair_support, 4),
                    "confidence": round(confidence, 4),
                    "lift": round(lift, 4),
                    "jaccard": round(jaccard, 4)
                })

        if not records:
            return pd.DataFrame()
        frame = pd.DataFrame(records)
        return frame.sort_values(["item", "confidence", "lift"], ascending=[True, False, False]).groupby("item", group_keys=False).head(top_n).reset_index(drop=True)

    def perform_affinity_recommendations(self):
        params = self._ask_recommendation_parameters()
        if not params:
            return
        try:
            columns = [column.strip() for column in params["columns"].split(",") if column.strip()]
            missing_columns = [column for column in columns if column not in self.data.columns]
            if not columns or missing_columns:
                messagebox.showerror("Error", f"Invalid item columns: {', '.join(missing_columns) if missing_columns else 'none selected'}")
                return

            matrix = self._build_transaction_matrix(columns)
            if matrix.shape[1] < 2:
                messagebox.showerror("Error", "At least two distinct items are required.")
                return

            frame = self._compute_affinity_recommendations(matrix, params["min_support"], params["min_confidence"], params["top_n"])
            if frame.empty:
                messagebox.showinfo("No Results", "No item pairs matched the selected thresholds.")
                return
            figure = self._build_affinity_plot(frame)
            self.show_report_window({"Recommendations": frame}, "Recommendation / Affinity Report", figure)
        except Exception as error:
            messagebox.showerror("Affinity Analysis Failed", str(error))

    def _binary_target_values(self, values):
        def convert(value):
            if pd.isna(value):
                return np.nan
            text = str(value).strip().lower()
            if text in ("1", "true", "yes", "y", "churn", "responded", "converted"):
                return 1
            if text in ("0", "false", "no", "n", "retained", "not churn", "not responded"):
                return 0
            try:
                return 1 if float(text) >= 1 else 0
            except ValueError:
                return np.nan

        return pd.to_numeric(pd.Series(values).map(convert), errors="coerce")

    def _compute_clv(self, working, customer_column, date_column, amount_column, retention=0.95, horizon=12):
        frame = working[[customer_column, date_column, amount_column]].copy()
        frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce")
        frame[amount_column] = pd.to_numeric(frame[amount_column], errors="coerce")
        frame = frame.dropna(subset=[date_column, amount_column])
        if frame.empty:
            raise ValueError("No valid dates and amounts were found.")
        grouped = frame.groupby(customer_column).agg(
            FirstPurchase=(date_column, "min"),
            LastPurchase=(date_column, "max"),
            Orders=(date_column, "count"),
            HistoricalValue=(amount_column, "sum")
        )
        grouped["LifespanMonths"] = ((grouped["LastPurchase"] - grouped["FirstPurchase"]).dt.days / 30.4 + 1).clip(lower=1)
        grouped["AvgOrderValue"] = grouped["HistoricalValue"] / grouped["Orders"]
        grouped["PurchaseFrequency"] = grouped["Orders"] / grouped["LifespanMonths"]
        grouped["CLV"] = grouped["AvgOrderValue"] * grouped["PurchaseFrequency"] * horizon * retention
        result = grouped.reset_index().sort_values("CLV", ascending=False)
        summary = pd.DataFrame([
            {"Metric": "Customers", "Value": len(result)},
            {"Metric": "Historical value", "Value": round(float(result["HistoricalValue"].sum()), 2)},
            {"Metric": "Average CLV", "Value": round(float(result["CLV"].mean()), 2)},
            {"Metric": "Top customer CLV", "Value": round(float(result["CLV"].max()), 2)},
            {"Metric": "Retention rate", "Value": retention},
            {"Metric": "Horizon months", "Value": horizon},
        ])
        return result, summary

    def perform_clv_analysis(self):
        if self.data is None:
            messagebox.showerror("Error", "No data loaded!")
            return
        customer_default = "CustomerID" if "CustomerID" in self.data.columns else str(self.data.columns[0])
        date_default = "Date" if "Date" in self.data.columns else str(self.data.columns[0])
        amount_default = "Amount" if "Amount" in self.data.columns else str(self.data.columns[-1])
        fields = (
            ("customer", "Customer column", tk.StringVar(value=customer_default)),
            ("date", "Date column", tk.StringVar(value=date_default)),
            ("amount", "Amount column", tk.StringVar(value=amount_default)),
            ("retention", "Retention rate (0 to 1)", tk.StringVar(value="0.95")),
            ("horizon", "Forecast horizon in months", tk.StringVar(value="12")),
        )
        params = self._ask_field_dialog("Customer Lifetime Value (CLV)", fields)
        if not params:
            return
        try:
            retention = float(params["retention"])
            horizon = int(params["horizon"])
            if not 0 < retention <= 1 or horizon < 1:
                raise ValueError("Retention must be between 0 and 1 and horizon must be positive.")
            result, summary = self._compute_clv(self.data, params["customer"], params["date"], params["amount"], retention, horizon)
            top = result.head(10)
            figure, axes = plt.subplots(figsize=(6, 4))
            axes.bar(top[params["customer"]].astype(str), top["CLV"], color="seagreen")
            axes.set_ylabel("Estimated CLV")
            axes.set_title("Top Customers by Lifetime Value")
            axes.tick_params(axis="x", rotation=30)
            figure.tight_layout()
            self.show_report_window({"Customer CLV": result, "Summary": summary}, "Customer Lifetime Value (CLV)", figure)
        except Exception as error:
            messagebox.showerror("CLV Analysis Failed", str(error))

    def _compute_churn_prediction(self, working, customer_column, target_column, features, test_size=0.25):
        missing = [column for column in [customer_column, target_column] + features if column not in working.columns]
        if missing:
            raise ValueError(f"Columns not found: {', '.join(missing)}")
        if not features:
            raise ValueError("Select at least one numeric feature column.")
        frame = working[[customer_column, target_column] + features].copy()
        frame["__Target"] = self._binary_target_values(frame[target_column]).to_numpy()
        for feature in features:
            frame[feature] = pd.to_numeric(frame[feature], errors="coerce")
        frame = frame.dropna(subset=["__Target"] + features)
        if frame["__Target"].nunique() != 2 or len(frame) < 10:
            raise ValueError("Churn prediction needs both target classes and at least 10 complete rows.")
        features_frame = frame[features]
        target = frame["__Target"].astype(int)
        stratify = target if target.value_counts().min() >= 2 else None
        x_train, x_test, y_train, y_test = train_test_split(features_frame, target, test_size=test_size, random_state=42, stratify=stratify)
        model = LogisticRegression(max_iter=1000)
        model.fit(x_train, y_train)
        probability = model.predict_proba(x_test)[:, 1]
        prediction = (probability >= 0.5).astype(int)
        from sklearn.metrics import roc_auc_score
        result = pd.DataFrame({
            customer_column: frame.loc[x_test.index, customer_column].to_numpy(),
            "Actual": y_test.to_numpy(),
            "Predicted": prediction,
            "ChurnProbability": np.round(probability, 4),
        }).reset_index(drop=True)
        coefficients = pd.DataFrame({"Feature": features, "Coefficient": np.round(model.coef_[0], 4)})
        summary = pd.DataFrame([
            {"Metric": "Rows used", "Value": len(frame)},
            {"Metric": "Accuracy", "Value": round(float((prediction == y_test.to_numpy()).mean()), 4)},
            {"Metric": "ROC AUC", "Value": round(float(roc_auc_score(y_test, probability)), 4)},
            {"Metric": "Churn rate", "Value": round(float(target.mean()), 4)},
        ])
        return result, coefficients, summary

    def perform_churn_prediction(self):
        if self.data is None:
            messagebox.showerror("Error", "No data loaded!")
            return
        customer_default = "CustomerID" if "CustomerID" in self.data.columns else str(self.data.columns[0])
        target_default = "Churn" if "Churn" in self.data.columns else str(self.data.columns[-1])
        excluded = {customer_default, target_default}
        numeric_features = [str(column) for column in self.data.select_dtypes(include=[np.number]).columns if str(column) not in excluded]
        fields = (
            ("customer", "Customer column", tk.StringVar(value=customer_default)),
            ("target", "Churn target column", tk.StringVar(value=target_default)),
            ("features", "Numeric feature columns (comma-separated)", tk.StringVar(value=", ".join(numeric_features[:5]))),
            ("test_size", "Test-set share (0.1 to 0.5)", tk.StringVar(value="0.25")),
        )
        params = self._ask_field_dialog("Churn Prediction", fields)
        if not params:
            return
        try:
            features = [column.strip() for column in params["features"].split(",") if column.strip()]
            test_size = float(params["test_size"])
            if not 0.1 <= test_size <= 0.5:
                raise ValueError("Test-set share must be between 0.1 and 0.5.")
            result, coefficients, summary = self._compute_churn_prediction(self.data, params["customer"], params["target"], features, test_size)
            figure, axes = plt.subplots(figsize=(6, 4))
            axes.barh(coefficients["Feature"], coefficients["Coefficient"], color="indianred")
            axes.axvline(0, color="black", linewidth=0.8)
            axes.set_xlabel("Model coefficient")
            axes.set_title("Churn Drivers")
            figure.tight_layout()
            self.show_report_window({"Predictions": result, "Coefficients": coefficients, "Summary": summary}, "Churn Prediction", figure)
        except Exception as error:
            messagebox.showerror("Churn Prediction Failed", str(error))

    def _compute_campaign_uplift(self, working, target_column, segment_column, value_column=None):
        missing = [column for column in [target_column, segment_column] if column not in working.columns]
        if value_column:
            missing.append(value_column)
        missing = [column for column in missing if column not in working.columns]
        if missing:
            raise ValueError(f"Columns not found: {', '.join(missing)}")
        frame = working[[target_column, segment_column] + ([value_column] if value_column else [])].copy()
        frame["__Target"] = self._binary_target_values(frame[target_column]).to_numpy()
        frame = frame.dropna(subset=["__Target", segment_column])
        if frame.empty or frame["__Target"].nunique() != 2:
            raise ValueError("Campaign uplift needs both response outcomes in the target column.")
        overall_rate = float(frame["__Target"].mean())
        grouped = frame.groupby(segment_column).agg(
            Customers=("__Target", "size"),
            Responders=("__Target", "sum"),
        )
        grouped["ResponseRate"] = grouped["Responders"] / grouped["Customers"]
        grouped["Uplift"] = grouped["ResponseRate"] - overall_rate
        if value_column:
            values = pd.to_numeric(frame[value_column], errors="coerce").fillna(0)
            grouped["ValuePerResponder"] = values.groupby(frame[segment_column]).sum() / grouped["Responders"].replace(0, np.nan)
        result = grouped.reset_index().sort_values("Uplift", ascending=False)
        best = result.iloc[0]
        summary = pd.DataFrame([
            {"Metric": "Overall response rate", "Value": round(overall_rate, 4)},
            {"Metric": "Best segment", "Value": str(best[segment_column])},
            {"Metric": "Best response rate", "Value": round(float(best["ResponseRate"]), 4)},
            {"Metric": "Best uplift", "Value": round(float(best["Uplift"]), 4)},
        ])
        return result, summary

    def perform_campaign_response_analysis(self):
        if self.data is None:
            messagebox.showerror("Error", "No data loaded!")
            return
        segment_default = "CampaignChannel" if "CampaignChannel" in self.data.columns else str(self.data.columns[0])
        target_default = "Response" if "Response" in self.data.columns else str(self.data.columns[-1])
        value_default = "Amount" if "Amount" in self.data.columns else ""
        fields = (
            ("target", "Response target column", tk.StringVar(value=target_default)),
            ("segment", "Campaign segment column", tk.StringVar(value=segment_default)),
            ("value", "Optional value column", tk.StringVar(value=value_default)),
        )
        params = self._ask_field_dialog("Campaign Response Uplift", fields)
        if not params:
            return
        try:
            value_column = params["value"].strip() or None
            result, summary = self._compute_campaign_uplift(self.data, params["target"], params["segment"], value_column)
            figure, axes = plt.subplots(figsize=(6, 4))
            axes.bar(result[params["segment"]].astype(str), result["ResponseRate"], color="royalblue")
            axes.set_ylabel("Response rate")
            axes.set_title("Campaign Response by Segment")
            axes.tick_params(axis="x", rotation=25)
            figure.tight_layout()
            self.show_report_window({"Segment Uplift": result, "Summary": summary}, "Campaign Response Uplift", figure)
        except Exception as error:
            messagebox.showerror("Campaign Response Analysis Failed", str(error))

    def _compute_price_elasticity(self, working, price_column, quantity_column):
        missing = [column for column in (price_column, quantity_column) if column not in working.columns]
        if missing:
            raise ValueError(f"Columns not found: {', '.join(missing)}")
        frame = working[[price_column, quantity_column]].apply(pd.to_numeric, errors="coerce").dropna()
        frame = frame[(frame[price_column] > 0) & (frame[quantity_column] > 0)]
        if len(frame) < 3:
            raise ValueError("Price elasticity needs at least three positive price and quantity pairs.")
        log_price = np.log(frame[price_column].to_numpy(dtype=float))
        log_quantity = np.log(frame[quantity_column].to_numpy(dtype=float))
        elasticity, intercept = np.polyfit(log_price, log_quantity, 1)
        predicted = intercept + elasticity * log_price
        residual_sum = float(np.sum((log_quantity - predicted) ** 2))
        total_sum = float(np.sum((log_quantity - log_quantity.mean()) ** 2))
        r_squared = 1 - residual_sum / total_sum if total_sum else 0.0
        current_revenue = float((frame[price_column] * frame[quantity_column]).sum())
        multiplier = 0.95 if elasticity < -1 else 1.05
        projected_price = frame[price_column].to_numpy(dtype=float) * multiplier
        projected_quantity = np.exp(intercept) * projected_price ** elasticity
        projected_revenue = float((projected_price * projected_quantity).sum())
        summary = pd.DataFrame([
            {"Metric": "Observations", "Value": len(frame)},
            {"Metric": "Price elasticity", "Value": round(float(elasticity), 4)},
            {"Metric": "R-squared", "Value": round(float(r_squared), 4)},
            {"Metric": "Current revenue", "Value": round(current_revenue, 2)},
            {"Metric": "Suggested price multiplier", "Value": multiplier},
            {"Metric": "Projected revenue", "Value": round(projected_revenue, 2)},
            {"Metric": "Projected revenue change (%)", "Value": round((projected_revenue / current_revenue - 1) * 100, 2) if current_revenue else 0.0},
        ])
        return summary, frame.assign(FittedLogQuantity=predicted)

    def perform_price_elasticity_analysis(self):
        if self.data is None:
            messagebox.showerror("Error", "No data loaded!")
            return
        price_default = "Price" if "Price" in self.data.columns else str(self.data.columns[0])
        quantity_default = "Quantity" if "Quantity" in self.data.columns else str(self.data.columns[-1])
        fields = (
            ("price", "Price column", tk.StringVar(value=price_default)),
            ("quantity", "Quantity column", tk.StringVar(value=quantity_default)),
        )
        params = self._ask_field_dialog("Price Elasticity Analysis", fields)
        if not params:
            return
        try:
            summary, fitted = self._compute_price_elasticity(self.data, params["price"], params["quantity"])
            figure, axes = plt.subplots(figsize=(6, 4))
            axes.scatter(np.log(fitted[params["price"]]), np.log(fitted[params["quantity"]]), alpha=0.6)
            order = np.argsort(np.log(fitted[params["price"]]).to_numpy())
            axes.plot(np.log(fitted[params["price"]]).to_numpy()[order], fitted["FittedLogQuantity"].to_numpy()[order], color="darkorange")
            axes.set_xlabel("Log price")
            axes.set_ylabel("Log quantity")
            axes.set_title("Price Elasticity Fit")
            figure.tight_layout()
            self.show_report_window({"Summary": summary}, "Price Elasticity Analysis", figure)
        except Exception as error:
            messagebox.showerror("Price Elasticity Analysis Failed", str(error))

    def show_frequencies(self):
        if self.data is None:
            messagebox.showerror("Error", "No data loaded!")
            return

        selected_columns = simpledialog.askstring("Frequencies", "Enter categorical variables (comma-separated):")
        if not selected_columns:
            return

        selected_columns = [col.strip() for col in selected_columns.split(',') if col.strip() in self.data.columns]
        if not selected_columns:
            messagebox.showerror("Error", "Invalid column names!")
            return

        result = {col: self.data[col].value_counts() for col in selected_columns}
        fig, ax = plt.subplots()
        self.data[selected_columns].apply(pd.Series.value_counts).plot(kind='bar', ax=ax)
        self.show_report_window(result, "Frequency Tables", fig)
    
    def show_summary_statistics(self):
        if self.data is None:
            messagebox.showerror("Error", "No data loaded!")
            return

        selected_columns = simpledialog.askstring("Summary Statistics", "Enter numeric variables (comma-separated):")
        if not selected_columns:
            return

        selected_columns = [col.strip() for col in selected_columns.split(',') if col.strip() in self.data.columns]
        if not selected_columns:
            messagebox.showerror("Error", "Invalid column names!")
            return

        result = self.data[selected_columns].describe()
        fig, ax = plt.subplots()
        self.data[selected_columns].plot(kind='box', ax=ax)
        self.show_report_window(result, "Summary Statistics", fig)

    def show_dataset_properties(self):
        if self.data is None:
            messagebox.showerror("Error", "No data loaded!")
            return
        report = self.dataset_properties(show=False)
        self.show_report_window(report, "Dataset Properties")
        return report

    def perform_ttest(self):
        if self.data is None:
            messagebox.showerror("Error", "No data loaded!")
            return

        selected_columns = simpledialog.askstring("T-Test", "Enter two numeric variables (comma-separated):")
        if not selected_columns:
            return

        selected_columns = [col.strip() for col in selected_columns.split(',') if col.strip() in self.data.columns]
        if len(selected_columns) != 2:
            messagebox.showerror("Error", "Please enter exactly two numeric variables!")
            return

        t_stat, p_value = stats.ttest_ind(self.data[selected_columns[0]], self.data[selected_columns[1]], nan_policy='omit')
        result = pd.DataFrame({"Statistic": [t_stat], "P-Value": [p_value]})
        fig, ax = plt.subplots()
        self.data.boxplot(column=selected_columns, ax=ax)
        self.show_report_window(result, "T-Test Results", fig)
    
    def perform_chisquare(self):
        if self.data is None:
            messagebox.showerror("Error", "No data loaded!")
            return

        selected_columns = simpledialog.askstring("Chi-Square Test", "Enter two categorical variables (comma-separated):")
        if not selected_columns:
            return

        selected_columns = [col.strip() for col in selected_columns.split(',') if col.strip() in self.data.columns]
        if len(selected_columns) != 2:
            messagebox.showerror("Error", "Please enter exactly two categorical variables!")
            return

        contingency_table = pd.crosstab(self.data[selected_columns[0]], self.data[selected_columns[1]])
        chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
        result = pd.DataFrame({"Chi-Square": [chi2], "P-Value": [p], "Degrees of Freedom": [dof]})
        fig, ax = plt.subplots()
        contingency_table.plot(kind='bar', stacked=True, ax=ax)
        self.show_report_window(result, "Chi-Square Test Results", fig)
    
    def perform_normality_tests(self):
        if self.data is None:
            messagebox.showerror("Error", "No data loaded!")
            return

        selected_columns = simpledialog.askstring("Normality Tests", "Enter numeric variables (comma-separated):")
        if not selected_columns:
            return

        selected_columns = [col.strip() for col in selected_columns.split(',') if col.strip() in self.data.columns]
        if not selected_columns:
            messagebox.showerror("Error", "Please enter valid numeric variables!")
            return

        result_dict = {}
        for col in selected_columns:
            shapiro_test = stats.shapiro(self.data[col].dropna())
            ks_test = stats.kstest(self.data[col].dropna(), 'norm')
            result_dict[col] = {
                "Shapiro-Wilk Statistic": shapiro_test.statistic,
                "Shapiro-Wilk P-Value": shapiro_test.pvalue,
                "Kolmogorov-Smirnov Statistic": ks_test.statistic,
                "Kolmogorov-Smirnov P-Value": ks_test.pvalue
            }
        
        result_df = pd.DataFrame(result_dict).T
        fig, ax = plt.subplots()
        self.data[selected_columns].hist(ax=ax)
        self.show_report_window(result_df, "Normality Test Results", fig)

    def perform_anova(self):
        if self.data is None:
            messagebox.showerror("Error", "No data loaded!")
            return

        selected_columns = simpledialog.askstring("ANOVA", "Enter numeric dependent variable and categorical independent variable (comma-separated):")
        if not selected_columns:
            return

        selected_columns = [col.strip() for col in selected_columns.split(',') if col.strip() in self.data.columns]
        if len(selected_columns) != 2:
            messagebox.showerror("Error", "Please enter one numeric and one categorical variable!")
            return

        dependent_var, independent_var = selected_columns
        groups = [self.data[self.data[independent_var] == cat][dependent_var].dropna() for cat in self.data[independent_var].unique()]
        f_stat, p_value = stats.f_oneway(*groups)
        result = pd.DataFrame({"F-Statistic": [f_stat], "P-Value": [p_value]})
        fig, ax = plt.subplots()
        self.data.boxplot(column=[dependent_var], by=independent_var, ax=ax)
        self.show_report_window(result, "ANOVA Results", fig)
    
    def perform_manova(self):
        if self.data is None:
            messagebox.showerror("Error", "No data loaded!")
            return

        selected_columns = simpledialog.askstring("MANOVA", "Enter dependent variables (comma-separated) and independent variable:")
        if not selected_columns:
            return

        selected_columns = [col.strip() for col in selected_columns.split(',') if col.strip() in self.data.columns]
        if len(selected_columns) < 2:
            messagebox.showerror("Error", "Please enter at least one dependent variable and one independent variable!")
            return

        dependent_vars = selected_columns[:-1]
        independent_var = selected_columns[-1]
        
        formula = ' + '.join(dependent_vars) + ' ~ ' + independent_var
        model = MANOVA.from_formula(formula, data=self.data)
        result = model.mv_test()
        result_df = pd.DataFrame(result.results)
        fig, ax = plt.subplots()
        pd.plotting.scatter_matrix(self.data[dependent_vars], ax=ax, diagonal='hist')
        self.show_report_window(result_df, "MANOVA Results", fig)

    def perform_efa(self):
        if self.data is None:
            messagebox.showerror("Error", "No data loaded!")
            return

        selected_columns = simpledialog.askstring("Exploratory Factor Analysis", "Enter numeric variables (comma-separated):")
        if not selected_columns:
            return

        selected_columns = [col.strip() for col in selected_columns.split(',') if col.strip() in self.data.columns]
        if not selected_columns:
            messagebox.showerror("Error", "Please enter valid numeric variables!")
            return

        df = self.data[selected_columns].dropna()
        chi_square_value, p_value = calculate_bartlett_sphericity(df)
        kmo_all, kmo_model = calculate_kmo(df)
        
        fa = FactorAnalyzer(n_factors=2, rotation="varimax")
        fa.fit(df)
        result_df = pd.DataFrame(fa.loadings_, index=selected_columns, columns=["Factor 1", "Factor 2"])
        
        fig, ax = plt.subplots()
        ax.matshow(fa.loadings_, cmap="coolwarm")
        ax.set_xticks(range(len(result_df.columns)))
        ax.set_xticklabels(result_df.columns)
        ax.set_yticks(range(len(result_df.index)))
        ax.set_yticklabels(result_df.index)
        
        scree_fig, scree_ax = plt.subplots()
        scree_ax.plot(range(1, len(fa.get_eigenvalues()[0]) + 1), fa.get_eigenvalues()[0], marker='o', linestyle='--')
        scree_ax.set_title("Scree Plot")
        scree_ax.set_xlabel("Factors")
        scree_ax.set_ylabel("Eigenvalue")
        
        bartlett_kmo_results = pd.DataFrame({
            "Bartlett's Chi-Square": [chi_square_value],
            "Bartlett's P-Value": [p_value],
            "KMO Overall": [kmo_model]
        })
        
        self.show_report_window(result_df, "Exploratory Factor Analysis", fig)
        self.show_report_window(bartlett_kmo_results, "Bartlett's & KMO Test Results", scree_fig)
    
    def perform_cfa(self):
        if self.data is None:
            messagebox.showerror("Error", "No data loaded!")
            return
        
        num_factors = simpledialog.askinteger("Confirmatory Factor Analysis", "Enter the number of factors:")
        if not num_factors or num_factors < 1:
            return
        
        factor_model = []
        selected_columns = set()
        
        for i in range(num_factors):
            factor_name = simpledialog.askstring("Factor Name", f"Enter name for Factor {i+1}:")
            if not factor_name:
                return
            
            variables = simpledialog.askstring("Factor Variables", f"Enter variables for {factor_name} (comma-separated):")
            if not variables:
                return
            
            var_list = [var.strip() for var in variables.split(',') if var.strip() in self.data.columns]
            if not var_list:
                messagebox.showerror("Error", "Invalid variables selected!")
                return
            
            factor_model.append(f"{factor_name} =~ {' + '.join(var_list)}")
            selected_columns.update(var_list)
        
        model_description = "\n".join(factor_model)
        
        try:
            df = self.data[list(selected_columns)].dropna()
            model = Model(model_description)
            model.fit(df)
            result_df = pd.DataFrame(model.inspect())
        except Exception as e:
            messagebox.showerror("Error", f"CFA Model error: {e}")
            return
        
        # Create SEM Path Diagram using networkx
        fig, ax = plt.subplots()
        G = nx.DiGraph()
        
        for line in factor_model:
            parts = line.split('=~')
            if len(parts) == 2:
                latent, observed_vars = parts[0].strip(), parts[1].split('+')
                G.add_node(latent, color='red')
                for obs in observed_vars:
                    obs = obs.strip()
                    G.add_node(obs, color='blue')
                    G.add_edge(latent, obs)
        
        pos = nx.spring_layout(G)
        colors = ['red' if node in [f.split('=~')[0].strip() for f in factor_model] else 'blue' for node in G.nodes]
        nx.draw(G, pos, with_labels=True, node_color=colors, edge_color='gray', ax=ax)
        ax.set_title("CFA Structural Equation Model")
        
        self.show_report_window(result_df, "Confirmatory Factor Analysis", fig)

    def perform_correlation_analysis(self):
        analysis_type = simpledialog.askstring("Correlation Analysis", "Choose analysis type: bivariate, multivariate, canonical")
        if not analysis_type:
            return
        
        if analysis_type == "bivariate":
            selected_columns = simpledialog.askstring("Bivariate Correlation", "Enter two numeric variables (comma-separated):")
            if not selected_columns or ',' not in selected_columns:
                messagebox.showerror("Error", "Please enter exactly two variables separated by a comma.")
                return
            col1, col2 = [col.strip() for col in selected_columns.split(',') if col.strip() in self.data.columns]
            if len({col1, col2}) < 2:
                messagebox.showerror("Error", "Invalid variable selection!")
                return
            corr_coeff, p_value = stats.pearsonr(self.data[col1], self.data[col2])
            result = pd.DataFrame({"Correlation Coefficient": [corr_coeff], "P-Value": [p_value]})
            fig, ax = plt.subplots()
            ax.scatter(self.data[col1], self.data[col2])
            ax.set_title("Bivariate Correlation")
        
        elif analysis_type == "multivariate":
            selected_columns = simpledialog.askstring("Multivariate Correlation", "Enter multiple numeric variables (comma-separated):")
            columns = [col.strip() for col in selected_columns.split(',') if col.strip() in self.data.columns]
            if not columns:
                messagebox.showerror("Error", "Invalid variable selection!")
                return
            result = self.data[columns].corr()
            fig, ax = plt.subplots()
            cax = ax.matshow(result, cmap="coolwarm")
            plt.colorbar(cax)
        
        elif analysis_type == "canonical":
            selected_columns = simpledialog.askstring("Canonical Correlation", "Enter two sets of variables (comma-separated, separated by |):")
            if not selected_columns or '|' not in selected_columns:
                messagebox.showerror("Error", "Please enter two sets of variables separated by '|'")
                return
            set1, set2 = selected_columns.split('|')
            set1 = [col.strip() for col in set1.split(',') if col.strip() in self.data.columns]
            set2 = [col.strip() for col in set2.split(',') if col.strip() in self.data.columns]
            if not set1 or not set2:
                messagebox.showerror("Error", "Invalid variable selection!")
                return
            X, Y = self.data[set1], self.data[set2]
            cca_model = CCA()
            cca_model.fit(X, Y)
            canonical_scores = cca_model.score(X, Y)
            result = pd.DataFrame({"Canonical Correlation Score": [canonical_scores]})
            fig, ax = plt.subplots()
            ax.plot(cca_model.transform(X)[:, 0], marker='o')
        
        self.show_report_window(result, "Correlation Analysis", fig)        
    
    def perform_regression_analysis(self):
        analysis_type = simpledialog.askstring("Regression Analysis", "Choose analysis type: simple, multiple, glm")
        if not analysis_type:
            return
        
        selected_columns = simpledialog.askstring("Regression Analysis", "Enter dependent variable and independent variables (comma-separated):")
        columns = [col.strip() for col in selected_columns.split(',') if col.strip() in self.data.columns]
        if len(columns) < 2:
            messagebox.showerror("Error", "Please enter at least one dependent and one independent variable.")
            return
        dependent_var, independent_vars = columns[0], columns[1:]
        X = sm.add_constant(self.data[independent_vars])
        Y = self.data[dependent_var]
        
        if analysis_type == "simple" and len(independent_vars) == 1:
            model = sm.OLS(Y, X).fit()
        elif analysis_type == "multiple" and len(independent_vars) > 1:
            model = sm.OLS(Y, X).fit()
        elif analysis_type == "glm":
            model = sm.GLM(Y, X, family=sm.families.Gaussian()).fit()
        else:
            messagebox.showerror("Error", "Invalid regression type for the selected variables.")
            return
        
        result = pd.DataFrame(model.summary().tables[1])
        fig, ax = plt.subplots()
        ax.scatter(model.fittedvalues, Y)
        ax.set_xlabel('Fitted Values')
        ax.set_ylabel('Observed Values')
        ax.axhline(y=0, color='red', linestyle='--')
        ax.set_title(f"{analysis_type.capitalize()} Regression Residual Plot")
        
        self.show_report_window(result, "Regression Analysis", fig)

    def _select_numeric_columns(self, prompt, title):
        selected = simpledialog.askstring(title, prompt)
        if not selected:
            return []
        columns = [column.strip() for column in selected.split(",") if column.strip()]
        if self.data is None:
            return []
        return [column for column in columns if column in self.data.columns and pd.api.types.is_numeric_dtype(self.data[column])]

    def perform_kmeans(self):
        if self.data is None:
            messagebox.showerror("Error", "No data loaded!")
            return
        num_clusters = simpledialog.askinteger("K-Means Clustering", "Enter number of clusters:", minvalue=2)
        if not num_clusters:
            return
        columns = self._select_numeric_columns("Enter numeric variables (comma-separated):", "K-Means Clustering")
        if not columns:
            messagebox.showerror("Error", "Invalid variable selection!")
            return
        frame = self.data[columns].dropna()
        if len(frame) < num_clusters:
            messagebox.showerror("Error", "Number of clusters cannot exceed the number of valid rows.")
            return
        model = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(frame)
        self.data.loc[frame.index, "Cluster"] = labels
        self.display_data()
        figure, axes = plt.subplots(figsize=(8, 5))
        if len(columns) >= 2:
            scatter = axes.scatter(frame[columns[0]], frame[columns[1]], c=labels, cmap="viridis")
            figure.colorbar(scatter, ax=axes)
            axes.set_xlabel(columns[0])
            axes.set_ylabel(columns[1])
        else:
            scatter = axes.scatter(frame.index, frame[columns[0]], c=labels, cmap="viridis")
            figure.colorbar(scatter, ax=axes)
            axes.set_xlabel("Row ID")
            axes.set_ylabel(columns[0])
        axes.set_title("K-Means Clustering")
        figure.tight_layout()
        report = {
            "Cluster Centers": pd.DataFrame(model.cluster_centers_, columns=columns),
            "Cluster Counts": pd.Series(labels).value_counts().sort_index().rename("Count").to_frame(),
            "Model Details": pd.DataFrame({"Property": ["Algorithm", "Clusters", "Rows Used"], "Value": ["K-Means", num_clusters, len(frame)]})
        }
        self.show_report_window(report, "K-Means Clustering", fig=figure)

    def perform_hierarchical(self):
        if self.data is None:
            messagebox.showerror("Error", "No data loaded!")
            return
        columns = self._select_numeric_columns("Enter numeric variables (comma-separated):", "Hierarchical Clustering")
        if not columns:
            messagebox.showerror("Error", "Invalid variable selection!")
            return
        frame = self.data[columns].dropna()
        if len(frame) < 2:
            messagebox.showerror("Error", "At least two valid rows are required.")
            return
        linkage_matrix = linkage(frame, method="ward")
        cluster_count = min(3, len(frame))
        cluster_labels = fcluster(linkage_matrix, t=cluster_count, criterion="maxclust")
        self.data.loc[frame.index, "Cluster"] = cluster_labels
        self.display_data()
        figure, axes = plt.subplots(figsize=(9, 6))
        dendrogram(linkage_matrix, labels=[str(index) for index in frame.index], ax=axes)
        axes.set_title("Hierarchical Clustering Dendrogram")
        axes.set_ylabel("Distance")
        figure.tight_layout()
        report = {
            "Linkage Matrix": pd.DataFrame(linkage_matrix, columns=["Cluster 1", "Cluster 2", "Distance", "Sample Count"]),
            "Cluster Counts": pd.Series(cluster_labels).value_counts().sort_index().rename("Count").to_frame(),
            "Model Details": pd.DataFrame({"Property": ["Method", "Linkage Rows", "Cut Clusters"], "Value": ["Ward", len(linkage_matrix), cluster_count]})
        }
        self.show_report_window(report, "Hierarchical Clustering", fig=figure)

    def perform_dbscan(self):
        if self.data is None:
            messagebox.showerror("Error", "No data loaded!")
            return
        epsilon = simpledialog.askfloat("DBSCAN Clustering", "Enter eps (neighborhood radius):", minvalue=0.01, initialvalue=0.5)
        min_samples = simpledialog.askinteger("DBSCAN Clustering", "Enter minimum samples:", minvalue=2, initialvalue=5)
        columns = self._select_numeric_columns("Enter numeric variables (comma-separated):", "DBSCAN Clustering")
        if not epsilon or not min_samples or not columns:
            return
        frame = self.data[columns].dropna()
        scaled = StandardScaler().fit_transform(frame)
        model = DBSCAN(eps=epsilon, min_samples=min_samples)
        labels = model.fit_predict(scaled)
        self.data.loc[frame.index, "Cluster"] = labels
        self.display_data()
        figure, axes = plt.subplots(figsize=(8, 5))
        if len(columns) >= 2:
            scatter = axes.scatter(frame[columns[0]], frame[columns[1]], c=labels, cmap="viridis")
            axes.set_xlabel(columns[0])
            axes.set_ylabel(columns[1])
        else:
            scatter = axes.scatter(frame.index, frame[columns[0]], c=labels, cmap="viridis")
            axes.set_xlabel("Row ID")
            axes.set_ylabel(columns[0])
        figure.colorbar(scatter, ax=axes)
        axes.set_title("DBSCAN Clustering")
        figure.tight_layout()
        centers = frame.assign(Cluster=labels).groupby("Cluster")[columns].mean()
        report = {
            "Cluster Centers": centers,
            "Cluster Counts": pd.Series(labels).value_counts().sort_index().rename("Count").to_frame(),
            "Model Details": pd.DataFrame({"Property": ["Algorithm", "Eps", "Minimum Samples", "Noise Points"], "Value": ["DBSCAN", epsilon, min_samples, int((labels == -1).sum())]})
        }
        self.show_report_window(report, "DBSCAN Clustering", fig=figure)

    def perform_gaussian_mixture(self):
        if self.data is None:
            messagebox.showerror("Error", "No data loaded!")
            return
        components = simpledialog.askinteger("Gaussian Mixture", "Enter number of components:", minvalue=2, initialvalue=3)
        columns = self._select_numeric_columns("Enter numeric variables (comma-separated):", "Gaussian Mixture")
        if not components or not columns:
            return
        frame = self.data[columns].dropna()
        if len(frame) < components:
            messagebox.showerror("Error", "Components cannot exceed the number of valid rows.")
            return
        model = GaussianMixture(n_components=components, random_state=42, n_init=5)
        labels = model.fit_predict(frame)
        self.data.loc[frame.index, "Cluster"] = labels
        self.display_data()
        figure, axes = plt.subplots(figsize=(8, 5))
        if len(columns) >= 2:
            scatter = axes.scatter(frame[columns[0]], frame[columns[1]], c=labels, cmap="viridis", alpha=0.7)
            axes.set_xlabel(columns[0])
            axes.set_ylabel(columns[1])
            for index, center in enumerate(model.means_):
                covariance = model.covariances_[index][:2, :2]
                eigenvalues, eigenvectors = np.linalg.eigh(covariance)
                order = eigenvalues.argsort()[::-1]
                eigenvalues = eigenvalues[order]
                eigenvectors = eigenvectors[:, order]
                angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
                ellipse = Ellipse(center[:2], width=4 * np.sqrt(eigenvalues[0]), height=4 * np.sqrt(eigenvalues[1]), angle=angle, fill=False, linewidth=2)
                axes.add_patch(ellipse)
                axes.scatter(center[0], center[1], marker="x", s=120, c="red")
        else:
            scatter = axes.scatter(frame.index, frame[columns[0]], c=labels, cmap="viridis", alpha=0.7)
            axes.set_xlabel("Row ID")
            axes.set_ylabel(columns[0])
        figure.colorbar(scatter, ax=axes)
        axes.set_title("Gaussian Mixture Clustering")
        figure.tight_layout()
        report = {
            "Component Means": pd.DataFrame(model.means_, columns=columns),
            "Component Weights": pd.DataFrame({"Component": range(components), "Weight": model.weights_}),
            "Cluster Counts": pd.Series(labels).value_counts().sort_index().rename("Count").to_frame(),
            "Model Details": pd.DataFrame({"Property": ["Algorithm", "Components", "Converged", "Iterations"], "Value": ["Gaussian Mixture", components, model.converged_, model.n_iter_]})
        }
        self.show_report_window(report, "Gaussian Mixture", fig=figure)

    def perform_pca(self):
        if self.data is None:
            messagebox.showerror("Error", "No data loaded!")
            return
        columns = self._select_numeric_columns("Enter numeric variables (comma-separated):", "Principal Component Analysis")
        if len(columns) < 2:
            messagebox.showerror("Error", "PCA requires at least two numeric variables.")
            return
        requested = simpledialog.askinteger("Principal Component Analysis", "Enter number of components:", minvalue=2, initialvalue=2)
        if not requested:
            return
        frame = self.data[columns].dropna()
        component_count = min(requested, len(columns), len(frame))
        scaled = StandardScaler().fit_transform(frame)
        model = PCA(n_components=component_count)
        scores = model.fit_transform(scaled)
        score_columns = [f"PC{index + 1}" for index in range(component_count)]
        score_frame = pd.DataFrame(scores, columns=score_columns, index=frame.index)
        self.data.loc[frame.index, score_columns] = score_frame
        self.display_data()
        variance_percent = model.explained_variance_ratio_ * 100
        variance_frame = pd.DataFrame({
            "Component": score_columns,
            "Explained Variance (%)": variance_percent,
            "Cumulative (%)": np.cumsum(variance_percent)
        })
        components = pd.DataFrame(model.components_, columns=columns, index=score_columns)
        scree_figure, scree_axes = plt.subplots(figsize=(7, 4))
        scree_axes.bar(score_columns, variance_percent, color="steelblue")
        scree_axes.set_title("PCA Scree Plot")
        scree_axes.set_ylabel("Explained Variance (%)")
        scree_figure.tight_layout()
        score_figure, score_axes = plt.subplots(figsize=(7, 5))
        score_axes.scatter(score_frame[score_columns[0]], score_frame[score_columns[1]], alpha=0.75)
        score_axes.set_xlabel(score_columns[0])
        score_axes.set_ylabel(score_columns[1])
        score_axes.set_title("PCA Score Plot")
        score_figure.tight_layout()
        report = {
            "Explained Variance": variance_frame,
            "Components": components,
            "Scores (First Rows)": score_frame.head(20)
        }
        self.show_report_window(report, "Principal Component Analysis", figures=[scree_figure, score_figure])

    def perform_stationarity_tests(self):
        selected_column = simpledialog.askstring("Stationarity Tests", "Enter the time series variable:")
        if not selected_column or selected_column not in self.data.columns:
            messagebox.showerror("Error", "Invalid variable selection!")
            return
        
        df = self.data[[selected_column]].dropna()
        adf_test = adfuller(df[selected_column])
        kpss_test = kpss(df[selected_column], regression='c')
        
        result = pd.DataFrame({
            "Test": ["ADF Test", "KPSS Test"],
            "Statistic": [adf_test[0], kpss_test[0]],
            "P-Value": [adf_test[1], kpss_test[1]]
        })
        
        fig, ax = plt.subplots()
        df[selected_column].plot(ax=ax, title="Time Series Plot")
        
        self.show_report_window(result, "Stationarity Tests", fig)

    def perform_seasonal_decomposition(self):
        selected_column = simpledialog.askstring("Seasonal Decomposition", "Enter the time series variable:")
        if not selected_column or selected_column not in self.data.columns:
            messagebox.showerror("Error", "Invalid variable selection!")
            return
        
        df = self.data[[selected_column]].dropna()
        result = seasonal_decompose(df, model='additive', period=12)
        
        fig, axes = plt.subplots(4, 1, figsize=(10, 8))
        result.observed.plot(ax=axes[0], title='Observed')
        result.trend.plot(ax=axes[1], title='Trend')
        result.seasonal.plot(ax=axes[2], title='Seasonal')
        result.resid.plot(ax=axes[3], title='Residual')
        
        self.show_report_window(pd.DataFrame({'Trend': result.trend, 'Seasonal': result.seasonal, 'Residual': result.resid}), "Seasonal Decomposition", fig)

    
    def perform_holt_winters(self):
        selected_column = simpledialog.askstring("Holt-Winters Method", "Enter the time series variable:")
        if not selected_column or selected_column not in self.data.columns:
            messagebox.showerror("Error", "Invalid variable selection!")
            return
        
        df = self.data[[selected_column]].dropna()
        model = ExponentialSmoothing(df, trend='add', seasonal='add', seasonal_periods=12).fit()
        df['Forecast'] = model.fittedvalues
        
        mse = np.mean((df[selected_column] - df['Forecast'])**2)
        mae = np.mean(np.abs(df[selected_column] - df['Forecast']))
        rmse = np.sqrt(mse)
        metrics = pd.DataFrame({'Metric': ['MSE', 'MAE', 'RMSE'], 'Value': [mse, mae, rmse]})
        
        fig, ax = plt.subplots()
        df[selected_column].plot(ax=ax, label='Original')
        df['Forecast'].plot(ax=ax, label='Holt-Winters Forecast', linestyle='dashed')
        ax.legend()
        
        self.show_report_window(metrics, "Holt-Winters - Fitness Measures", fig)
    
    def perform_moving_averages(self):
        selected_column = simpledialog.askstring("Moving Averages", "Enter the time series variable:")
        window_size = simpledialog.askinteger("Moving Averages", "Enter window size:")
        if not selected_column or selected_column not in self.data.columns or not window_size:
            messagebox.showerror("Error", "Invalid input!")
            return
        
        df = self.data[[selected_column]].dropna()
        df['Moving Average'] = df[selected_column].rolling(window=window_size).mean()
        
        mse = np.mean((df[selected_column] - df['Moving Average'])**2)
        mae = np.mean(np.abs(df[selected_column] - df['Moving Average']))
        rmse = np.sqrt(mse)
        metrics = pd.DataFrame({'Metric': ['MSE', 'MAE', 'RMSE'], 'Value': [mse, mae, rmse]})
        
        fig, ax = plt.subplots()
        df[selected_column].plot(ax=ax, label='Original')
        df['Moving Average'].plot(ax=ax, label=f'{window_size}-Period Moving Average', linestyle='dashed')
        ax.legend()
        
        self.show_report_window(metrics, "Moving Averages - Fitness Measures", fig)

##    def perform_logistic_regression(self):
##        selected_columns = simpledialog.askstring("Logistic Regression", "Enter dependent variable and independent variables (comma-separated):")
##        columns = [col.strip() for col in selected_columns.split(',') if col.strip() in self.data.columns]
##        if len(columns) < 2:
##            messagebox.showerror("Error", "Please enter at least one dependent and one independent variable.")
##            return
##        dependent_var, independent_vars = columns[0], columns[1:]
##        X_train, X_test, y_train, y_test = train_test_split(self.data[independent_vars], self.data[dependent_var], test_size=0.2, random_state=42)
##        
##        model = LogisticRegression(max_iter=200)
##        model.fit(X_train, y_train)
##        predictions = model.predict(X_test)
##        accuracy = accuracy_score(y_test, predictions)
##        report = classification_report(y_test, predictions)
##        
##        result = pd.DataFrame({"Actual": y_test, "Predicted": predictions})
##        metrics = pd.DataFrame({'Metric': ['Accuracy'], 'Value': [accuracy]})
##        
##        fig, ax = plt.subplots()
##        ax.hist(predictions, bins=10, alpha=0.5, label="Predicted", color='blue')
##        ax.hist(y_test, bins=10, alpha=0.5, label="Actual", color='green')
##        ax.legend()
##        ax.set_title("Logistic Regression Predictions")
##        
##        self.show_report_window(metrics, "Logistic Regression - Accuracy", fig)
##
##    def perform_decision_tree(self):
##        selected_columns = simpledialog.askstring("Decision Tree", "Enter dependent variable and independent variables (comma-separated):")
##        columns = [col.strip() for col in selected_columns.split(',') if col.strip() in self.data.columns]
##        if len(columns) < 2:
##            messagebox.showerror("Error", "Please enter at least one dependent and one independent variable.")
##            return
##        dependent_var, independent_vars = columns[0], columns[1:]
##        X_train, X_test, y_train, y_test = train_test_split(self.data[independent_vars], self.data[dependent_var], test_size=0.2, random_state=42)
##        
##        model = DecisionTreeClassifier()
##        model.fit(X_train, y_train)
##        predictions = model.predict(X_test)
##        accuracy = accuracy_score(y_test, predictions)
##        
##        metrics = pd.DataFrame({'Metric': ['Accuracy'], 'Value': [accuracy]})
##        fig, ax = plt.subplots()
##        ax.hist(predictions, bins=10, alpha=0.5, label="Predicted", color='blue')
##        ax.hist(y_test, bins=10, alpha=0.5, label="Actual", color='green')
##        ax.legend()
##        ax.set_title("Decision Tree Predictions")
##        
##        self.show_report_window(metrics, "Decision Tree - Accuracy", fig)
    
    def perform_random_forest(self):
        selected_columns = simpledialog.askstring("Random Forest", "Enter dependent variable and independent variables (comma-separated):")
        columns = [col.strip() for col in selected_columns.split(',') if col.strip() in self.data.columns]
        if len(columns) < 2:
            messagebox.showerror("Error", "Please enter at least one dependent and one independent variable.")
            return
        dependent_var, independent_vars = columns[0], columns[1:]
        X_train, X_test, y_train, y_test = train_test_split(self.data[independent_vars], self.data[dependent_var], test_size=0.2, random_state=42)
        
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        metrics = pd.DataFrame({'Metric': ['Accuracy'], 'Value': [accuracy]})
        fig, ax = plt.subplots()
        ax.hist(predictions, bins=10, alpha=0.5, label="Predicted", color='blue')
        ax.hist(y_test, bins=10, alpha=0.5, label="Actual", color='green')
        ax.legend()
        ax.set_title("Random Forest Predictions")
        
        self.show_report_window(metrics, "Random Forest - Accuracy", fig)
    
    def get_train_test_split(self, independent_vars, dependent_var):
        train_size = simpledialog.askfloat("Train-Test Split", "Enter training data ratio (e.g., 0.8 for 80%):", minvalue=0.1, maxvalue=0.9)
        if not train_size:
            train_size = 0.8
        return train_test_split(self.data[independent_vars], self.data[dependent_var], test_size=1-train_size, random_state=42)
    
    def _ml_feature_frame(self, values):
        if isinstance(values, pd.DataFrame):
            return values.reset_index(drop=True).copy()
        return pd.DataFrame(values)

    def _ml_test_set_frame(self, X_test, y_test, predictions):
        features = self._ml_feature_frame(X_test)
        actual_values = list(np.asarray(y_test).ravel())
        predicted_values = list(np.asarray(predictions).ravel())
        row_ids = list(X_test.index) if hasattr(X_test, "index") else list(range(len(actual_values)))
        result = pd.DataFrame({
            "Row ID": row_ids,
            "Actual": actual_values,
            "Predicted": predicted_values,
            "Correct": [str(actual) == str(predicted) for actual, predicted in zip(actual_values, predicted_values)]
        })
        for column in features.columns:
            name = str(column)
            if name in result.columns:
                name = f"Feature {name}"
            result[name] = features[column].values
        return result

    def _ml_confusion_frame(self, y_test, predictions):
        actual_values = np.asarray(y_test).ravel()
        predicted_values = np.asarray(predictions).ravel()
        labels = list(pd.unique(np.concatenate([actual_values, predicted_values])))
        predicted_columns = [f"Predicted {label}" for label in labels]
        matrix = confusion_matrix(actual_values, predicted_values, labels=labels)
        frame = pd.DataFrame(matrix, index=[f"Actual {label}" for label in labels], columns=predicted_columns)
        frame["Total Actual"] = frame.sum(axis=1)
        totals = frame[predicted_columns].sum(axis=0)
        frame.loc["Total Predicted"] = [int(totals[column]) for column in predicted_columns] + [len(actual_values)]
        return frame

    def _ml_technique_frame(self, model, title, X_test, y_test, predictions):
        properties = {
            "Technique": title,
            "Model": type(model).__name__,
            "Test Rows": len(y_test),
            "Accuracy": accuracy_score(y_test, predictions)
        }
        if hasattr(model, "n_features_in_"):
            properties["Features Used"] = model.n_features_in_
        if isinstance(model, DecisionTreeClassifier):
            properties["Tree Depth"] = model.get_depth()
            properties["Leaf Count"] = model.get_n_leaves()
            names = [str(column) for column in (X_test.columns if hasattr(X_test, "columns") else range(X_test.shape[1]))]
            properties["Tree Rules"] = export_text(model, feature_names=names, max_depth=5)
        elif isinstance(model, RandomForestClassifier):
            properties["Trees"] = model.n_estimators
            properties["Bootstrap"] = model.bootstrap
        elif isinstance(model, LogisticRegression):
            properties["Solver"] = model.solver
            properties["Max Iterations"] = model.max_iter
        elif isinstance(model, GaussianNB):
            properties["Class Priors"] = ", ".join(f"{label}: {value:.3f}" for label, value in zip(model.classes_, model.class_prior_))
        elif isinstance(model, KNeighborsClassifier):
            properties["Neighbors"] = model.n_neighbors
            properties["Weights"] = model.weights
        elif isinstance(model, SVC):
            properties["Kernel"] = model.kernel
            properties["C"] = model.C
            properties["Support Vectors"] = len(model.support_vectors_) if hasattr(model, "support_vectors_") else "unavailable"
        elif isinstance(model, MLPClassifier):
            properties["Hidden Layers"] = model.hidden_layer_sizes
            properties["Iterations"] = getattr(model, "n_iter_", "unavailable")
            properties["Final Loss"] = getattr(model, "loss_", "unavailable")
        return pd.DataFrame({"Property": list(properties), "Value": [str(value) for value in properties.values()]})

    def _ml_two_feature_view(self, X_train, X_test):
        train = self._ml_feature_frame(X_train)
        test = self._ml_feature_frame(X_test)
        numeric_train = train.select_dtypes(include=[np.number])
        if numeric_train.shape[1] < 2:
            return None
        names = list(numeric_train.columns[:2])
        numeric_test = test[names].apply(pd.to_numeric, errors="coerce").fillna(0)
        return numeric_train.iloc[:, :2], numeric_test, names

    def _ml_build_figures(self, model, X_train, X_test, y_train, y_test, predictions, title):
        figures = []
        try:
            confusion_frame = self._ml_confusion_frame(y_test, predictions)
            heatmap_frame = confusion_frame.iloc[:-1, :-1].copy()
            heatmap_frame.columns = [str(column).replace("Predicted ", "") for column in heatmap_frame.columns]
            heatmap_frame.index = [str(index).replace("Actual ", "") for index in heatmap_frame.index]
            figure, axes = plt.subplots(figsize=(7, 5))
            sns.heatmap(heatmap_frame, annot=True, fmt="d", cmap="Blues", ax=axes)
            axes.set_title(f"{title} - Confusion Matrix")
            axes.set_xlabel("Predicted")
            axes.set_ylabel("Actual")
            figure.tight_layout()
            figures.append(figure)
        except Exception:
            pass

        try:
            if isinstance(model, DecisionTreeClassifier):
                feature_names = [str(column) for column in (X_test.columns if hasattr(X_test, "columns") else range(X_test.shape[1]))]
                figure, axes = plt.subplots(figsize=(12, 7))
                plot_tree(model, filled=True, ax=axes, feature_names=feature_names, class_names=[str(label) for label in model.classes_], max_depth=5, fontsize=7)
                axes.set_title(f"{title} - Decision Tree")
                figure.tight_layout()
                figures.append(figure)
        except Exception:
            pass

        try:
            if isinstance(model, RandomForestClassifier) and hasattr(model, "feature_importances_"):
                names = [str(column) for column in (X_test.columns if hasattr(X_test, "columns") else range(len(model.feature_importances_)))]
                values = pd.Series(model.feature_importances_, index=names).sort_values().tail(15)
                figure, axes = plt.subplots(figsize=(8, 5))
                axes.barh(values.index, values.values, color="seagreen")
                axes.set_title(f"{title} - Feature Importances")
                figure.tight_layout()
                figures.append(figure)
        except Exception:
            pass

        try:
            if isinstance(model, LogisticRegression) and model.coef_ is not None:
                names = [str(column) for column in (X_test.columns if hasattr(X_test, "columns") else range(model.coef_.shape[1]))]
                coefficients = np.asarray(model.coef_)
                values = np.mean(np.abs(coefficients), axis=0) if coefficients.ndim > 1 else np.abs(coefficients[0])
                series = pd.Series(values, index=names).sort_values()
                figure, axes = plt.subplots(figsize=(8, 5))
                axes.barh(series.index, series.values, color="steelblue")
                axes.set_title(f"{title} - Mean Absolute Coefficients")
                figure.tight_layout()
                figures.append(figure)
        except Exception:
            pass

        try:
            if isinstance(model, GaussianNB):
                figure, axes = plt.subplots(figsize=(7, 4))
                axes.bar([str(label) for label in model.classes_], model.class_prior_, color="mediumpurple")
                axes.set_title(f"{title} - Class Priors")
                axes.set_ylabel("Prior Probability")
                figure.tight_layout()
                figures.append(figure)
        except Exception:
            pass

        try:
            if isinstance(model, KNeighborsClassifier):
                prepared = self._ml_two_feature_view(X_train, X_test)
                if prepared:
                    train_view, test_view, names = prepared
                    figure, axes = plt.subplots(figsize=(8, 5))
                    train_labels = list(np.asarray(y_train).ravel())
                    for label in np.unique(np.asarray(train_labels)):
                        subset = train_view[np.asarray(train_labels) == label]
                        axes.scatter(subset.iloc[:, 0], subset.iloc[:, 1], label=f"Train {label}", alpha=0.6)
                    axes.scatter(test_view.iloc[:, 0], test_view.iloc[:, 1], marker="s", edgecolors="black", c="orange", label="Test Set")
                    axes.set_xlabel(names[0])
                    axes.set_ylabel(names[1])
                    axes.set_title(f"{title} - Neighborhood View")
                    axes.legend(fontsize=8)
                    figure.tight_layout()
                    figures.append(figure)
        except Exception:
            pass

        try:
            if isinstance(model, SVC):
                prepared = self._ml_two_feature_view(X_train, X_test)
                if prepared:
                    train_view, test_view, names = prepared
                    boundary_model = SVC(**model.get_params())
                    boundary_model.fit(train_view, np.asarray(y_train).ravel())
                    x_min, x_max = train_view.iloc[:, 0].min() - 1, train_view.iloc[:, 0].max() + 1
                    y_min, y_max = train_view.iloc[:, 1].min() - 1, train_view.iloc[:, 1].max() + 1
                    grid_x, grid_y = np.meshgrid(np.linspace(x_min, x_max, 80), np.linspace(y_min, y_max, 80))
                    grid_points = pd.DataFrame({names[0]: grid_x.ravel(), names[1]: grid_y.ravel()})
                    decision_values = boundary_model.decision_function(grid_points)
                    if decision_values.ndim == 1:
                        boundary_values = decision_values.reshape(grid_x.shape)
                    else:
                        boundary_values = np.argmax(decision_values, axis=1).reshape(grid_x.shape)
                    figure, axes = plt.subplots(figsize=(8, 5))
                    axes.contourf(grid_x, grid_y, boundary_values, alpha=0.25, cmap="coolwarm")
                    if hasattr(boundary_model, "support_vectors_"):
                        support = boundary_model.support_vectors_
                        axes.scatter(support[:, 0], support[:, 1], marker="x", c="black", label="Support Vectors")
                    axes.scatter(test_view.iloc[:, 0], test_view.iloc[:, 1], c="orange", edgecolors="black", label="Test Set")
                    axes.set_xlabel(names[0])
                    axes.set_ylabel(names[1])
                    axes.set_title(f"{title} - SVM Decision Boundary")
                    axes.legend(fontsize=8)
                    figure.tight_layout()
                    figures.append(figure)
        except Exception:
            pass

        try:
            if isinstance(model, MLPClassifier):
                if getattr(model, "loss_curve_", None) is not None:
                    figure, axes = plt.subplots(figsize=(8, 4))
                    axes.plot(model.loss_curve_, color="darkorange")
                    axes.set_title(f"{title} - Training Loss Curve")
                    axes.set_xlabel("Iteration")
                    axes.set_ylabel("Loss")
                    figure.tight_layout()
                    figures.append(figure)
                if getattr(model, "coefs_", None) and len(model.coefs_) > 0:
                    figure, axes = plt.subplots(figsize=(8, 4))
                    image = axes.imshow(model.coefs_[0], aspect="auto", cmap="viridis")
                    figure.colorbar(image, ax=axes)
                    axes.set_title(f"{title} - First Layer Weights")
                    figure.tight_layout()
                    figures.append(figure)
        except Exception:
            pass

        return figures

    def _build_classification_report(self, model, X_train, X_test, y_train, y_test, predictions, title, include_figures=True):
        report = pd.DataFrame(classification_report(y_test, predictions, output_dict=True)).transpose()
        report_data = {
            "Test Set Predictions": self._ml_test_set_frame(X_test, y_test, predictions),
            "Confusion Matrix": self._ml_confusion_frame(y_test, predictions),
            "Classification Report": report,
            "Technique Details": self._ml_technique_frame(model, title, X_test, y_test, predictions)
        }
        figures = self._ml_build_figures(model, X_train, X_test, y_train, y_test, predictions, title) if include_figures else []
        return {"report": report_data, "figures": figures}

    def evaluate_model(self, model, X_train, X_test, y_train, y_test, predictions, title):
        payload = self._build_classification_report(model, X_train, X_test, y_train, y_test, predictions, title)
        self.show_report_window(payload["report"], f"{title} - Evaluation", figures=payload["figures"])
        return payload
    
    def perform_logistic_regression(self):
        selected_columns = simpledialog.askstring("Logistic Regression", "Enter dependent and independent variables (comma-separated):")
        columns = [col.strip() for col in selected_columns.split(',') if col.strip() in self.data.columns]
        if len(columns) < 2:
            messagebox.showerror("Error", "Please enter at least one dependent and one independent variable.")
            return
        dependent_var, independent_vars = columns[0], columns[1:]
        X_train, X_test, y_train, y_test = self.get_train_test_split(independent_vars, dependent_var)
        
        model = LogisticRegression(max_iter=200)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        self.evaluate_model(model, X_train, X_test, y_train, y_test, predictions, "Logistic Regression")
    
    def perform_decision_tree(self):
        selected_columns = simpledialog.askstring("Decision Tree", "Enter dependent and independent variables (comma-separated):")
        columns = [col.strip() for col in selected_columns.split(',') if col.strip() in self.data.columns]
        if len(columns) < 2:
            messagebox.showerror("Error", "Please enter at least one dependent and one independent variable.")
            return
        dependent_var, independent_vars = columns[0], columns[1:]
        X_train, X_test, y_train, y_test = self.get_train_test_split(independent_vars, dependent_var)
        
        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        self.evaluate_model(model, X_train, X_test, y_train, y_test, predictions, "Decision Tree")
    
    def perform_random_forest(self):
        selected_columns = simpledialog.askstring("Random Forest", "Enter dependent and independent variables (comma-separated):")
        columns = [col.strip() for col in selected_columns.split(',') if col.strip() in self.data.columns]
        if len(columns) < 2:
            messagebox.showerror("Error", "Please enter at least one dependent and one independent variable.")
            return
        dependent_var, independent_vars = columns[0], columns[1:]
        X_train, X_test, y_train, y_test = self.get_train_test_split(independent_vars, dependent_var)
        
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        self.evaluate_model(model, X_train, X_test, y_train, y_test, predictions, "Random Forest")

    def perform_naive_bayes(self):
        selected_columns = simpledialog.askstring("Naive Bayes", "Enter dependent and independent variables (comma-separated):")
        columns = [col.strip() for col in selected_columns.split(',') if col.strip() in self.data.columns]
        if len(columns) < 2:
            messagebox.showerror("Error", "Please enter at least one dependent and one independent variable.")
            return
        dependent_var, independent_vars = columns[0], columns[1:]
        X_train, X_test, y_train, y_test = self.get_train_test_split(independent_vars, dependent_var)
        
        model = GaussianNB()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        self.evaluate_model(model, X_train, X_test, y_train, y_test, predictions, "Naive Bayes")
    
    def perform_knn(self):
        selected_columns = simpledialog.askstring("K-Nearest Neighbors", "Enter dependent and independent variables (comma-separated):")
        columns = [col.strip() for col in selected_columns.split(',') if col.strip() in self.data.columns]
        if len(columns) < 2:
            messagebox.showerror("Error", "Please enter at least one dependent and one independent variable.")
            return
        dependent_var, independent_vars = columns[0], columns[1:]
        X_train, X_test, y_train, y_test = self.get_train_test_split(independent_vars, dependent_var)
        
        model = KNeighborsClassifier()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        self.evaluate_model(model, X_train, X_test, y_train, y_test, predictions, "K-Nearest Neighbors")
    
    def perform_svm(self):
        selected_columns = simpledialog.askstring("Support Vector Machine", "Enter dependent and independent variables (comma-separated):")
        columns = [col.strip() for col in selected_columns.split(',') if col.strip() in self.data.columns]
        if len(columns) < 2:
            messagebox.showerror("Error", "Please enter at least one dependent and one independent variable.")
            return
        dependent_var, independent_vars = columns[0], columns[1:]
        X_train, X_test, y_train, y_test = self.get_train_test_split(independent_vars, dependent_var)
        
        model = SVC()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        self.evaluate_model(model, X_train, X_test, y_train, y_test, predictions, "Support Vector Machine")
    
    def perform_neural_network(self):
        selected_columns = simpledialog.askstring("Neural Network", "Enter dependent and independent variables (comma-separated):")
        columns = [col.strip() for col in selected_columns.split(',') if col.strip() in self.data.columns]
        if len(columns) < 2:
            messagebox.showerror("Error", "Please enter at least one dependent and one independent variable.")
            return
        dependent_var, independent_vars = columns[0], columns[1:]
        X_train, X_test, y_train, y_test = self.get_train_test_split(independent_vars, dependent_var)
        
        model = MLPClassifier(max_iter=500)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        self.evaluate_model(model, X_train, X_test, y_train, y_test, predictions, "Neural Network")

    def documentation_metadata(self):
        license_note = {
            "FULL": "FULL license active; report and plot exports are enabled.",
            "TRIAL": "TRIAL license active; report and plot exports are locked until a FULL key is activated."
        }.get(self.license_type, "Custom license active.")
        data_summary = "No data loaded"
        if self.data is not None:
            data_summary = f"{self.data.shape[0]} rows x {self.data.shape[1]} columns"
        return [
            ("Title", APP_NAME),
            ("Version", APP_VERSION),
            ("Author", APP_AUTHOR),
            ("Firm", APP_COMPANY),
            ("Email", APP_EMAIL),
            ("Author Profile", APP_PROFILE),
            ("License Status", self.license_type),
            ("License Details", license_note),
            ("Current Data", data_summary),
            ("Generated On", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        ]

    def documentation_entries(self):
        entries = []

        def add(menu, technique, inputs, process, outputs, app_example, df_example):
            entries.append({
                "menu": menu,
                "technique": technique,
                "inputs": inputs,
                "process": process,
                "outputs": outputs,
                "app": app_example,
                "df": df_example
            })

        add("File", "Open CSV File", "A CSV file path chosen in the file dialog.", "Reads the CSV with pandas and renders it in the data grid.", "Loaded data grid and refreshed console df.", 'app.load_csv("data.csv")', 'df = pd.read_csv("data.csv"); app.set_df(df)')
        add("File", "Save Data", "Edited cells in the visible grid.", "Writes grid edits back to the in-memory DataFrame.", "Updated in-memory data.", "app.save_csv(app.get_df(), \"saved.csv\")", "app.save_data()")
        add("File", "Save Data File", "A target CSV file name.", "Exports the current DataFrame without the index.", "CSV file on disk.", 'app.save_csv("saved.csv")', 'df.to_csv("saved.csv", index=False)')
        add("File", "Clear Data", "Confirmation to clear the grid.", "Removes all rows and columns from the current data.", "Empty data grid.", "app.set_df(pd.DataFrame())", "df = df.iloc[0:0]")
        add("File", "View License", "License key string stored in license.json.", "Displays the license agreement and activation state.", "License window.", "print(app.license_type)", "license_status = app.license_type")
        add("File", "Exit", "Confirmation if data is present.", "Closes the application window.", "Application exit.", "root.destroy()", "df = None")

        add("Edit", "Rename Columns", "Existing column name and a new name.", "Renames the selected column and refreshes the grid.", "Updated column header.", 'app.set_df(app.get_df().rename(columns={"old": "new"}))', 'df = df.rename(columns={"old": "new"}); app.set_df(df)')
        add("Edit", "Rename Column...", "Currently active column and a replacement name.", "Renames one column through a dialog and refreshes the grid.", "Updated data grid and console df.", 'app.rename_column("old", "new")', 'df = df.rename(columns={"old": "new"})')
        add("Edit", "Edit Cell...", "Clicked row, column and replacement value.", "Coerces the value to the column type where possible and writes it to the DataFrame.", "Edited data grid and console df.", 'app.edit_cell(0, "age", 31)', 'df.loc[0, "age"] = 31; app.refresh_df()')
        add("Edit", "Edit Row...", "Selected row position and one value per column.", "Opens a dialog with editable entries for every value in the selected row and applies them to the DataFrame.", "Updated data grid and console df.", 'app.edit_cell(0, "age", 31)', 'df.loc[0] = [31, "East"]')
        add("Edit", "Edit Column...", "Selected column, optional new name and one value per row.", "Opens a dialog with the column values and applies edits after validating the row count.", "Updated data grid and console df.", 'app.rename_column("region", "Region")', 'df["Region"] = df["region"]')
        add("Edit > Insert", "Row Above", "Insertion position and row count.", "Inserts blank rows immediately above the selected grid row.", "Expanded data grid.", 'app.insert_row(2, count=1)', 'df = pd.concat([df.iloc[:2], pd.DataFrame([{}]), df.iloc[2:]])')
        add("Edit > Insert", "Row Below", "Insertion position and row count.", "Inserts blank rows immediately below the selected grid row.", "Expanded data grid.", 'app.insert_row(3, count=1)', 'df = pd.concat([df.iloc[:3], pd.DataFrame([{}]), df.iloc[3:]])')
        add("Edit > Insert", "Column Before", "Insertion position and new column name.", "Inserts a new blank column immediately before the selected column.", "Expanded data grid.", 'app.insert_column(1, "Region")', 'df.insert(1, "Region", np.nan)')
        add("Edit > Insert", "Column After", "Insertion position and new column name.", "Inserts a new blank column immediately after the selected column.", "Expanded data grid.", 'app.insert_column(2, "Region")', 'df.insert(2, "Region", np.nan)')
        add("Edit > Delete", "Selected Row", "Selected grid row or row position.", "Drops the row from the DataFrame and refreshes the grid.", "Updated data grid.", 'app.delete_row_at(2)', 'df = df.drop(df.index[2]).reset_index(drop=True)')
        add("Edit > Delete", "Selected Column", "Selected grid column or column name.", "Drops the column from the DataFrame and refreshes the grid.", "Updated data grid.", 'app.delete_column("Region")', 'df = df.drop(columns=["Region"])')
        add("Edit", "Copy Column Names", "None; uses the active DataFrame.", "Copies all column names as a comma-separated string.", "Clipboard text.", 'app.copy_column_names()', '", ".join(df.columns)')
        add("Edit", "Compute Variable", "Target column name and a pandas expression; the target may exist or be new.", "Evaluates the expression row-wise and assigns the result to the target column, overwriting it when it exists.", "Computed values in a new or existing column.", 'app.compute_column("Total", "Price * Qty")', 'df["Total"] = df["Price"] * df["Qty"]')
        add("Edit", "Evaluate Expression", "A pandas-style expression and optional DataFrame.", "Evaluates formulas with column names, common math functions, numpy expressions, and scalar reductions without changing the grid.", "Computed values suitable for preview or assignment.", 'app.evaluate_expression("log(Income)")', 'np.log(df["Income"])')
        add("Edit", "Recode Variable", "Source column and old-to-new value mapping.", "Replaces values using the supplied mapping.", "Recoded column.", 'app.recode_column("gender", {"M": "Male", "F": "Female"})', 'df["gender"] = df["gender"].replace({"M": "Male", "F": "Female"})')
        add("Transformations > Missing Data Treatment", "Basic Treatment...", "Column and treatment method such as mean, median, mode, or drop.", "Fills or removes missing values and refreshes the grid.", "Treated column.", 'app.treat_missing("age", "mean")', 'df["age"] = df["age"].fillna(df["age"].mean())')
        add("Transformations > Missing Data Treatment", "Advanced Imputation...", "Target column, predictor columns, and one of regression, random forest, KNN, or iterative imputation.", "Opens the advanced imputation dialog, fits the selected model on observed rows, and predicts missing target values.", "Imputed column and model-specific report.", 'app.treat_missing_advanced("income", "knn", ["age", "score"])', 'df["income"] = df["income"].fillna(df["income"].mean())')
        add("Transformations > Missing Data Treatment", "Regression Imputation", "Numeric target column and numeric predictor columns.", "Fits a linear regression on observed rows and predicts missing target values.", "Imputed target column and before/after report with a chart.", 'app.treat_missing_advanced("income", "regression", ["age", "score"])', 'df["income"] = df["income"].fillna(df["income"].mean())')
        add("Transformations > Missing Data Treatment", "Random Forest Imputation", "Target column and predictor columns; the target may be numeric or categorical.", "Fits a random forest model on observed rows and predicts missing values.", "Imputed target column and model-based report.", 'app.treat_missing_advanced("segment", "random_forest", ["age", "income"])', 'df["segment"] = df["segment"].fillna(df["segment"].mode().iloc[0])')
        add("Transformations > Missing Data Treatment", "KNN Imputation", "Numeric target and predictor columns.", "Uses KNNImputer to estimate missing values from the nearest complete cases.", "Imputed target column and before/after report.", 'app.treat_missing_advanced("income", "knn", ["age", "score"])', 'from sklearn.impute import KNNImputer; df[["income"]] = KNNImputer().fit_transform(df[["income"]])')
        add("Transformations > Missing Data Treatment", "Iterative Imputation", "Numeric target and predictor columns.", "Uses iterative multivariate imputation with scikit-learn IterativeImputer.", "Imputed target column and before/after report.", 'app.treat_missing_advanced("income", "iterative", ["age", "score"])', 'from sklearn.experimental import enable_iterative_imputer; from sklearn.impute import IterativeImputer')
        add("Transformations > Missing Data Treatment", "Simulate Missing Data...", "Columns, missing rate, mechanism, optional MAR predictor, and seed.", "Introduces MCAR, MAR, or MNAR missing values for demonstrations and tests.", "Updated grid with simulated missing cells and a missing-count report.", 'app.simulate_missing_data(["age", "income"], 0.2, "mar", "score", 42)', 'df.loc[df.sample(frac=0.2).index, "age"] = np.nan')
        add("Transformations > Missing Data Treatment", "Missingness Significance Test...", "Columns to test, optional group column, and alpha.", "Runs chi-square or observed-variable screening with multiple-comparison correction.", "Test statistics, p-values, adjusted thresholds, and interpretation.", 'app.test_missing_significance(["age", "income"], group_column="region")', 'pd.crosstab(df["region"], df["age"].isna())')
        add("Transformations > Missing Data Treatment", "Missingness Report", "The current data grid.", "Summarizes missing counts, percentages, and missingness patterns.", "Missingness summary report with a percentage chart.", "app.missingness_report()", "df.isna().sum()")
        add("Edit", "Set Values", "Row id, column, and replacement value.", "Updates the selected cell in the DataFrame.", "Edited data grid.", 'app.get_df().loc[2, "age"] = 30', 'df.loc[2, "age"] = 30; app.refresh_df()')

        add("Transformations", "Data Simulations", "Pre-filled rows, seed, and variable specifications in one dialog.", "Generates int, float, normal, or category variables from inferred defaults.", "Simulated columns appended to the grid.", "app.simulate({'age': ('int', 18, 65)}, rows=100, seed=42)", 'df["age"] = np.random.default_rng(42).integers(18, 65, 100)')
        add("Transformations", "Multivariate Normal Distribution", "Pre-filled names, means, covariance matrix, samples, and seed.", "Draws samples from a multivariate normal distribution.", "Simulated numeric columns appended to the grid.", "app.simulate_multivariate_normal(['x', 'y'], [0, 1], [[1, 0.5], [0.5, 1]], samples=100)", 'df[["x", "y"]] = pd.DataFrame(np.random.multivariate_normal([0, 1], [[1, .5], [.5, 1]], 100), columns=["x", "y"])')
        for label, method in (("Cholesky Decomposition", "cholesky"), ("QR Decomposition", "qr"), ("SVD Decomposition", "svd"), ("Eigen Decomposition", "eig")):
            add("Transformations > Data Decomposition", label, "Numeric columns from the grid.", f"Computes the {method.upper()} matrix factorization.", "Factor matrices rendered in a report.", f'app.decompose_columns(["x", "y"], "{method}")', 'U, S, Vt = np.linalg.svd(df[["x", "y"]])')
        for label, method in (("Min-Max Scaling", "Min-Max"), ("Z-Score Standardization", "Z-Score"), ("Decimal Scaling", "Decimal Scaling"), ("Log Transformation", "Log"), ("Log-Normal Transformation", "Log-Normal")):
            add("Transformations > Data Standardization", label, "Numeric variable to transform.", f"Applies {label.lower()} and stores the result as a new column.", "Standardized or transformed variable.", f'app.standardize_column("x", "{method}")', 'df["x_scaled"] = (df["x"] - df["x"].min()) / (df["x"].max() - df["x"].min())')

        add("Data Analytics > Descriptive Statistics", "Frequency Tables", "One or more categorical columns.", "Counts each distinct value and builds a grouped chart.", "Frequency tables and bar chart.", "app.frequencies(['gender', 'education'])", 'df["gender"].value_counts()')
        add("Data Analytics > Descriptive Statistics", "Summary Statistics", "Numeric columns.", "Computes count, mean, standard deviation, quartiles, min, and max.", "Descriptive statistics report and boxplot.", "app.summary_stats(['age', 'income'])", "df[['age', 'income']].describe()")
        add("Data Analytics > Descriptive Statistics", "Dataset Properties", "None; uses the active data grid, or an optional frame and row limit from the console.", "Reports row count, column count, dimensions, memory use, duplicate/missing cells, data types and a copyable row list.", "Properties, column summary, and selectable list of rows in the output window.", "app.dataset_properties(show=True)", "app.dataset_properties(frame=df, limit=50)")
        add("Data Analytics > Inferential Statistics", "T-Test", "Two numeric columns.", "Runs an independent two-sample t-test.", "Statistic, p-value, and boxplot.", 'app.ttest("score_a", "score_b")', 'stats.ttest_ind(df["score_a"], df["score_b"], nan_policy="omit")')
        add("Data Analytics > Inferential Statistics", "Chi-Square Test", "Two categorical columns.", "Tests independence using a contingency table.", "Chi-square statistic, p-value, degrees of freedom, chart.", 'app.chisquare("gender", "purchased")', 'stats.chi2_contingency(pd.crosstab(df["gender"], df["purchased"]))')
        add("Data Analytics > Inferential Statistics", "Normality Tests", "Numeric columns.", "Runs Shapiro-Wilk and Kolmogorov-Smirnov tests.", "Per-column test statistics and histograms.", "app.normality(['age', 'income'])", 'stats.shapiro(df["age"].dropna())')
        add("Data Analytics > Inferential Statistics", "ANOVA", "Dependent numeric variable and a categorical factor.", "Runs one-way analysis of variance.", "F-statistic, p-value, and group boxplot.", 'app.anova("score", "group")', 'stats.f_oneway(*[g["score"] for _, g in df.groupby("group")])')
        add("Data Analytics > Inferential Statistics", "MANOVA", "Multiple dependent variables and a categorical factor.", "Fits a multivariate analysis of variance model.", "MANOVA result table and scatter matrix.", 'app.manova(["score_a", "score_b"], "group")', 'MANOVA.from_formula("score_a + score_b ~ group", data=df).mv_test()')
        add("Data Analytics > Factor Analysis", "Exploratory Factor Analysis", "Numeric columns and number of factors.", "Extracts latent factors and computes Bartlett and KMO tests.", "Loadings, test results, scree and loading plots.", "app.efa(['q1', 'q2', 'q3'], n_factors=2)", 'FactorAnalyzer(n_factors=2).fit(df[["q1", "q2", "q3"]])')
        add("Data Analytics > Factor Analysis", "Confirmatory Factor Analysis", "semopy model description and columns.", "Fits the specified structural equation model.", "Model inspection table and SEM path diagram.", 'app.cfa("F1 =~ q1 + q2 + q3")', 'Model("F1 =~ q1 + q2 + q3").fit(df)')
        add("Data Analytics > Correlation & Regression", "Correlation Analysis", "Numeric columns or two variable sets.", "Computes Pearson correlation or canonical correlation.", "Coefficient table, p-values, heatmap or scatter plot.", "app.correlation(['x', 'y', 'z'])", 'df[["x", "y", "z"]].corr()')
        add("Data Analytics > Correlation & Regression", "Regression Analysis", "Dependent variable, independent variables, and model kind.", "Fits OLS or GLM regression with statsmodels.", "Coefficient table and fitted-values plot.", 'app.regression("sales", ["price", "ads"], kind="ols")', 'sm.OLS(df["sales"], sm.add_constant(df[["price", "ads"]])).fit()')
        add("Data Analytics > Cluster Analysis", "K-Means Clustering", "Numeric columns and cluster count.", "Standardized k-means fit with cluster labels written to the grid.", "Centers, cluster counts, and scatter plot.", "app.kmeans(['x', 'y'], n_clusters=3, show=True)", 'KMeans(n_clusters=3, n_init=10).fit_predict(df[["x", "y"]])')
        add("Data Analytics > Cluster Analysis", "Hierarchical Clustering", "Numeric columns.", "Builds a Ward linkage tree and cuts it into clusters.", "Linkage matrix, cluster counts, dendrogram.", "app.hierarchical(['x', 'y'], show=True)", "linkage(df[['x', 'y']], method='ward')")
        add("Data Analytics > Time Series Analysis", "Stationarity Tests", "A time series column.", "Runs ADF and KPSS tests.", "Test statistics, p-values, and series plot.", 'app.stationarity("sales")', 'adfuller(df["sales"].dropna())')
        add("Data Analytics > Time Series Analysis", "Seasonal Decomposition", "A time series column and period.", "Decomposes the series into trend, seasonal, and residual parts.", "Component table and four-panel plot.", 'app.seasonal_decomposition("sales", period=12)', 'seasonal_decompose(df["sales"].dropna(), period=12)')
        add("Data Analytics > Time Series Analysis", "Holt-Winters Method", "A time series column and seasonal periods.", "Fits exponential smoothing with trend and seasonality.", "Fitness measures and actual-versus-forecast plot.", 'app.holt_winters("sales", seasonal_periods=12)', 'ExponentialSmoothing(df["sales"], seasonal_periods=12).fit()')
        add("Data Analytics > Time Series Analysis", "Moving Averages", "A time series column and window size.", "Computes rolling mean and error metrics.", "Moving-average table, metrics, and chart.", 'app.moving_average("sales", window=3)[0]', 'df["sales"].rolling(3).mean()')
        add("Data Analytics > Time Series Analysis > Data Simulations", "Simulate Stationary Series", "Time variable, data variable, rows, start, frequency, base, noise, and seed; prefilled from current data.", "Generates a flat series with noise and no trend or seasonality.", "New time-series grid suited to ADF and KPSS stationarity tests.", 'app.simulate_time_series(preset="stationary", time_column="Date", value_column="Sales", rows=96)', 'df = pd.DataFrame({"Date": pd.date_range("2020-01-01", periods=96, freq="ME"), "Sales": np.random.normal(100, 2, 96)})')
        add("Data Analytics > Time Series Analysis > Data Simulations", "Simulate Non-Stationary Series", "Prefilled time and data options with trend slope and noise.", "Generates a series with a linear trend and no seasonality.", "New grid for demonstrating non-stationary behavior.", 'app.simulate_time_series(preset="non_stationary", trend=2, rows=96)', 'df["Sales"] = 100 + 2 * np.arange(96) + np.random.normal(0, 3, 96)')
        add("Data Analytics > Time Series Analysis > Data Simulations", "Simulate Seasonal Series", "Prefilled time and data options with seasonal model, amplitude, and period.", "Generates additive or multiplicative seasonality with optional trend.", "New grid for seasonal decomposition demonstrations.", 'app.simulate_time_series(preset="seasonal", seasonality=15, period=12, rows=96)', 'df["Sales"] = 100 + 15 * np.sin(2 * np.pi * np.arange(96) / 12)')
        add("Data Analytics > Time Series Analysis > Data Simulations", "Simulate Trend + Seasonal Series", "Prefilled trend type, trend slope, seasonal amplitude, period, and noise.", "Generates trend-plus-seasonality data suited to Holt-Winters fitting.", "New grid with a complete actual-versus-forecast demo dataset.", 'app.simulate_time_series(preset="holt_winters", trend=1.5, seasonality=20, period=12)', 'df["Sales"] = 100 + 1.5 * np.arange(96) + 20 * np.sin(2 * np.pi * np.arange(96) / 12)')
        add("Data Analytics > Time Series Analysis > Data Simulations", "Simulate Moving-Average Series", "Prefilled smoothing-friendly trend, noise, and window context.", "Generates a noisy but smooth trend series for moving-average analysis.", "New grid suitable for rolling-mean demonstrations.", 'app.simulate_time_series(preset="moving_average", noise=4, rows=96)', 'df["Sales"] = df["Sales"].rolling(3).mean()')
        add("Data Analytics > Time Series Analysis > Data Simulations", "Simulate Random Walk Series", "Prefilled base, drift, noise, and seed.", "Generates an integrated random-walk series.", "New grid for demonstrating strong non-stationarity.", 'app.simulate_time_series(preset="random_walk", random_walk=True)', 'df["Sales"] = 100 + np.cumsum(np.random.normal(0, 2, 96))')
        add("Data Analytics > Time Series Analysis > Data Simulations", "Simulate Custom Series", "All time-variable and data-variable options in one auto-populated dialog.", "Generates custom date, numeric-index, or period-label series with selected trend, seasonality, and noise.", "New grid and simulation report.", 'app.simulate_time_series(time_mode="numeric", trend_type="quadratic", seasonal_model="multiplicative", noise_type="uniform")', 'df = app.simulate_time_series(apply=False)')

        for label, method in (("Logistic Regression", "logistic"), ("Decision Tree", "decision_tree"), ("Random Forest", "random_forest"), ("Naive Bayes", "naive_bayes"), ("K-Nearest Neighbors", "knn"), ("Neural Network", "neural_network"), ("Support Vector Machine", "svm")):
            df_code = f'df = app.generate_ml_demo("{method}", apply=False); df.head()'
            add("Machine Learning > Supervised Learning", label, "Target column and feature columns, plus a train ratio.", "Splits data, fits the classifier, predicts the test set, and builds a full report.", "Test-set predictions, margin-labeled confusion matrix, metrics, technique details, and technique-specific plots.", f'app.classify("{method}", "Target", ["Feature_1", "Feature_2"], show=True)', df_code)
        for label, method in (("K-Means Clustering", "kmeans"), ("Hierarchical Clustering", "hierarchical"), ("DBSCAN Clustering", "dbscan"), ("Gaussian Mixture", "gaussian_mixture"), ("Principal Component Analysis", "pca")):
            add("Machine Learning > Unsupervised Learning", label, "Numeric feature columns and algorithm parameters.", "Fits the unsupervised model and attaches labels or components when requested.", "Cluster or component tables, counts, scores, and algorithm-specific plots.", f'app.{method}(["Feature_1", "Feature_2"], show=True)', 'features = df.select_dtypes(include=[np.number])')
        for label, key in self._ml_simulation_options():
            add("Machine Learning > Simulation Data", label, "None; the button uses inferred defaults.", "Generates a technique-appropriate synthetic dataset with a fixed seed.", "A new data grid ready for the matching technique.", f'app.generate_ml_demo("{key}", apply=True)', f'df = app.generate_ml_demo("{key}", apply=False)')

        add("Marketing > Generate Demo Data > Transactional Data", "Basket Matrix", "Recommended for frequent itemsets, association rules, market basket, and recommendations; optional replacement confirmation.", "Creates a wide transaction basket matrix.", "Rows as baskets and columns as items.", 'app.generate_basket(transactions=200, item_count=10)', "df = pd.read_csv('basket.csv')")
        add("Marketing > Generate Demo Data > Transactional Data", "Customer Transaction Log", "Recommended for RFM and item-list transaction mining; optional replacement confirmation.", "Creates long-format customer basket records with customer, date, amount, and item-list fields.", "Customer id, transaction id, date, spend, and item columns.", "app.generate_long_transactions(customers=40, transactions_per_customer=4)", "df.groupby('CustomerID')['Amount'].agg(['count', 'sum'])")
        add("Marketing > Generate Demo Data > Transactional Data", "Sequence Log", "Recommended for sequential purchase patterns; optional replacement confirmation.", "Creates customer purchase sequences.", "Customer id and ordered item-sequence columns.", "app.generate_sequence_log(customers=40, sequence_length=6)", "df['Sequence'].str.split(' > ')")
        add("Marketing > Generate Demo Data > Transactional Data", "Profit / Utility Basket", "Recommended for profit-weighted rules; optional replacement confirmation.", "Creates baskets with item profit weights.", "Basket matrix plus profit columns.", "app.generate_profit_basket(transactions=200, item_count=10)", "df.filter(like='Profit').mean()")
        add("Marketing > Generate Demo Data > Customer & Campaign Data", "Marketing Campaign Log", "Recommended for CLV, churn, response uplift, and elasticity; optional replacement confirmation plus customer, purchase, and seed settings.", "Generates customer campaigns with channel, response, churn, price, quantity, and amount fields.", "A marketing-ready grid for CLV, churn, uplift, and elasticity analyses.", 'app.generate_marketing_log(customers=200, transactions_per_customer=3, apply=True)', 'df = app.generate_marketing_log(customers=200, transactions_per_customer=3, apply=False)')
        add("Marketing > Generate Demo Data", "Clear Data", "Confirmation to clear the grid.", "Removes all transaction data.", "Empty data grid.", "app.set_df(pd.DataFrame())", "df = df.iloc[0:0]")
        add("Marketing > Transactional Analysis", "Transaction Matrix", "Item columns or an item-list column.", "Converts transactions into a boolean item matrix for marketing pattern mining.", "Boolean matrix with transactions as rows and items as columns.", 'app.transaction_matrix(["Items"])', "matrix = df['Items'].str.get_dummies(', ')")
        for label, algorithm in (("Apriori", "Apriori"), ("FP-Growth", "FP-Growth"), ("Eclat", "Eclat")):
            df_code = 'basket = df.drop(columns=["CustomerID"], errors="ignore").astype(bool)'
            add("Marketing > Transactional Analysis > Frequent Itemsets", label, "Basket columns and a minimum support value.", f"Runs {algorithm} to discover frequent itemsets.", "Frequent itemsets with support and chart.", f'app.frequent_itemsets(min_support=0.1, algorithm="{algorithm}")', df_code)
        add("Marketing > Transactional Analysis", "Closed Itemsets", "Basket columns and minimum support.", "Finds closed frequent itemsets locally.", "Closed itemsets with support.", "app.closed_itemsets(min_support=0.1)", "df.drop(columns=['CustomerID'], errors='ignore').astype(bool)")
        add("Marketing > Transactional Analysis", "Maximal Itemsets", "Basket columns and minimum support.", "Finds maximal frequent itemsets locally.", "Maximal itemsets with support.", "app.maximal_itemsets(min_support=0.1)", "df.drop(columns=['CustomerID'], errors='ignore').astype(bool)")
        add("Marketing > Transactional Analysis", "Association Rules", "Basket columns, support, confidence, and lift thresholds.", "Mines frequent itemsets and derives antecedent-to-consequent rules.", "Rules with support, confidence, lift, and a rule chart.", 'app.association_rules(min_support=0.1, min_confidence=0.6)', "rules = df.drop(columns=['CustomerID']).astype(bool)")
        add("Marketing > Transactional Analysis", "Market Basket Report", "Minimum support and confidence.", "Combines frequent itemsets and association rules.", "Itemset table, rules table, and rule plot.", "app.market_basket(min_support=0.1, min_confidence=0.5)", "df.drop(columns=['CustomerID'], errors='ignore').astype(bool)")
        add("Marketing > Transactional Analysis", "Sequential Purchase Patterns", "Customer id and sequence column from the Sequence Log simulation.", "Runs a prefix-span style sequence miner.", "Sequential patterns with support and bar chart.", "app.sequential_patterns(min_support=0.2)", "df['Sequence'].str.split(' > ')")
        add("Marketing > Customer Analytics", "RFM Segmentation", "Customer id, date, and spend columns from the Customer Transaction Log or Marketing Campaign Log.", "Computes recency, frequency, monetary scores and segments.", "RFM scores, segment summary, and segment chart.", "app.rfm()", "df.groupby('CustomerID')['Amount'].agg(['count', 'sum'])")
        add("Marketing > Customer Analytics", "Customer Lifetime Value (CLV)", "Customer id, date, amount, retention rate, and forecast horizon from the Marketing Campaign Log.", "Estimates order value and purchase frequency from transaction history, then projects value over the horizon.", "Per-customer CLV table, summary metrics, and top-customer chart.", 'app.customer_lifetime_value(retention=0.95, horizon=12)', 'df.groupby("CustomerID")["Amount"].agg(["count", "sum"])')
        add("Marketing > Customer Analytics", "Churn Prediction", "Customer id, binary churn target, numeric feature columns, and test-set share from the Marketing Campaign Log.", "Fits logistic regression, scores held-out customers, and reports accuracy, ROC AUC, and coefficients.", "Predictions, churn drivers, summary metrics, and coefficient chart.", "app.churn_prediction(features=['Age', 'TenureMonths'])", 'from sklearn.linear_model import LogisticRegression; LogisticRegression().fit(df[["Age"]], df["Churn"])')
        add("Marketing > Campaign & Pricing Analytics", "Campaign Response Uplift", "Binary response target, campaign segment column, and optional value column from the Marketing Campaign Log.", "Compares segment response rates with the overall rate and ranks segments by uplift.", "Segment uplift table, summary, and response-rate chart.", "app.campaign_response(segment_column='CampaignChannel')", 'df.groupby("CampaignChannel")["Response"].mean()')
        add("Marketing > Campaign & Pricing Analytics", "Price Elasticity Analysis", "Positive price and quantity columns from the Marketing Campaign Log.", "Fits a log-log demand curve and projects revenue under a modest price adjustment.", "Elasticity, fit quality, revenue projection, and log-demand chart.", "app.price_elasticity(price_column='Price', quantity_column='Quantity')", 'np.polyfit(np.log(df["Price"]), np.log(df["Quantity"]), 1)')
        add("Marketing > Campaign & Pricing Analytics", "Profit-Weighted Rules", "Basket items, profit columns, support, and confidence from the Profit / Utility Basket simulation.", "Generates rules ranked by utility-weighted lift.", "Profit-weighted rule report.", "app.profit_weighted_rules(min_support=0.05, min_confidence=0.3)", "df.filter(like='Profit').mean()")
        add("Marketing > Campaign & Pricing Analytics", "Recommendation / Affinity Report", "Target item and rule thresholds from the Basket Matrix or Customer Transaction Log.", "Finds items that frequently co-occur with the target.", "Recommendation table and affinity plot.", "app.recommendations(min_support=0.05, min_confidence=0.3)", "df.corr().sort_values('Item_A', ascending=False)")

        domain_documents = {
            "credit_risk": ("Encodes features, splits into train/test, fits logistic regression or gradient boosting, and computes classification metrics.", "Risk scores, AUC, confusion matrix, feature effects and ROC plot."),
            "fraud_detection": ("Standardizes numeric features and applies Isolation Forest, Local Outlier Factor or One-Class SVM.", "Anomaly scores, flagged transactions, optional labelled performance and diagnostic plots."),
            "portfolio_risk": ("Computes weighted portfolio returns, VaR/CVaR, drawdowns, correlations and risk contributions.", "Risk summary, correlation matrix, drawdown curve and return-distribution plot."),
            "fair_lending": ("Compares approval rates across protected groups and fits an adjusted OLS model with control features.", "Selection rates, adverse impact ratio, confidence intervals and group outcome plots."),
            "demand_forecast": ("Backtests Holt-Winters or ARIMA on a numeric series, then forecasts future periods with interval bounds.", "Forecast table, backtest metrics, actual-versus-forecast chart and residual diagnostics."),
            "attrition": ("Fits a classifier and converts predicted probabilities into Low, Medium and High attrition risk bands.", "Attrition risk bands, classification metrics, feature effects and probability plots."),
            "pay_equity": ("Fits OLS with group, role and performance controls and compares adjusted group coefficients.", "Unadjusted and adjusted pay gaps, confidence intervals and diagnostic plots."),
            "employee_segmentation": ("Standardizes numeric features and applies k-means with silhouette and elbow diagnostics.", "Cluster counts, profiles, assignments, PCA projection and elbow curve."),
            "workforce_forecast": ("Backtests a seasonal trend model on headcount history and projects future staffing demand.", "Headcount forecast, interval bounds, backtest metrics and residual charts."),
            "predictive_maintenance": ("Scores sensor telemetry with anomaly detection and optionally evaluates labelled failures.", "Failure-risk scores, flagged assets, optional labelled precision and diagnostic plots."),
            "incident_priority": ("Fits a classifier to ticket features and predicts P1/P2/P3 priority classes.", "Priority confusion matrix, classification metrics, feature effects and probability plots."),
            "sla_breach": ("Fits a classifier to ticket age, affected users and category codes to predict SLA breaches.", "Breach confusion matrix, precision/recall, feature effects and ROC plot."),
            "capacity_forecast": ("Backtests a seasonal forecast on capacity demand and projects future workload.", "Capacity forecast, interval bounds, backtest errors and trend chart."),
            "inventory_reorder": ("Combines demand variability and lead time with a service-level target, then runs Monte Carlo stockout scenarios.", "Cycle stock, safety stock, reorder point, stockout probability and sensitivity charts."),
            "supplier_risk": ("Clusters supplier risk features and builds a country-supplier dependency graph with network centrality.", "Risk profiles, cluster assignments, degree/betweenness centrality and network diagram."),
            "process_cycle": ("Reconstructs case sequences from an event log and measures dwell times and cycle times.", "Top transitions, cycle-time distribution, activity dwell times and transition chart."),
            "country_segmentation": ("Standardizes macro indicators and clusters countries or markets with k-means diagnostics.", "Market clusters, country profiles, PCA projection and silhouette/elbow plots."),
            "export_forecast": ("Filters an optional country or market, backtests a seasonal model, and forecasts export value.", "Filtered export forecast, confidence bounds, backtest metrics and residual charts."),
            "fx_risk": ("Computes FX portfolio VaR/CVaR and overlays user-defined currency stress shocks.", "Risk metrics, correlation matrix, drawdowns, return distribution and stress-scenario chart."),
            "trade_anomaly": ("Scores trade flows with anomaly detection and optionally evaluates labelled irregular transactions.", "Flagged flows, anomaly scores, optional labelled performance and diagnostic plots."),
            "gravity_model": ("Regresses log trade flow on log exporter GDP, log importer GDP, distance and an optional currency dummy.", "Elasticity coefficients, confidence intervals, fitted-versus-actual and residual plots."),
            "tariff_impact": ("Simulates price elasticity draws under a tariff and recomputes demand and revenue across scenarios.", "Expected demand/revenue changes, product impact table and Monte Carlo distributions."),
        }
        for domain_key, menu_name, analysis_menu in (
            ("finance", "Finance", "Business Analytics"),
            ("hrm", "HRM", "People Analytics"),
            ("it_operations", "IT & Operations", "Operations Analytics"),
            ("international", "International Business", "International Analytics"),
        ):
            demo_api = f"{domain_key}_demo"
            first_preset = domain_analytics.DOMAIN_SIMULATIONS[domain_key][0][1]
            for label, preset in domain_analytics.DOMAIN_SIMULATIONS[domain_key]:
                add(f"{menu_name} > Generate Demo Data", label, "Optional confirmation to replace the current grid.", f"Generates a seeded {label.lower()} dataset with business-realistic columns.", "New data grid ready for domain analytics.", f'app.{demo_api}("{preset}", apply=True)', f'df = app.{demo_api}("{preset}", apply=False)')
            for label, technique in domain_analytics.DOMAIN_ANALYSES[domain_key]:
                process, outputs = domain_documents[technique]
                add(f"{menu_name} > {analysis_menu}", label, "Column roles and parameters prefilled from the current data grid.", process, outputs, f'app.{technique}(show=True)', f'df = app.{demo_api}("{first_preset}", apply=False); app.{technique}(frame=df, show=True)')

        add("Plots", "Plot Builder", "Plot type, variable roles, and styling options.", "Opens one dialog with variable dropdowns, options, and a live report-style output.", "Selected plot in a downloadable report window.", "app.visualize('scatter', x='sales', y='profit', hue='region')", "df.plot.scatter(x='sales', y='profit')")
        add("Plots", "Quick Plot (Auto)", "One, two, or three columns.", "Infers histogram, bar, box, or scatter from column types.", "Automatically chosen plot.", "app.plot(['sales', 'profit'])", "df[['sales', 'profit']].plot()")
        for label, kind in (("Histogram", "histogram"), ("KDE / Density", "kde"), ("Box Plot", "box"), ("Violin Plot", "violin"), ("Strip Plot", "strip")):
            df_code = f'df["x"].plot(kind="hist")' if kind == "histogram" else "sns.displot(df['x'], kde=True)"
            add("Plots > Distribution", label, "A numeric value column and optional grouping column.", f"Builds the {label.lower()} with configurable bins, grouping, and palette.", "Distribution plot in a report window.", f'app.visualize("{kind}", x="x")', df_code)
        for label, kind in (("Bar / Count Chart", "bar"), ("Horizontal Bar", "horizontal_bar"), ("Pie Chart", "pie"), ("Grouped Bar", "grouped_bar"), ("Stacked Bar", "stacked_bar")):
            app_example = f'app.visualize("{kind}", x="category", y="value")'
            if kind in ("grouped_bar", "stacked_bar"):
                app_example += ', hue="group"'
            add("Plots > Categorical", label, "Category column, optional value column, and optional grouping column.", f"Aggregates values and draws a {label.lower()}.", "Categorical comparison plot.", app_example, 'df.groupby("category")["value"].sum().plot(kind="bar")')
        for label, kind in (("Scatter Plot", "scatter"), ("Bubble Chart", "bubble"), ("Line Plot", "line"), ("Area Plot", "area"), ("Correlation Heatmap", "heatmap"), ("Pair Plot", "pairplot")):
            app_example = f'app.visualize("{kind}", columns=["x", "y"])' if kind in ("heatmap", "pairplot") else f'app.visualize("{kind}", x="x", y="y")'
            add("Plots > Relationships", label, "Numeric columns and optional size or group columns.", f"Draws the {label.lower()} to inspect relationships.", "Relationship plot in a report window.", app_example, 'df.plot.scatter(x="x", y="y")')
        for label, kind in (("Pareto Chart", "pareto"), ("Time Series Line", "time_series"), ("Moving Average", "moving_average")):
            add("Plots > Business Analytics", label, "Category or time column plus a value column.", f"Builds a {label.lower()} with cumulative or rolling behavior.", "Business analytics plot.", f'app.visualize("{kind}", x="period", y="value")', 'df["value"].rolling(3).mean().plot()')
        add("Plots", "Plot Help", "None.", "Shows quick instructions and console examples.", "Help dialog.", "app.show_plot_help()", "df.head()")

        add("Python Console", "Open Console", "None.", "Opens a persistent Python shell with app, df, pandas, numpy, matplotlib, seaborn, scipy, and sklearn loaded.", "Interactive console window.", "app.open_console()", "df.head()")
        add("Python Console", "df Object", "The active DataFrame.", "Represents the current data grid; assigning a new DataFrame syncs it back to the app after execution.", "DataFrame-backed grid data.", "app.get_df()", "df.describe()")
        add("Python Console", "app Object", "The running SADSA application instance.", "Exposes all data, analytics, simulation, reporting, and plotting methods.", "Programmatic control of the app.", "app.help_api()", "app.set_df(df)")
        add("Python Console", "New Script", "None.", "Creates a new empty script tab in the built-in editor.", "Empty script editor tab.", "app.open_console().new_script()", "script = ''")
        add("Python Console", "Open Script...", "A Python script path chosen in the file dialog.", "Opens an existing script in a new editor tab.", "Script editor tab with loaded source.", "app.open_console().open_script()", "script = Path('analysis.py').read_text()")
        add("Python Console", "Save Script", "Script text and its current path.", "Writes the active editor contents to its existing .py file.", "Saved script file.", "app.open_console().save_script()", "Path('analysis.py').write_text(script)")
        add("Python Console", "Save Script As...", "Script text and a target path.", "Writes the editor contents to a chosen .py file.", "Saved script file at the chosen path.", "app.open_console().save_script_as()", "Path('analysis.py').write_text('print(1)')")
        add("Python Console", "Run Script", "Script text or selected lines.", "Executes code with stdout and stderr captured in the console.", "Console output and updated app state.", "app.open_console().run_script()", "exec('print(df.shape)')")

        add("Help > Documentation", "Open Documentation", "None.", "Opens this documentation page with searchable sections for every menu, method, and technique.", "Readable documentation with app/author/company/license heading.", "app.open_documentation()", "print(app.documentation_text())")
        add("Help > Documentation", "Download TXT", "None.", "Exports the complete documentation guide as a plain-text file.", "Downloadable .txt documentation file.", 'app.download_documentation("txt")', "text = app.documentation_text()")
        add("Help > Documentation", "Download Markdown", "None.", "Exports the complete documentation guide as a Markdown file.", "Downloadable .md documentation file.", 'app.download_documentation("md")', "markdown = app.documentation_text()")
        add("Help > Documentation", "Download DOCX", "None.", "Exports the complete documentation guide as a Word document.", "Downloadable .docx documentation file.", 'app.download_documentation("docx")', "document = app.documentation_metadata()")
        add("Help > Documentation", "Download PDF", "None.", "Exports the complete documentation guide as a PDF document.", "Downloadable .pdf documentation file.", 'app.download_documentation("pdf")', "pdf_ready = bool(app.documentation_sections())")
        add("Help", "Contact", "None.", "Shows company contact information.", "Contact dialog.", 'app.show_message("Contact", "Amchik Solutions, India")', "company = 'Amchik Solutions, India'")
        add("Help", "Author", "None.", "Shows author name, email, and profile link.", "Author dialog.", 'app.show_message("Author", "Dr. Kamakshaiah Musunuru")', "author = 'Dr. Kamakshaiah Musunuru'")
        add("Help", "Version", "None.", "Shows the application version.", f"Version dialog: {APP_VERSION}.", f'app.show_message("Version", "{APP_VERSION}")', f'version = "{APP_VERSION}"')
        add("Help", "About", "None.", "Shows a short description of SADSA.", "About dialog.", 'app.show_message("About", "SADSA by Amchik Solutions, India")', "about = 'SADSA'")
        return entries

    def documentation_sections(self):
        grouped = {}
        for entry in self.documentation_entries():
            parts = [part.strip() for part in entry["menu"].split(">") if part.strip()]
            menu = parts[0] if parts else "Main Menu"
            subsection = " > ".join(parts[1:]) if len(parts) > 1 else "Main Menu"
            grouped.setdefault(menu, {}).setdefault(subsection, []).append(entry)
        sections = []
        for menu, subsections in grouped.items():
            section_subsections = []
            for name, entries in subsections.items():
                section_subsections.append({"name": name, "entries": entries})
            sections.append({"menu": menu, "subsections": section_subsections})
        return sections

    def documentation_text(self):
        lines = [APP_NAME, APP_VERSION, "DOCUMENTATION GUIDE", "", "TITLE SECTION", ""]
        for key, value in self.documentation_metadata():
            lines.append(f"{key}: {value}")
        lines.append("")
        lines.append("CONTENTS")
        sections = self.documentation_sections()
        for section_index, section in enumerate(sections, 1):
            lines.append(f"{section_index}. {section['menu']}")
            for subsection_index, subsection in enumerate(section["subsections"], 1):
                lines.append(f"  {section_index}.{subsection_index}. {subsection['name']}")
        lines.append("")
        lines.append("TECHNIQUE DOCUMENTATION")
        for section_index, section in enumerate(sections, 1):
            lines.append(f"{section_index}. {section['menu']}")
            for subsection_index, subsection in enumerate(section["subsections"], 1):
                lines.append(f"  {section_index}.{subsection_index}. {subsection['name']}")
                for technique_index, entry in enumerate(subsection["entries"], 1):
                    lines.append(f"    {section_index}.{subsection_index}.{technique_index}. {entry['technique']}")
                    lines.append(f"      Menu path: {entry['menu']} > {entry['technique']}")
                    lines.append(f"      Inputs: {entry['inputs']}")
                    lines.append(f"      Process: {entry['process']}")
                    lines.append(f"      Expected outputs: {entry['outputs']}")
                    lines.append(f"      App console: {entry['app']}")
                    lines.append(f"      df console: {entry['df']}")
                    lines.append("")
        return "\n".join(lines)

    def documentation_markdown(self):
        lines = [f"# {APP_NAME}", "", f"## {APP_VERSION}", "", "> Documentation Guide", ""]
        lines.append("## Title Section")
        lines.append("")
        lines.append("| Field | Detail |")
        lines.append("| --- | --- |")
        for key, value in self.documentation_metadata():
            lines.append(f"| {key} | {value} |")
        lines.append("")
        sections = self.documentation_sections()
        lines.append("## Contents")
        lines.append("")
        for section_index, section in enumerate(sections, 1):
            lines.append(f"{section_index}. {section['menu']}")
            for subsection_index, subsection in enumerate(section["subsections"], 1):
                lines.append(f"  {section_index}.{subsection_index}. {subsection['name']}")
        lines.append("")
        lines.append("## Technique Documentation")
        lines.append("")
        lines.append("Each technique lists inputs, processing steps, expected outputs, and console examples for both `app` and `df`.")
        lines.append("")
        for section_index, section in enumerate(sections, 1):
            lines.append(f"## {section_index}. {section['menu']}")
            lines.append("")
            for subsection_index, subsection in enumerate(section["subsections"], 1):
                lines.append(f"### {section_index}.{subsection_index}. {subsection['name']}")
                lines.append("")
                for technique_index, entry in enumerate(subsection["entries"], 1):
                    lines.append(f"#### {section_index}.{subsection_index}.{technique_index}. {entry['technique']}")
                    lines.append("")
                    lines.append(f"**Menu path:** {entry['menu']} > {entry['technique']}")
                    lines.append("")
                    lines.append(f"- **Inputs:** {entry['inputs']}")
                    lines.append(f"- **Process:** {entry['process']}")
                    lines.append(f"- **Expected outputs:** {entry['outputs']}")
                    lines.append("")
                    lines.append("**App console**")
                    lines.append("")
                    lines.append("```python")
                    lines.append(entry["app"])
                    lines.append("```")
                    lines.append("")
                    lines.append("**df console**")
                    lines.append("")
                    lines.append("```python")
                    lines.append(entry["df"])
                    lines.append("```")
                    lines.append("")
        return "\n".join(lines)

    def _docx_set_cell_shading(self, cell, fill):
        cell_properties = cell._tc.get_or_add_tcPr()
        shading = cell_properties.find(qn("w:shd"))
        if shading is None:
            shading = OxmlElement("w:shd")
            cell_properties.append(shading)
        shading.set(qn("w:fill"), fill)

    def _docx_set_cell_text(self, cell, value, bold=False, color=None, size=10):
        cell.text = ""
        paragraph = cell.paragraphs[0]
        run = paragraph.add_run(str(value))
        run.bold = bold
        run.font.size = Pt(size)
        if color is not None:
            run.font.color.rgb = RGBColor(*color)
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER

    def _docx_add_code_block(self, document, label, code):
        heading = document.add_paragraph()
        heading.paragraph_format.space_before = Pt(6)
        heading.paragraph_format.space_after = Pt(2)
        heading_run = heading.add_run(label)
        heading_run.bold = True
        heading_run.font.color.rgb = RGBColor(31, 78, 121)
        paragraph = document.add_paragraph()
        paragraph.paragraph_format.left_indent = Inches(0.2)
        paragraph.paragraph_format.space_after = Pt(6)
        run = paragraph.add_run(code)
        run.font.name = "Consolas"
        run.font.size = Pt(9)
        run.font.color.rgb = RGBColor(45, 45, 45)
        shading = OxmlElement("w:shd")
        shading.set(qn("w:fill"), "F2F2F2")
        paragraph._p.get_or_add_pPr().append(shading)

    def _docx_add_page_field(self, paragraph):
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = paragraph.add_run()
        begin = OxmlElement("w:fldChar")
        begin.set(qn("w:fldCharType"), "begin")
        instruction = OxmlElement("w:instrText")
        instruction.set(qn("xml:space"), "preserve")
        instruction.text = " PAGE "
        separate = OxmlElement("w:fldChar")
        separate.set(qn("w:fldCharType"), "separate")
        placeholder = OxmlElement("w:t")
        placeholder.text = "1"
        end = OxmlElement("w:fldChar")
        end.set(qn("w:fldCharType"), "end")
        for element in (begin, instruction, separate, placeholder, end):
            run._r.append(element)

    def documentation_docx(self):
        document = Document()
        document.core_properties.title = APP_NAME
        document.core_properties.subject = f"Documentation for {APP_VERSION}"
        document.core_properties.author = APP_AUTHOR
        document.core_properties.company = APP_COMPANY
        document.core_properties.comments = f"Contact: {APP_EMAIL} | Profile: {APP_PROFILE}"
        section = document.sections[0]
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.85)
        section.right_margin = Inches(0.85)
        styles = document.styles
        styles["Normal"].font.name = "Calibri"
        styles["Normal"].font.size = Pt(10)
        for style_name, size in (("Title", 24), ("Heading 1", 16), ("Heading 2", 13), ("Heading 3", 11), ("Heading 4", 10)):
            style = styles[style_name]
            style.font.name = "Calibri"
            style.font.size = Pt(size)
            style.font.bold = True
            style.font.color.rgb = RGBColor(31, 78, 121)
            style.paragraph_format.keep_with_next = True
        header = section.header.paragraphs[0]
        header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        header_run = header.add_run(f"{APP_NAME} | {APP_VERSION}")
        header_run.font.size = Pt(8)
        header_run.font.color.rgb = RGBColor(100, 100, 100)
        footer = section.footer.paragraphs[0]
        footer_prefix = footer.add_run(f"{APP_SHORT_NAME} Documentation | {APP_VERSION} | Page ")
        footer_prefix.font.size = Pt(8)
        footer_prefix.font.color.rgb = RGBColor(100, 100, 100)
        self._docx_add_page_field(footer)
        banner = document.add_table(rows=1, cols=1)
        banner.alignment = WD_TABLE_ALIGNMENT.CENTER
        banner_cell = banner.cell(0, 0)
        self._docx_set_cell_shading(banner_cell, "1F4E79")
        self._docx_set_cell_text(banner_cell, APP_NAME, bold=True, color=(255, 255, 255), size=20)
        banner_cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        subtitle = document.add_paragraph()
        subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
        subtitle_run = subtitle.add_run(APP_VERSION)
        subtitle_run.bold = True
        subtitle_run.font.size = Pt(14)
        subtitle_run.font.color.rgb = RGBColor(31, 78, 121)
        guide = document.add_paragraph()
        guide.alignment = WD_ALIGN_PARAGRAPH.CENTER
        guide_run = guide.add_run("Documentation Guide")
        guide_run.font.size = Pt(12)
        guide_run.font.color.rgb = RGBColor(80, 80, 80)
        document.add_heading("Title Section", level=1)
        metadata_table = document.add_table(rows=1, cols=2)
        metadata_table.style = "Table Grid"
        metadata_table.alignment = WD_TABLE_ALIGNMENT.CENTER
        self._docx_set_cell_text(metadata_table.cell(0, 0), "Field", bold=True, color=(255, 255, 255))
        self._docx_set_cell_text(metadata_table.cell(0, 1), "Detail", bold=True, color=(255, 255, 255))
        for cell in metadata_table.rows[0].cells:
            self._docx_set_cell_shading(cell, "1F4E79")
        for key, value in self.documentation_metadata():
            row_cells = metadata_table.add_row().cells
            self._docx_set_cell_text(row_cells[0], key, bold=True)
            self._docx_set_cell_text(row_cells[1], value)
            self._docx_set_cell_shading(row_cells[0], "EAF1F8")
        metadata_table.columns[0].width = Inches(1.6)
        metadata_table.columns[1].width = Inches(5.2)
        document.add_page_break()
        document.add_heading("Contents", level=1)
        sections = self.documentation_sections()
        for section_index, section in enumerate(sections, 1):
            document.add_paragraph(f"{section_index}. {section['menu']}")
            for subsection_index, subsection in enumerate(section["subsections"], 1):
                paragraph = document.add_paragraph(f"{section_index}.{subsection_index}. {subsection['name']}")
                paragraph.paragraph_format.left_indent = Inches(0.3)
        document.add_page_break()
        document.add_heading("Technique Documentation", level=1)
        for section_index, section in enumerate(sections, 1):
            document.add_heading(f"{section_index}. {section['menu']}", level=1)
            for subsection_index, subsection in enumerate(section["subsections"], 1):
                document.add_heading(f"{section_index}.{subsection_index}. {subsection['name']}", level=2)
                for technique_index, entry in enumerate(subsection["entries"], 1):
                    document.add_heading(f"{section_index}.{subsection_index}.{technique_index}. {entry['technique']}", level=3)
                    detail_table = document.add_table(rows=4, cols=2)
                    detail_table.style = "Table Grid"
                    detail_table.alignment = WD_TABLE_ALIGNMENT.CENTER
                    details = (
                        ("Menu path", f"{entry['menu']} > {entry['technique']}"),
                        ("Inputs", entry["inputs"]),
                        ("Process", entry["process"]),
                        ("Expected outputs", entry["outputs"]),
                    )
                    for row_index, (label, value) in enumerate(details):
                        label_cell = detail_table.cell(row_index, 0)
                        value_cell = detail_table.cell(row_index, 1)
                        self._docx_set_cell_text(label_cell, label, bold=True)
                        self._docx_set_cell_text(value_cell, value)
                        self._docx_set_cell_shading(label_cell, "EAF1F8")
                    detail_table.columns[0].width = Inches(1.6)
                    detail_table.columns[1].width = Inches(5.2)
                    self._docx_add_code_block(document, "App console", entry["app"])
                    self._docx_add_code_block(document, "df console", entry["df"])
        return document

    def documentation_pdf(self):
        pdf = DocumentationPDF(self)
        pdf.add_page()
        pdf.set_fill_color(31, 78, 121)
        pdf.rect(0, 0, pdf.w, 38, "F")
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Arial", "B", 16)
        pdf.set_y(5)
        pdf.multi_cell(pdf.content_width, 8, pdf.safe(APP_NAME), align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Arial", "B", 11)
        pdf.cell(0, 7, pdf.safe(APP_VERSION), align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)
        pdf.ln(8)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, pdf.safe("DOCUMENTATION GUIDE"), align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)
        pdf.section_heading("Title Section", level=1)
        for key, value in self.documentation_metadata():
            pdf.detail_row(key, value)
        pdf.add_page()
        pdf.set_font("Arial", "B", 15)
        pdf.cell(0, 9, pdf.safe("Contents"), align="L", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)
        sections = self.documentation_sections()
        for section_index, section in enumerate(sections, 1):
            pdf.set_font("Arial", "B", 11)
            pdf.cell(pdf.content_width, 7, pdf.safe(f"{section_index}. {section['menu']}"), align="L", new_x="LMARGIN", new_y="NEXT")
            for subsection_index, subsection in enumerate(section["subsections"], 1):
                pdf.set_font("Arial", size=10)
                pdf.cell(pdf.content_width, 6, pdf.safe(f"  {section_index}.{subsection_index}. {subsection['name']}"), align="L", new_x="LMARGIN", new_y="NEXT")
        pdf.add_page()
        pdf.set_font("Arial", "B", 15)
        pdf.cell(0, 9, pdf.safe("Technique Documentation"), align="L", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)
        for section_index, section in enumerate(sections, 1):
            pdf.section_heading(f"{section_index}. {section['menu']}", level=1)
            for subsection_index, subsection in enumerate(section["subsections"], 1):
                pdf.section_heading(f"{section_index}.{subsection_index}. {subsection['name']}", level=2)
                for technique_index, entry in enumerate(subsection["entries"], 1):
                    pdf.section_heading(f"{section_index}.{subsection_index}.{technique_index}. {entry['technique']}", level=3)
                    pdf.detail_row("Menu path", f"{entry['menu']} > {entry['technique']}")
                    pdf.detail_row("Inputs", entry["inputs"])
                    pdf.detail_row("Process", entry["process"])
                    pdf.detail_row("Expected outputs", entry["outputs"])
                    pdf.code_block("App console", entry["app"])
                    pdf.code_block("df console", entry["df"])
                    pdf.ln(2)
        return pdf

    def download_documentation(self, file_format="txt"):
        format_key = str(file_format).strip().lower().lstrip(".")
        if format_key == "markdown":
            format_key = "md"
        labels = {"txt": "Text File", "md": "Markdown File", "docx": "Word Document", "pdf": "PDF File"}
        if format_key not in labels:
            raise ValueError(f"Unsupported documentation format: {file_format}")
        file_path = filedialog.asksaveasfilename(
            defaultextension=f".{format_key}",
            initialfile=f"{APP_SHORT_NAME.lower()}_documentation_{APP_VERSION.replace(' ', '_').lower()}",
            filetypes=[(labels[format_key], f"*.{format_key}")]
        )
        if not file_path:
            return None
        if format_key == "txt":
            Path(file_path).write_text(self.documentation_text(), encoding="utf-8")
        elif format_key == "md":
            Path(file_path).write_text(self.documentation_markdown(), encoding="utf-8")
        elif format_key == "docx":
            self.documentation_docx().save(file_path)
        elif format_key == "pdf":
            self.documentation_pdf().output(file_path)
        messagebox.showinfo("Success", f"Documentation saved as {format_key.upper()}.")
        return file_path

    def open_documentation(self):
        documentation_window = DocumentationWindow(self.root, self)
        documentation_window.lift()
        return documentation_window

    def show_message(self, title, message):
        messagebox.showinfo(title, message)

    def open_plot_builder(self, plot_type="auto"):
        if self.data is None:
            messagebox.showerror("Error", "No data loaded!")
            return None
        dialog = PlotBuilderDialog(self.root, self, plot_type)
        dialog.lift()
        return dialog

    def generate_plot(self):
        return self.open_plot_builder("auto")

    def show_plot_help(self):
        messagebox.showinfo(
            "Plot Help",
            "Use Plots > Plot Builder or the Plots toolbar button.\n\n"
            "Choose a plot type, select variables from the column list or dropdowns, "
            "set options such as bins, aggregation, top N, grouping, size, and title, then click Plot.\n\n"
            "Python console examples:\n"
            "  app.histogram('sales')\n"
            "  app.bar_chart('region', 'sales')\n"
            "  app.pie_chart('region', 'sales')\n"
            "  app.scatter_plot('sales', 'profit', hue='region')\n"
            "  app.heatmap(['sales', 'profit', 'discount'])\n"
            "  app.pareto('region', 'sales')"
        )

    def show_plot_window(self, fig, title):
        return self.show_report_window(None, title, fig=fig)

    def download_plot_report(self, fig, title, figures=None):
        if self.license_type != "FULL":
            messagebox.showinfo("Feature Locked", "Unlock this feature by buying a license. Go to Help menu for more details.")
            return
        all_figures = []
        if fig is not None:
            all_figures.append(fig)
        if figures:
            all_figures.extend([figure for figure in figures if figure is not None])
        if self._export_report_files(title, None, all_figures):
            messagebox.showinfo("Success", "Report saved successfully!")

    def _require_data(self):
        if self.data is None:
            raise ValueError("No data loaded. Use app.load_csv(path), app.set_df(df), or a demo-data generator.")
        return self.data

    def _as_column_list(self, columns):
        if columns is None:
            columns = self._default_item_columns()
        if isinstance(columns, str):
            return [column.strip() for column in columns.split(",") if column.strip()]
        if isinstance(columns, (list, tuple, set, pd.Index)):
            return list(columns)
        return [columns]

    def _resolve_columns(self, columns):
        resolved = self._as_column_list(columns)
        if isinstance(resolved, str):
            resolved = [resolved]
        missing = [column for column in resolved if column not in self._require_data().columns]
        if missing:
            raise ValueError(f"Columns not found: {', '.join(missing)}")
        return resolved

    def _sync_console_df(self):
        if getattr(self, "console", None) is not None:
            self.console.set_df(self.data)

    def open_console(self):
        try:
            console_exists = self.console is not None and self.console.winfo_exists()
        except tk.TclError:
            console_exists = False
        if not console_exists:
            self.console = PythonConsoleWindow(self.root, self)
        self.console.deiconify()
        self.console.lift()
        self.console.focus_set()
        self._sync_console_df()
        return self.console

    def set_df(self, frame, refresh=True):
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("frame must be a pandas DataFrame")
        self.data = frame
        self.file_path = None
        if refresh:
            self.display_data()
        return self.data

    def get_df(self):
        return self._require_data()

    def refresh_df(self):
        self._sync_console_df()
        return self.data

    def load_csv(self, file_path):
        frame = pd.read_csv(file_path)
        self.file_path = str(file_path)
        self.data = frame
        self.display_data()
        return frame

    def save_csv(self, file_path, index=False):
        self._require_data().to_csv(file_path, index=index)
        return str(file_path)

    def head(self, n=5):
        return self._require_data().head(n)

    def tail(self, n=5):
        return self._require_data().tail(n)

    def describe(self, include="all"):
        return self._require_data().describe(include=include)

    def dtypes(self):
        return self._require_data().dtypes.to_frame("dtype")

    def columns(self):
        return list(self._require_data().columns)

    def shape(self):
        return self._require_data().shape

    def info_text(self):
        buffer = io.StringIO()
        self._require_data().info(buf=buffer)
        return buffer.getvalue()

    def missing(self):
        frame = self._require_data()
        return frame.isna().sum().to_frame("Missing")

    def value_counts(self, column, normalize=False, dropna=True):
        return self._require_data()[column].value_counts(normalize=normalize, dropna=dropna).rename_axis("Value").reset_index(name="Count")

    def groupby_agg(self, by, aggregation):
        return self._require_data().groupby(by).agg(aggregation)

    def pivot(self, index, columns, values, aggfunc="mean"):
        return pd.pivot_table(self._require_data(), index=index, columns=columns, values=values, aggfunc=aggfunc)

    def correlation(self, columns=None, method="pearson"):
        frame = self._require_data()
        if columns is None:
            return frame.corr(method=method, numeric_only=True)
        return frame[self._resolve_columns(columns)].corr(method=method, numeric_only=True)

    def _computed_value_length(self, result, frame):
        if isinstance(result, pd.DataFrame):
            if result.shape[1] != 1:
                raise ValueError("Expression returned more than one column; return a single result.")
            result = result.iloc[:, 0]
        if np.isscalar(result):
            return [result] * len(frame)
        values = pd.Series(result).reset_index(drop=True)
        if len(values) != len(frame):
            raise ValueError(f"Expression returned {len(values)} values for {len(frame)} rows.")
        return values

    def evaluate_expression(self, expression, frame=None):
        frame = self._require_data() if frame is None else frame
        text = str(expression).strip()
        if not text:
            raise ValueError("Enter a formula to compute.")
        functions = {
            "log": np.log,
            "log10": np.log10,
            "log2": np.log2,
            "sqrt": np.sqrt,
            "exp": np.exp,
            "abs": np.abs,
            "round": np.round,
            "floor": np.floor,
            "ceil": np.ceil,
            "sin": np.sin,
            "cos": np.cos,
            "tan": np.tan,
            "len": len,
        }

        def reduce_values(reducer, items):
            stacked = np.vstack([np.asarray(item, dtype=float) for item in items])
            return reducer(stacked, axis=0) if len(items) > 1 else reducer(stacked)

        def stack_mean(*items):
            return reduce_values(np.mean, items)

        def stack_median(*items):
            return reduce_values(np.median, items)

        def stack_min(*items):
            return reduce_values(np.min, items)

        def stack_max(*items):
            return reduce_values(np.max, items)

        def stack_sum(*items):
            return reduce_values(np.sum, items)

        functions.update({"mean": stack_mean, "median": stack_median, "min": stack_min, "max": stack_max, "sum": stack_sum})
        try:
            result = frame.eval(text, engine="python", local_dict=functions)
            return self._computed_value_length(result, frame)
        except Exception as first_error:
            namespace = {column: frame[column] for column in frame.columns}
            namespace.update(functions)
            namespace.update({"np": np, "pd": pd, "frame": frame, "df": frame})
            try:
                result = eval(text, {"__builtins__": {}}, namespace)
            except Exception as second_error:
                raise ValueError(f"Invalid expression: {first_error}") from second_error
            return self._computed_value_length(result, frame)

    def compute_column(self, name, expression):
        frame = self._require_data()
        target = self._unique_column_name([column for column in frame.columns if str(column) != str(name)], name)
        frame[target] = self.evaluate_expression(expression, frame)
        self.data = frame
        self.display_data()
        return frame

    def recode_column(self, column, mapping, new_column=None):
        frame = self._require_data()
        new_column = new_column or f"{column}_Recoded"
        frame[new_column] = frame[column].replace(mapping)
        self.display_data()
        return frame

    def treat_missing(self, column, method="mean"):
        frame = self._require_data()
        method_name = method.lower().replace(" ", "_")
        if method_name in ("mean", "median"):
            value = frame[column].mean() if method_name == "mean" else frame[column].median()
            frame[column] = frame[column].fillna(value)
        elif method_name == "mode":
            modes = frame[column].mode()
            if modes.empty:
                raise ValueError(f"No mode available for column '{column}'")
            frame[column] = frame[column].fillna(modes.iloc[0])
        elif method_name in ("drop_rows", "drop"):
            self.data = frame.dropna(subset=[column]).copy()
            self.display_data()
            return self.data
        elif method_name in ("drop_column", "drop_col"):
            self.data = frame.drop(columns=[column]).copy()
            self.display_data()
            return self.data
        else:
            raise ValueError("method must be mean, median, mode, drop_rows, or drop_column")
        self.display_data()
        return frame

    def _missing_overview_figure(self, before, after, columns, title):
        figure, axis = plt.subplots(figsize=(8, 4))
        positions = np.arange(len(columns))
        axis.bar(positions - 0.18, before, 0.36, label="Before")
        axis.bar(positions + 0.18, after, 0.36, label="After")
        axis.set_xticks(positions)
        axis.set_xticklabels(columns, rotation=45, ha="right")
        axis.set_ylabel("Missing values")
        axis.set_title(title)
        axis.legend()
        figure.tight_layout()
        return figure

    def _missingness_weights(self, values):
        numeric = pd.to_numeric(values, errors="coerce")
        if numeric.notna().any():
            ranks = numeric.rank(pct=True, na_option="keep").fillna(0.5).to_numpy(dtype=float)
        else:
            codes, _ = pd.factorize(values.astype(str))
            unique_count = len(set(codes))
            ranks = (np.asarray(codes, dtype=float) + 1) / max(unique_count, 1)
        return 0.25 + 1.5 * ranks

    def _imputation_matrix(self, frame, predictors, train_index, target_index):
        training = frame.loc[train_index, predictors].copy()
        scoring = frame.loc[target_index, predictors].copy()
        combined = pd.concat([training, scoring], axis=0)
        combined = pd.get_dummies(combined, dummy_na=False)
        if combined.shape[1] == 0:
            raise ValueError("At least one predictor column is required.")
        fills = combined.iloc[:len(training)].median(numeric_only=True).reindex(combined.columns).fillna(0)
        return combined.iloc[:len(training)].fillna(fills), combined.iloc[len(training):].fillna(fills)

    def open_advanced_imputation(self):
        if self.data is None:
            messagebox.showerror("Error", "No data loaded!")
            return None
        dialog = AdvancedImputationDialog(self.root, self)
        dialog.lift()
        return dialog

    def open_missing_data_simulation(self):
        if self.data is None:
            messagebox.showerror("Error", "No data loaded!")
            return None
        dialog = MissingDataSimulationDialog(self.root, self)
        dialog.lift()
        return dialog

    def open_missingness_test(self):
        if self.data is None:
            messagebox.showerror("Error", "No data loaded!")
            return None
        dialog = MissingnessTestDialog(self.root, self)
        dialog.lift()
        return dialog

    def treat_missing_advanced(self, column, method="regression", predictors=None, show=True):
        frame = self._require_data().copy()
        if column not in frame.columns:
            raise ValueError(f"Column not found: {column}")
        method_name = str(method).strip().lower().replace(" ", "_")
        method_labels = {
            "regression": "Regression imputation",
            "linear_regression": "Regression imputation",
            "linear": "Regression imputation",
            "random_forest": "Random forest imputation",
            "rf": "Random forest imputation",
            "ml": "Random forest imputation",
            "knn": "K-nearest neighbors imputation",
            "knn_imputer": "K-nearest neighbors imputation",
            "iterative": "Iterative imputation",
            "iterative_imputer": "Iterative imputation",
            "em": "Iterative imputation"
        }
        if method_name not in method_labels:
            raise ValueError("method must be regression, random_forest, knn, or iterative")
        missing_mask = frame[column].isna()
        if not missing_mask.any():
            raise ValueError(f"Column '{column}' has no missing values to impute.")
        if predictors is None:
            predictors = [candidate for candidate in frame.columns if candidate != column and pd.api.types.is_numeric_dtype(frame[candidate])]
        else:
            predictors = self._as_column_list(predictors)
        invalid = [candidate for candidate in predictors if candidate not in frame.columns or candidate == column]
        if invalid:
            raise ValueError(f"Invalid predictor columns: {', '.join(invalid)}")
        if not predictors:
            raise ValueError("Select at least one predictor column.")
        train_index = frame.index[frame[column].notna()]
        target_index = frame.index[missing_mask]
        if len(train_index) == 0:
            raise ValueError("At least one observed value is required for imputation.")
        label = method_labels[method_name]
        if method_name in ("knn", "knn_imputer", "iterative", "iterative_imputer", "em"):
            numeric = frame[[column] + predictors].apply(pd.to_numeric, errors="coerce")
            if numeric[column].isna().sum() != int(missing_mask.sum()):
                raise ValueError(f"{label} requires a numeric target column and numeric predictors.")
            if method_name in ("knn", "knn_imputer"):
                imputer = KNNImputer(n_neighbors=min(5, max(1, len(numeric))))
            else:
                imputer = IterativeImputer(random_state=42, max_iter=20)
            imputed = imputer.fit_transform(numeric)
            frame.loc[target_index, column] = pd.Series(imputed[:, 0], index=frame.index).loc[target_index]
        else:
            if method_name in ("regression", "linear_regression", "linear") and not pd.api.types.is_numeric_dtype(frame[column]):
                raise ValueError("Regression imputation requires a numeric target column; use random_forest for categorical targets.")
            x_train, x_missing = self._imputation_matrix(frame, predictors, train_index, target_index)
            y_train = frame.loc[train_index, column]
            if pd.api.types.is_numeric_dtype(frame[column]):
                model = LinearRegression() if method_name in ("regression", "linear_regression", "linear") else RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            else:
                model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(x_train, y_train)
            frame.loc[target_index, column] = model.predict(x_missing)
        self.data = frame
        self.display_data()
        before = np.asarray([int(missing_mask.sum())])
        after = np.asarray([int(frame[column].isna().sum())])
        report = {
            "Method": label,
            "Target column": column,
            "Predictors": ", ".join(predictors),
            "Rows imputed": int(missing_mask.sum()),
            "Missing before": int(missing_mask.sum()),
            "Missing after": int(frame[column].isna().sum()),
            "Imputed rows preview": frame.loc[target_index, [column] + predictors].head(10)
        }
        if show:
            figure = self._missing_overview_figure(before, after, [column], f"{label}: Missing Values Before and After")
            self.show_report_window(report, f"{label} Report", fig=figure)
        return frame

    def simulate_missing_data(self, columns=None, rate=0.2, mechanism="mcar", predictor=None, seed=42, apply=True, show=True):
        frame = self._require_data().copy()
        if columns is None:
            columns = list(frame.columns)
        else:
            columns = self._as_column_list(columns)
        invalid = [column for column in columns if column not in frame.columns]
        if invalid:
            raise ValueError(f"Columns not found: {', '.join(invalid)}")
        if not columns:
            raise ValueError("Select at least one column.")
        rate = float(rate)
        if not 0 <= rate <= 1:
            raise ValueError("rate must be between 0 and 1.")
        mechanism_name = str(mechanism).strip().lower()
        aliases = {"mcar": "MCAR", "missing_completely_at_random": "MCAR", "mar": "MAR", "missing_at_random": "MAR", "mnar": "MNAR", "missing_not_at_random": "MNAR"}
        if mechanism_name not in aliases:
            raise ValueError("mechanism must be mcar, mar, or mnar.")
        mechanism_label = aliases[mechanism_name]
        if mechanism_label == "MAR":
            if predictor not in frame.columns:
                raise ValueError(f"Predictor column not found: {predictor}")
        rng = np.random.default_rng(int(seed))
        before = frame[columns].isna().sum()
        for column in columns:
            eligible_index = frame.index[frame[column].notna()]
            if len(eligible_index) == 0:
                continue
            if mechanism_label == "MCAR":
                weights = np.ones(len(eligible_index), dtype=float)
            elif mechanism_label == "MAR":
                weights = self._missingness_weights(frame.loc[eligible_index, predictor])
            else:
                weights = self._missingness_weights(frame.loc[eligible_index, column])
            mean_weight = float(np.mean(weights))
            if mean_weight <= 0:
                continue
            probabilities = np.clip(rate * weights / mean_weight, 0, 0.95)
            hits = rng.random(len(eligible_index)) < probabilities
            frame.loc[eligible_index[hits], column] = np.nan
        after = frame[columns].isna().sum()
        summary = pd.DataFrame({
            "Column": columns,
            "Missing before": [int(before[column]) for column in columns],
            "Missing after": [int(after[column]) for column in columns],
            "Introduced": [int(after[column] - before[column]) for column in columns],
            "Percent after": [round(float(after[column] / len(frame) * 100), 2) for column in columns]
        })
        if apply:
            self.data = frame
            self.display_data()
        report = {
            "Mechanism": f"{mechanism_label} ({'missing completely at random' if mechanism_label == 'MCAR' else 'missing at random' if mechanism_label == 'MAR' else 'missing not at random'})",
            "Target rate": rate,
            "Seed": int(seed),
            "Predictor": predictor if mechanism_label == "MAR" else "None",
            "Applied to grid": bool(apply),
            "Missing summary": summary
        }
        if show:
            figure = self._missing_overview_figure(before.to_numpy(), after.to_numpy(), columns, "Simulated Missing Values")
            self.show_report_window(report, "Missing Data Simulation Report", fig=figure)
        return frame

    def test_missing_significance(self, columns=None, group_column=None, alpha=0.05, show=True):
        frame = self._require_data()
        if columns is None:
            columns = [column for column in frame.columns if frame[column].isna().any()]
        else:
            columns = self._as_column_list(columns)
        columns = [column for column in columns if column in frame.columns]
        if not columns:
            raise ValueError("No columns with missing values were selected.")
        if group_column is not None and group_column not in frame.columns:
            raise ValueError(f"Group column not found: {group_column}")
        alpha = float(alpha)
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1.")
        results = []
        if group_column is not None:
            adjusted_alpha = alpha / max(1, len(columns))
            for column in columns:
                indicator = frame[column].isna()
                if indicator.nunique() < 2:
                    results.append({"Column": column, "Test": "Chi-square", "Variable": group_column, "Status": "Not testable: no variation in missingness"})
                    continue
                indicator_labels = indicator.map({False: "Present", True: "Missing"})
                table = pd.crosstab(frame[group_column], indicator_labels)
                table = table.reindex(columns=["Present", "Missing"], fill_value=0)
                if table.shape[0] < 2 or int(table.values.sum()) == 0:
                    results.append({"Column": column, "Test": "Chi-square", "Variable": group_column, "Status": "Not testable"})
                    continue
                statistic, p_value, degrees, _ = stats.chi2_contingency(table)
                results.append({
                    "Column": column,
                    "Test": "Chi-square",
                    "Variable": group_column,
                    "Statistic": round(float(statistic), 4),
                    "p-value": round(float(p_value), 6),
                    "df": int(degrees),
                    "Adjusted alpha": round(adjusted_alpha, 6),
                    "Significant": bool(p_value < adjusted_alpha)
                })
            method = "Chi-square test of independence"
        else:
            other_columns = [column for column in frame.columns if column not in columns]
            numeric_predictors = [column for column in other_columns if pd.api.types.is_numeric_dtype(frame[column])]
            categorical_predictors = [column for column in other_columns if column not in numeric_predictors]
            for column in columns:
                indicator = frame[column].isna()
                if indicator.nunique() < 2:
                    results.append({"Column": column, "Test": "Screening", "Variable": "None", "Status": "Not testable: no variation in missingness"})
                    continue
                for predictor in numeric_predictors:
                    missing_values = frame.loc[indicator, predictor].dropna()
                    observed_values = frame.loc[~indicator, predictor].dropna()
                    if len(missing_values) < 2 or len(observed_values) < 2:
                        continue
                    statistic, p_value = stats.ttest_ind(missing_values, observed_values, equal_var=False)
                    results.append({
                        "Column": column,
                        "Test": "Welch t-test",
                        "Variable": predictor,
                        "Statistic": round(float(statistic), 4),
                        "p-value": round(float(p_value), 6),
                        "Status": "Missing vs observed groups"
                    })
                for predictor in categorical_predictors:
                    indicator_labels = indicator.map({False: "Present", True: "Missing"})
                    table = pd.crosstab(frame[predictor], indicator_labels)
                    if table.empty:
                        continue
                    table = table.reindex(columns=["Present", "Missing"], fill_value=0)
                    if table.shape[0] < 2 or int(table.values.sum()) == 0:
                        continue
                    statistic, p_value, degrees, _ = stats.chi2_contingency(table)
                    results.append({
                        "Column": column,
                        "Test": "Chi-square",
                        "Variable": predictor,
                        "Statistic": round(float(statistic), 4),
                        "p-value": round(float(p_value), 6),
                        "df": int(degrees),
                        "Status": "Missing indicator vs category"
                    })
            adjusted_alpha = alpha / max(1, len(results))
            for result in results:
                if "p-value" in result:
                    result["Adjusted alpha"] = round(adjusted_alpha, 6)
                    result["Significant"] = bool(result["p-value"] < adjusted_alpha)
            method = "Observed-variable screening with Bonferroni correction"
        results_frame = pd.DataFrame(results)
        significant_count = int(results_frame["Significant"].sum()) if "Significant" in results_frame.columns else 0
        if significant_count:
            interpretation = f"Missingness is significantly associated with at least one tested variable at alpha={alpha} after correction."
        elif group_column is not None:
            interpretation = f"No tested column shows a significant missingness association with '{group_column}'."
        else:
            interpretation = "No significant association was detected between missingness and the observed variables."
        missing_summary = pd.DataFrame({
            "Column": columns,
            "Missing": [int(frame[column].isna().sum()) for column in columns],
            "Percent": [round(float(frame[column].isna().mean() * 100), 2) for column in columns]
        })
        report = {
            "Method": method,
            "Group column": group_column or "None",
            "Alpha": alpha,
            "Significant results": significant_count,
            "Interpretation": interpretation,
            "Test results": results_frame,
            "Missing summary": missing_summary
        }
        if show:
            values = missing_summary["Percent"].to_numpy()
            figure = self._missing_overview_figure(values, np.zeros_like(values), columns, "Missing Percentage by Column")
            self.show_report_window(report, "Missingness Significance Report", fig=figure)
        return report

    def missingness_report(self, show=True):
        frame = self._require_data()
        summary = pd.DataFrame({
            "Column": list(frame.columns),
            "Missing": [int(frame[column].isna().sum()) for column in frame.columns],
            "Present": [int(frame[column].notna().sum()) for column in frame.columns],
            "Percent": [round(float(frame[column].isna().mean() * 100), 2) for column in frame.columns]
        })
        patterns = frame.isna().apply(lambda row: "".join("1" if flag else "0" for flag in row), axis=1).value_counts()
        pattern_table = patterns.rename_axis("Pattern").reset_index(name="Rows")
        complete_rows = int((~frame.isna().any(axis=1)).sum())
        report = {
            "Rows": len(frame),
            "Columns": len(frame.columns),
            "Complete rows": complete_rows,
            "Rows with any missing value": len(frame) - complete_rows,
            "Missing summary": summary,
            "Missing patterns": pattern_table
        }
        if show:
            values = summary["Percent"].to_numpy()
            figure = self._missing_overview_figure(values, np.zeros_like(values), list(frame.columns), "Missing Percentage by Column")
            self.show_report_window(report, "Missingness Report", fig=figure)
        return report

    def standardize_column(self, column, method="Min-Max"):
        frame = self._require_data().copy()
        new_column = f"{column}_{method}"
        if method == "Min-Max":
            frame[new_column] = MinMaxScaler().fit_transform(frame[[column]])
        elif method == "Z-Score":
            frame[new_column] = StandardScaler().fit_transform(frame[[column]])
        elif method == "Decimal Scaling":
            max_abs = frame[column].abs().max()
            frame[new_column] = frame[column] / (10 ** np.ceil(np.log10(max_abs)))
        elif method == "Log":
            frame[new_column] = np.log(frame[column] + 1)
        elif method == "Log-Normal":
            frame[new_column] = np.exp(frame[column])
        else:
            raise ValueError("method must be Min-Max, Z-Score, Decimal Scaling, Log, or Log-Normal")
        self.data = frame
        self.display_data()
        return frame

    def decompose_columns(self, columns, method="svd", append=True):
        frame = self._require_data()
        selected = self._resolve_columns(columns)
        matrix = frame[selected].values.astype(float)
        if method == "cholesky":
            result = np.linalg.cholesky(matrix @ matrix.T)
        elif method == "qr":
            result, _ = np.linalg.qr(matrix)
        elif method == "svd":
            result = np.linalg.svd(matrix, full_matrices=False)[0]
        elif method == "eig":
            result = np.linalg.eig(matrix)[1]
        else:
            raise ValueError("method must be cholesky, qr, svd, or eig")
        result_frame = pd.DataFrame(result, columns=[f"{method}_comp_{index + 1}" for index in range(result.shape[1])])
        if append:
            self.data = pd.concat([frame.reset_index(drop=True), result_frame.reset_index(drop=True)], axis=1)
            self.display_data()
        return result_frame

    def simulate(self, specs, rows=100, seed=None, append=True):
        rng = np.random.default_rng(seed)
        generated = {}
        for name, spec in specs.items():
            if isinstance(spec, str) and ":" in spec:
                spec = spec.split(":", 1)
            if not isinstance(spec, (list, tuple)):
                spec = [spec]
            kind = str(spec[0]).lower()
            if kind in ("int", "integer"):
                generated[name] = rng.integers(int(spec[1]), int(spec[2]), rows)
            elif kind in ("float", "uniform"):
                generated[name] = rng.uniform(float(spec[1]), float(spec[2]), rows)
            elif kind in ("normal", "gaussian"):
                generated[name] = rng.normal(float(spec[1]), float(spec[2]), rows)
            elif kind in ("category", "categorical", "choice"):
                choices = spec[1] if isinstance(spec[1], (list, tuple, set, pd.Index)) else str(spec[1]).split(",")
                generated[name] = rng.choice([str(choice).strip() for choice in choices], rows)
            else:
                raise ValueError(f"Unsupported simulation type: {spec[0]}")
        frame = pd.DataFrame(generated)
        if append and self.data is not None:
            frame = pd.concat([self.data.reset_index(drop=True), frame.reset_index(drop=True)], axis=1)
        self.data = frame
        self.display_data()
        return frame

    def simulate_multivariate_normal(self, names, means, covariance, samples=100, seed=None, append=False):
        rng = np.random.default_rng(seed)
        values = rng.multivariate_normal(np.asarray(means, dtype=float), np.asarray(covariance, dtype=float), size=samples)
        frame = pd.DataFrame(values, columns=self._as_column_list(names))
        if append and self.data is not None:
            frame = pd.concat([self.data.reset_index(drop=True), frame.reset_index(drop=True)], axis=1)
        self.data = frame
        self.display_data()
        return frame

    def generate_basket(self, transactions=200, item_count=10, basket_size=3, strength=0.6, seed=42, apply=True):
        frame = self.generate_wide_basket_data(transactions, item_count, basket_size, strength, seed)
        if apply:
            self.set_df(frame)
        return frame

    def generate_long_transactions(self, customers=50, transactions_per_customer=4, item_count=10, basket_size=3, seed=42, apply=True):
        frame = self.generate_long_transaction_data(customers, transactions_per_customer, item_count, basket_size, seed)
        if apply:
            self.set_df(frame)
        return frame

    def generate_sequence_log(self, customers=50, sequence_length=6, item_count=10, seed=42, apply=True):
        frame = self.generate_sequence_data(customers, sequence_length, item_count, seed)
        if apply:
            self.set_df(frame)
        return frame

    def generate_profit_basket(self, transactions=200, item_count=10, basket_size=3, seed=42, apply=True):
        frame = self.generate_profit_basket_data(transactions, item_count, basket_size, seed)
        if apply:
            self.set_df(frame)
        return frame

    def transaction_matrix(self, columns=None):
        return self._build_transaction_matrix(self._resolve_columns(columns))

    def frequent_itemsets(self, columns=None, min_support=0.1, algorithm="Apriori", max_len=None, top_n=100, show=False):
        matrix = self.transaction_matrix(columns)
        frequent = self._mine_frequent_itemsets(algorithm, matrix, min_support, max_len)
        if frequent is None:
            return pd.DataFrame(columns=["support", "itemsets"])
        result = self._format_itemsets(frequent.head(top_n), "itemsets")
        if show:
            self.show_report_window(result, f"{algorithm} Frequent Itemsets")
        return result

    def closed_itemsets(self, columns=None, min_support=0.1, algorithm="Apriori", max_len=None, top_n=100, show=False):
        matrix = self.transaction_matrix(columns)
        frequent = self._mine_frequent_itemsets(algorithm, matrix, min_support, max_len)
        if frequent is None:
            return pd.DataFrame(columns=["support", "itemsets"])
        result = self._format_itemsets(self._closed_frequent_itemsets(frequent).head(top_n), "itemsets")
        if show:
            self.show_report_window(result, "Closed Frequent Itemsets")
        return result

    def maximal_itemsets(self, columns=None, min_support=0.1, max_len=None, top_n=100, show=False):
        from mlxtend.frequent_patterns import fpmax
        matrix = self.transaction_matrix(columns)
        frequent = fpmax(matrix, min_support=min_support, use_colnames=True, max_len=max_len)
        result = self._format_itemsets(frequent.head(top_n), "itemsets")
        if show:
            self.show_report_window(result, "Maximal Frequent Itemsets")
        return result

    def association_rules(self, columns=None, min_support=0.1, min_confidence=0.6, min_lift=1.0, algorithm="Apriori", max_len=None, top_n=100, show=False):
        from mlxtend.frequent_patterns import association_rules as mlxtend_rules
        matrix = self.transaction_matrix(columns)
        frequent = self._mine_frequent_itemsets(algorithm, matrix, min_support, max_len)
        if frequent is None or frequent.empty:
            return pd.DataFrame()
        rules = mlxtend_rules(frequent, num_itemsets=len(matrix), metric="confidence", min_threshold=min_confidence)
        rules = rules[rules["lift"] >= min_lift].copy()
        result = self._format_rules(rules.head(top_n))
        if show:
            self.show_report_window(result, "Association Rules", self._build_rules_plot(rules))
        return result

    def market_basket(self, columns=None, min_support=0.1, min_confidence=0.6, min_lift=1.0, algorithm="Apriori", max_len=None, top_n=100, show=False):
        result = {
            "Frequent Itemsets": self.frequent_itemsets(columns, min_support, algorithm, max_len, top_n),
            "Association Rules": self.association_rules(columns, min_support, min_confidence, min_lift, algorithm, max_len, top_n)
        }
        if show:
            self.show_report_window(result, "Market Basket Analysis")
        return result

    def sequential_patterns(self, sequence_column=None, min_support=0.2, max_len=5, top_n=100, show=False):
        database = self._build_sequence_database(sequence_column)
        patterns = self._prefixspan_frequent_sequences(database, min_support, max_len).head(top_n)
        if show:
            self.show_report_window(patterns, "Sequential Patterns")
        return patterns

    def rfm(self, customer_column="CustomerID", date_column="Date", amount_column="Amount", snapshot=None, show=False):
        frame, summary = self._compute_rfm(self._require_data(), customer_column, date_column, amount_column, snapshot)
        if show:
            self.show_report_window({"RFM Scores": frame.reset_index(), "Segment Summary": summary}, "RFM Segmentation")
        return frame, summary

    def profit_weighted_rules(self, columns=None, amount_column="Amount", min_support=0.1, min_confidence=0.6, min_lift=1.0, algorithm="Apriori", max_len=None, top_n=100, show=False):
        frame = self._require_data()
        if amount_column not in frame.columns:
            raise ValueError(f"Amount column not found: {amount_column}")
        matrix = self.transaction_matrix(columns)
        amounts = pd.to_numeric(frame.loc[matrix.index, amount_column], errors="coerce").fillna(0)
        weighted = self._compute_profit_weighted_rules(matrix, amounts, algorithm, min_support, min_confidence, min_lift, max_len, top_n)
        result = self._format_rules(weighted)
        if show:
            self.show_report_window(result, "Profit-Weighted Rules", self._build_rules_plot(weighted))
        return result

    def recommendations(self, columns=None, min_support=0.05, min_confidence=0.3, top_n=5, show=False):
        matrix = self.transaction_matrix(columns)
        result = self._compute_affinity_recommendations(matrix, min_support, min_confidence, top_n)
        if show:
            self.show_report_window(result, "Recommendation / Affinity Report", self._build_affinity_plot(result))
        return result

    def generate_marketing_log(self, customers=200, transactions_per_customer=3, seed=42, apply=True):
        frame = self.generate_marketing_data(customers, transactions_per_customer, seed)
        if apply:
            self.set_df(frame)
        return frame

    def customer_lifetime_value(self, customer_column="CustomerID", date_column="Date", amount_column="Amount", retention=0.95, horizon=12, show=False):
        result, summary = self._compute_clv(self._require_data(), customer_column, date_column, amount_column, retention, horizon)
        if show:
            self.show_report_window({"Customer CLV": result, "Summary": summary}, "Customer Lifetime Value (CLV)")
        return result, summary

    def churn_prediction(self, customer_column="CustomerID", target_column="Churn", features=None, test_size=0.25, show=False):
        features = list(features or [])
        result, coefficients, summary = self._compute_churn_prediction(self._require_data(), customer_column, target_column, features, test_size)
        if show:
            self.show_report_window({"Predictions": result, "Coefficients": coefficients, "Summary": summary}, "Churn Prediction")
        return result, coefficients, summary

    def campaign_response(self, target_column="Response", segment_column="CampaignChannel", value_column=None, show=False):
        result, summary = self._compute_campaign_uplift(self._require_data(), target_column, segment_column, value_column)
        if show:
            self.show_report_window({"Segment Uplift": result, "Summary": summary}, "Campaign Response Uplift")
        return result, summary

    def price_elasticity(self, price_column="Price", quantity_column="Quantity", show=False):
        summary, fitted = self._compute_price_elasticity(self._require_data(), price_column, quantity_column)
        if show:
            self.show_report_window({"Summary": summary, "Fitted Observations": fitted}, "Price Elasticity Analysis")
        return summary

    def frequencies(self, columns, show=False):
        selected = self._resolve_columns(columns)
        frame = self._require_data()
        parts = []
        for column in selected:
            counts = frame[column].value_counts(dropna=False).rename_axis("Value").reset_index(name="Count")
            counts.insert(0, "Column", column)
            parts.append(counts)
        result = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["Column", "Value", "Count"])
        if show:
            self.show_report_window(result, "Frequency Tables")
        return result

    def summary_stats(self, columns=None, show=False):
        frame = self._require_data()
        result = frame.describe(include="all") if columns is None else frame[self._resolve_columns(columns)].describe(include="all")
        if show:
            self.show_report_window(result, "Summary Statistics")
        return result

    def dataset_properties(self, frame=None, limit=None, show=False):
        frame = frame if frame is not None else self._require_data()
        row_count, column_count = frame.shape
        memory_bytes = int(frame.memory_usage(deep=True).sum())
        properties = pd.DataFrame([
            ("Rows", row_count),
            ("Columns", column_count),
            ("Dimensions", f"{row_count} x {column_count}"),
            ("Total cells", row_count * column_count),
            ("Memory usage (bytes)", f"{memory_bytes:,}"),
            ("Duplicate rows", int(frame.duplicated().sum())),
            ("Missing cells", int(frame.isna().sum().sum())),
            ("Numeric columns", int(frame.select_dtypes(include=[np.number]).shape[1])),
            ("Categorical/boolean columns", int(frame.select_dtypes(include=["object", "category", "bool"]).shape[1])),
            ("Datetime columns", int(frame.select_dtypes(include=["datetime64", "datetimetz"]).shape[1])),
            ("Index name", str(frame.index.name) if frame.index.name is not None else "(none)"),
            ("Column names", ", ".join(str(column) for column in frame.columns)),
        ], columns=["Property", "Value"])
        columns = pd.DataFrame({
            "Column": [str(column) for column in frame.columns],
            "Data Type": [str(dtype) for dtype in frame.dtypes],
            "Non-Null": frame.count().to_numpy(),
            "Null": frame.isna().sum().to_numpy(),
            "Unique": frame.nunique(dropna=True).to_numpy(),
        })
        rows = frame.copy()
        rows.insert(0, "Index", [str(index) for index in frame.index])
        rows.insert(0, "Row", np.arange(1, row_count + 1))
        if limit is not None:
            rows = rows.head(max(0, int(limit)))
        report = {"Properties": properties, "Columns": columns, "Rows": rows}
        if show:
            self.show_report_window(report, "Dataset Properties")
        return report

    def _coerce_cell_value(self, value, dtype):
        text = "" if value is None else str(value)
        stripped = text.strip()
        if not stripped:
            return np.nan
        if pd.api.types.is_numeric_dtype(dtype):
            try:
                number = float(stripped)
                if number.is_integer() and "." not in stripped and "e" not in stripped.lower():
                    return int(number)
                return number
            except ValueError:
                return text
        if pd.api.types.is_bool_dtype(dtype):
            lowered = stripped.lower()
            if lowered in ("true", "yes", "1", "t", "y"):
                return True
            if lowered in ("false", "no", "0", "f", "n"):
                return False
            return text
        if pd.api.types.is_datetime64_any_dtype(dtype):
            parsed = pd.to_datetime(text, errors="coerce")
            if not pd.isna(parsed):
                return parsed
        return text

    def _unique_column_name(self, columns, base):
        existing = {str(column) for column in columns}
        name = str(base).strip() or "NewColumn"
        if name not in existing:
            return name
        counter = 1
        while f"{name}_{counter}" in existing:
            counter += 1
        return f"{name}_{counter}"

    def insert_row(self, position=None, count=1, values=None):
        frame = self._require_data()
        count = max(1, int(count))
        position = len(frame) if position is None else max(0, min(int(position), len(frame)))
        supplied = dict(values or {})
        rows = pd.DataFrame(
            [{column: supplied.get(column, np.nan) for column in frame.columns} for _ in range(count)],
            columns=frame.columns,
        )
        self.data = pd.concat([frame.iloc[:position], rows, frame.iloc[position:]], ignore_index=True)
        self.display_data()
        return self.data

    def insert_column(self, position=None, name=None, values=None):
        frame = self._require_data()
        position = len(frame.columns) if position is None else max(0, min(int(position), len(frame.columns)))
        name = self._unique_column_name(frame.columns, name or f"Column_{len(frame.columns) + 1}")
        series = pd.Series([np.nan] * len(frame), index=frame.index, dtype=object)
        if values is not None:
            for offset, value in enumerate(list(values)[:len(frame)]):
                series.iloc[offset] = value
        frame.insert(position, name, series)
        self.data = frame
        self.display_data()
        return self.data

    def edit_cell(self, row, column, value):
        frame = self._require_data()
        position = int(row)
        if not (0 <= position < len(frame)):
            raise ValueError(f"Row position {position} is outside the data grid.")
        column_name = self._column_name_at(column) if isinstance(column, (int, np.integer)) else column
        if column_name not in frame.columns:
            raise ValueError(f"Column '{column}' was not found.")
        converted = self._coerce_cell_value(value, frame[column_name].dtype)
        frame.at[frame.index[position], column_name] = converted
        self.display_data()
        return frame

    def rename_column(self, column, new_name):
        frame = self._require_data()
        if column not in frame.columns:
            raise ValueError(f"Column '{column}' was not found.")
        name = self._unique_column_name([item for item in frame.columns if item != column], str(new_name).strip() or column)
        self.data = frame.rename(columns={column: name})
        self.display_data()
        return self.data

    def copy_column_names(self):
        text = ", ".join(str(column) for column in self._require_data().columns)
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.root.update()
        return text

    def ttest(self, first, second, equal_var=True):
        frame = self._require_data()
        statistic, p_value = stats.ttest_ind(frame[first], frame[second], equal_var=equal_var, nan_policy="omit")
        return pd.DataFrame({"Statistic": [statistic], "P-Value": [p_value]}, index=["T-Test"])

    def chisquare(self, first, second):
        frame = self._require_data()
        contingency = pd.crosstab(frame[first], frame[second])
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
        result = pd.DataFrame({"Chi-Square": [chi2], "P-Value": [p_value], "Degrees of Freedom": [dof]}, index=["Chi-Square Test"])
        return result, contingency

    def normality(self, columns):
        frame = self._require_data()[self._resolve_columns(columns)]
        rows = []
        for column in frame.columns:
            values = frame[column].dropna()
            shapiro_result = stats.shapiro(values)
            ks_result = stats.kstest(values, "norm")
            rows.append({
                "Column": column,
                "Shapiro-Wilk Statistic": shapiro_result.statistic,
                "Shapiro-Wilk P-Value": shapiro_result.pvalue,
                "Kolmogorov-Smirnov Statistic": ks_result.statistic,
                "Kolmogorov-Smirnov P-Value": ks_result.pvalue
            })
        return pd.DataFrame(rows).set_index("Column")

    def anova(self, dependent, independent):
        frame = self._require_data()
        groups = [frame[frame[independent] == category][dependent].dropna() for category in frame[independent].dropna().unique()]
        statistic, p_value = stats.f_oneway(*groups)
        return pd.DataFrame({"F-Statistic": [statistic], "P-Value": [p_value]}, index=["ANOVA"])

    def manova(self, dependent_vars, independent):
        frame = self._require_data()
        dependents = self._resolve_columns(dependent_vars)
        formula = " + ".join(dependents) + " ~ " + independent
        model = MANOVA.from_formula(formula, data=frame)
        return pd.DataFrame(model.mv_test().results)

    def efa(self, columns, n_factors=2):
        frame = self._require_data()[self._resolve_columns(columns)].dropna()
        chi_square, p_value = calculate_bartlett_sphericity(frame)
        _, kmo_model = calculate_kmo(frame)
        analyzer = FactorAnalyzer(n_factors=n_factors, rotation="varimax")
        analyzer.fit(frame)
        loadings = pd.DataFrame(analyzer.loadings_, index=frame.columns, columns=[f"Factor {index + 1}" for index in range(n_factors)])
        diagnostics = pd.DataFrame({"Bartlett Chi-Square": [chi_square], "Bartlett P-Value": [p_value], "KMO Overall": [kmo_model]})
        return loadings, diagnostics

    def cfa(self, model_description, columns=None):
        frame = self._require_data() if columns is None else self._require_data()[self._resolve_columns(columns)]
        model = Model(model_description)
        model.fit(frame.dropna())
        return pd.DataFrame(model.inspect())

    def canonical_correlation(self, set1, set2):
        frame = self._require_data()
        first = frame[self._resolve_columns(set1)]
        second = frame[self._resolve_columns(set2)]
        model = CCA()
        model.fit(first, second)
        return pd.DataFrame({"Canonical Correlation Score": [model.score(first, second)]})

    def regression(self, dependent, independent, kind="ols"):
        frame = self._require_data()
        predictors = self._resolve_columns(independent)
        X = sm.add_constant(frame[predictors])
        Y = frame[dependent]
        if kind.lower() == "glm":
            model = sm.GLM(Y, X, family=sm.families.Gaussian()).fit()
        else:
            model = sm.OLS(Y, X).fit()
        return pd.DataFrame(model.summary().tables[1])

    def kmeans(self, columns, n_clusters=3, random_state=42, apply=False, show=False):
        frame = self._require_data()
        selected = self._resolve_columns(columns)
        features = frame[selected].dropna()
        model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels = model.fit_predict(features)
        centers = pd.DataFrame(model.cluster_centers_, columns=selected)
        counts = pd.Series(labels).value_counts().sort_index().rename("Count").to_frame()
        if apply:
            frame.loc[features.index, "Cluster"] = labels
            self.display_data()
        if show:
            figure, axes = plt.subplots(figsize=(8, 5))
            if len(selected) >= 2:
                scatter = axes.scatter(features[selected[0]], features[selected[1]], c=labels, cmap="viridis")
                axes.set_xlabel(selected[0])
                axes.set_ylabel(selected[1])
            else:
                scatter = axes.scatter(features.index, features[selected[0]], c=labels, cmap="viridis")
                axes.set_xlabel("Row ID")
                axes.set_ylabel(selected[0])
            figure.colorbar(scatter, ax=axes)
            axes.set_title("K-Means Clustering")
            figure.tight_layout()
            report = {"Cluster Centers": centers, "Cluster Counts": counts}
            self.show_report_window(report, "K-Means Clustering", fig=figure)
        return centers, labels

    def hierarchical(self, columns, method="ward", show=False):
        frame = self._require_data()[self._resolve_columns(columns)].dropna()
        matrix = linkage(frame, method=method)
        result = pd.DataFrame(matrix, columns=["Cluster 1", "Cluster 2", "Distance", "Sample Count"])
        if show:
            figure, axes = plt.subplots(figsize=(9, 6))
            dendrogram(matrix, labels=[str(index) for index in frame.index], ax=axes)
            axes.set_title("Hierarchical Clustering Dendrogram")
            axes.set_ylabel("Distance")
            figure.tight_layout()
            self.show_report_window({"Linkage Matrix": result}, "Hierarchical Clustering", fig=figure)
        return result

    def dbscan(self, columns, eps=0.5, min_samples=5, apply=False, show=False):
        frame = self._require_data()
        selected = self._resolve_columns(columns)
        features = frame[selected].dropna()
        scaled = StandardScaler().fit_transform(features)
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(scaled)
        summary = features.assign(Cluster=labels).groupby("Cluster")[selected].mean()
        counts = pd.Series(labels).value_counts().sort_index().rename("Count").to_frame()
        if apply:
            frame.loc[features.index, "Cluster"] = labels
            self.display_data()
        if show:
            figure, axes = plt.subplots(figsize=(8, 5))
            if len(selected) >= 2:
                scatter = axes.scatter(features[selected[0]], features[selected[1]], c=labels, cmap="viridis")
                axes.set_xlabel(selected[0])
                axes.set_ylabel(selected[1])
            else:
                scatter = axes.scatter(features.index, features[selected[0]], c=labels, cmap="viridis")
                axes.set_xlabel("Row ID")
                axes.set_ylabel(selected[0])
            figure.colorbar(scatter, ax=axes)
            axes.set_title("DBSCAN Clustering")
            figure.tight_layout()
            report = {"Cluster Centers": summary, "Cluster Counts": counts, "Noise Points": pd.DataFrame({"Value": [int((labels == -1).sum())]}, index=["Noise"])}
            self.show_report_window(report, "DBSCAN Clustering", fig=figure)
        return summary, labels

    def gaussian_mixture(self, columns, n_components=3, random_state=42, apply=False, show=False):
        frame = self._require_data()
        selected = self._resolve_columns(columns)
        features = frame[selected].dropna()
        model = GaussianMixture(n_components=n_components, random_state=random_state, n_init=5)
        labels = model.fit_predict(features)
        means = pd.DataFrame(model.means_, columns=selected)
        weights = pd.DataFrame({"Component": range(n_components), "Weight": model.weights_})
        counts = pd.Series(labels).value_counts().sort_index().rename("Count").to_frame()
        if apply:
            frame.loc[features.index, "Cluster"] = labels
            self.display_data()
        if show:
            figure, axes = plt.subplots(figsize=(8, 5))
            if len(selected) >= 2:
                scatter = axes.scatter(features[selected[0]], features[selected[1]], c=labels, cmap="viridis", alpha=0.7)
                axes.set_xlabel(selected[0])
                axes.set_ylabel(selected[1])
                for index, center in enumerate(model.means_):
                    covariance = model.covariances_[index][:2, :2]
                    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
                    order = eigenvalues.argsort()[::-1]
                    eigenvalues = eigenvalues[order]
                    eigenvectors = eigenvectors[:, order]
                    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
                    ellipse = Ellipse(center[:2], width=4 * np.sqrt(eigenvalues[0]), height=4 * np.sqrt(eigenvalues[1]), angle=angle, fill=False, linewidth=2)
                    axes.add_patch(ellipse)
            else:
                scatter = axes.scatter(features.index, features[selected[0]], c=labels, cmap="viridis", alpha=0.7)
                axes.set_xlabel("Row ID")
                axes.set_ylabel(selected[0])
            figure.colorbar(scatter, ax=axes)
            axes.set_title("Gaussian Mixture")
            figure.tight_layout()
            report = {"Component Means": means, "Component Weights": weights, "Cluster Counts": counts}
            self.show_report_window(report, "Gaussian Mixture", fig=figure)
        return means, labels

    def pca(self, columns, n_components=2, apply=False, show=False):
        frame = self._require_data()
        selected = self._resolve_columns(columns)
        features = frame[selected].dropna()
        component_count = min(int(n_components), len(selected), len(features))
        scaler = StandardScaler()
        scaled = scaler.fit_transform(features)
        model = PCA(n_components=component_count)
        scores = model.fit_transform(scaled)
        score_columns = [f"PC{index + 1}" for index in range(component_count)]
        score_frame = pd.DataFrame(scores, columns=score_columns, index=features.index)
        variance_percent = model.explained_variance_ratio_ * 100
        metrics = pd.DataFrame({"Component": score_columns, "Explained Variance (%)": variance_percent, "Cumulative (%)": np.cumsum(variance_percent)})
        components = pd.DataFrame(model.components_, columns=selected, index=score_columns)
        if apply:
            frame.loc[features.index, score_columns] = score_frame
            self.display_data()
        if show:
            scree_figure, scree_axes = plt.subplots(figsize=(7, 4))
            scree_axes.bar(score_columns, variance_percent, color="steelblue")
            scree_axes.set_title("PCA Scree Plot")
            scree_axes.set_ylabel("Explained Variance (%)")
            scree_figure.tight_layout()
            figures = [scree_figure]
            if component_count >= 2:
                score_figure, score_axes = plt.subplots(figsize=(7, 5))
                score_axes.scatter(score_frame[score_columns[0]], score_frame[score_columns[1]], alpha=0.75)
                score_axes.set_xlabel(score_columns[0])
                score_axes.set_ylabel(score_columns[1])
                score_axes.set_title("PCA Score Plot")
                score_figure.tight_layout()
                figures.append(score_figure)
            self.show_report_window({"Explained Variance": metrics, "Components": components}, "Principal Component Analysis", figures=figures)
        return score_frame, components, metrics

    def stationarity(self, column):
        series = self._require_data()[column].dropna()
        adf_result = adfuller(series)
        kpss_result = kpss(series, regression="c")
        return pd.DataFrame({
            "Test": ["ADF", "KPSS"],
            "Statistic": [adf_result[0], kpss_result[0]],
            "P-Value": [adf_result[1], kpss_result[1]]
        })

    def seasonal_decomposition(self, column, model="additive", period=12):
        series = self._require_data()[column].dropna()
        result = seasonal_decompose(series, model=model, period=period)
        return pd.DataFrame({"Observed": result.observed, "Trend": result.trend, "Seasonal": result.seasonal, "Residual": result.resid})

    def holt_winters(self, column, trend="add", seasonal="add", seasonal_periods=12):
        series = self._require_data()[column].dropna()
        model = ExponentialSmoothing(series, trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods).fit()
        fitted = pd.DataFrame({"Actual": series, "Forecast": model.fittedvalues})
        mse = float(np.mean((series - fitted["Forecast"]) ** 2))
        mae = float(np.mean(np.abs(series - fitted["Forecast"])))
        metrics = pd.DataFrame({"Metric": ["MSE", "MAE", "RMSE"], "Value": [mse, mae, math.sqrt(mse)]})
        return metrics, fitted

    def moving_average(self, column, window=3):
        series = self._require_data()[column].dropna()
        values = pd.DataFrame({"Actual": series, "Moving Average": series.rolling(window=window).mean()})
        valid = values.dropna()
        mse = float(np.mean((valid["Actual"] - valid["Moving Average"]) ** 2))
        mae = float(np.mean(np.abs(valid["Actual"] - valid["Moving Average"])))
        metrics = pd.DataFrame({"Metric": ["MSE", "MAE", "RMSE"], "Value": [mse, mae, math.sqrt(mse)]})
        return values, metrics

    def train_test(self, dependent, independent, test_size=0.2, random_state=42):
        frame = self._require_data()
        features = pd.get_dummies(frame[self._resolve_columns(independent)], drop_first=False).fillna(0)
        target = frame[dependent]
        return train_test_split(features, target, test_size=test_size, random_state=random_state)

    def model_metrics(self, model, X_test, y_test, predictions, title="", X_train=None, y_train=None, include_figures=False):
        payload = self._build_classification_report(
            model, X_train, X_test, y_train if y_train is not None else y_test, y_test, predictions, title,
            include_figures=include_figures
        )
        report_data = payload["report"]
        return {
            "title": title,
            "model": model,
            "accuracy": accuracy_score(y_test, predictions),
            "confusion_matrix": report_data["Confusion Matrix"],
            "classification_report": report_data["Classification Report"],
            "predictions": predictions,
            "test_set": report_data["Test Set Predictions"],
            "report": report_data,
            "figures": payload["figures"]
        }

    def classify(self, model_type, dependent, independent, test_size=0.2, random_state=42, show=False, include_figures=False, **kwargs):
        builders = {
            "logistic": lambda **options: LogisticRegression(**options),
            "logisticregression": lambda **options: LogisticRegression(**options),
            "lr": lambda **options: LogisticRegression(**options),
            "decision_tree": lambda **options: DecisionTreeClassifier(**options),
            "dt": lambda **options: DecisionTreeClassifier(**options),
            "random_forest": lambda **options: RandomForestClassifier(**options),
            "rf": lambda **options: RandomForestClassifier(**options),
            "naive_bayes": lambda **options: GaussianNB(**options),
            "nb": lambda **options: GaussianNB(**options),
            "knn": lambda **options: KNeighborsClassifier(**options),
            "knearestneighbors": lambda **options: KNeighborsClassifier(**options),
            "svm": lambda **options: SVC(**options),
            "svc": lambda **options: SVC(**options),
            "neural_network": lambda **options: MLPClassifier(**{"max_iter": 500, **options}),
            "mlp": lambda **options: MLPClassifier(**{"max_iter": 500, **options})
        }
        key = str(model_type).lower().replace(" ", "_").replace("-", "_")
        if key not in builders:
            raise ValueError(f"Unknown model type '{model_type}'. Use api_help() for available names.")
        X_train, X_test, y_train, y_test = self.train_test(dependent, independent, test_size, random_state)
        builder_options = dict(kwargs)
        if key in ("logistic", "logisticregression", "lr"):
            builder_options.setdefault("max_iter", 500)
        model = builders[key](**builder_options)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        result = self.model_metrics(model, X_test, y_test, predictions, model_type, X_train=X_train, y_train=y_train, include_figures=(include_figures or show))
        if show:
            display_name = model_type.replace("_", " ").title()
            self.show_report_window(result["report"], f"{display_name} - Evaluation", figures=result["figures"])
        return result

    def logistic_regression(self, dependent, independent, **kwargs):
        return self.classify("logistic", dependent, independent, **kwargs)

    def decision_tree(self, dependent, independent, **kwargs):
        return self.classify("decision_tree", dependent, independent, **kwargs)

    def random_forest(self, dependent, independent, **kwargs):
        return self.classify("random_forest", dependent, independent, **kwargs)

    def naive_bayes(self, dependent, independent, **kwargs):
        return self.classify("naive_bayes", dependent, independent, **kwargs)

    def knn(self, dependent, independent, **kwargs):
        return self.classify("knn", dependent, independent, **kwargs)

    def svm(self, dependent, independent, **kwargs):
        return self.classify("svm", dependent, independent, **kwargs)

    def neural_network(self, dependent, independent, **kwargs):
        return self.classify("neural_network", dependent, independent, **kwargs)

    def _plot_type_key(self, plot_type):
        key = str(plot_type or "auto").strip().lower().replace(" ", "_").replace("/", "_").replace("-", "_")
        aliases = {
            "hist": "histogram", "histogram_plot": "histogram", "density": "kde", "density_plot": "kde",
            "boxplot": "box", "violinplot": "violin", "stripplot": "strip", "count": "bar",
            "barchart": "bar", "barplot": "bar", "barh": "horizontal_bar", "horizontal_bar_chart": "horizontal_bar",
            "piechart": "pie", "scatterplot": "scatter", "lineplot": "line", "areaplot": "area",
            "heat_map": "heatmap", "correlation_heatmap": "heatmap", "pair_plot": "pairplot",
            "pareto_chart": "pareto", "time_series_line": "time_series", "moving_average_plot": "moving_average"
        }
        key = aliases.get(key, key)
        if key not in PLOT_TYPES:
            raise ValueError(f"Unknown plot type '{plot_type}'. Use one of: {', '.join(PLOT_TYPES)}")
        return key

    def _optional_plot_column(self, value):
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _plot_column(self, value, required=False, numeric=False):
        frame = self._require_data()
        column = self._optional_plot_column(value)
        if column is None:
            if required:
                raise ValueError("A required variable is missing.")
            return None
        if column not in frame.columns:
            raise ValueError(f"Column not found: {column}")
        if numeric and not pd.api.types.is_numeric_dtype(frame[column]):
            raise ValueError(f"Column '{column}' must be numeric for this plot.")
        return column

    def _is_numeric_plot_column(self, column):
        return column is not None and pd.api.types.is_numeric_dtype(self._require_data()[column])

    def _numeric_plot_columns(self, columns=None):
        frame = self._require_data()
        if columns:
            selected = self._resolve_columns(columns)
        else:
            selected = list(frame.columns)
        numeric = [column for column in selected if pd.api.types.is_numeric_dtype(frame[column])]
        if not numeric:
            raise ValueError("At least one numeric column is required for this plot.")
        return numeric

    def _plot_bool(self, value):
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ("1", "true", "yes", "y", "on")

    def _parse_figsize(self, figsize):
        if figsize is None:
            return 8, 5
        if isinstance(figsize, (list, tuple)) and len(figsize) >= 2:
            return float(figsize[0]), float(figsize[1])
        text = str(figsize).replace("x", ",").replace("*", ",")
        parts = [part for part in re.split(r"[,\s]+", text.strip()) if part]
        if len(parts) < 2:
            return 8, 5
        return float(parts[0]), float(parts[1])

    def _aggregate_plot_frame(self, frame, x, y=None, hue=None, aggregation="mean", top_n=None):
        columns = [column for column in (x, y, hue) if column]
        working = frame.dropna(subset=columns)
        if working.empty:
            raise ValueError("No complete rows are available for the selected variables.")
        operation = str(aggregation or "mean").lower()
        if operation not in ("mean", "sum", "count", "median", "min", "max", "std"):
            operation = "mean"
        if y:
            if hue:
                summary = working.groupby([x, hue])[y].agg(operation).unstack(hue)
            else:
                summary = working.groupby(x)[y].agg(operation).to_frame("Value")
        else:
            if hue:
                summary = working.groupby([x, hue]).size().unstack(hue).fillna(0)
            else:
                summary = working[x].value_counts().rename("Count").to_frame()
        if top_n:
            if summary.shape[1] == 1:
                summary = summary.sort_values(summary.columns[0], ascending=False).head(top_n)
            else:
                summary = summary.head(top_n)
        return summary

    def _finish_plot(self, figure, axes, title, xlabel, ylabel, show, tight=True):
        if axes is not None:
            if title:
                axes.set_title(title)
            if xlabel:
                axes.set_xlabel(xlabel)
            if ylabel:
                axes.set_ylabel(ylabel)
        if tight:
            figure.tight_layout()
        if show:
            self.show_plot_window(figure, title or "Plot")
        return figure, axes

    def visualize(self, plot_type="auto", x=None, y=None, hue=None, size=None, columns=None,
                  title=None, xlabel=None, ylabel=None, bins=20, top_n=10, agg="mean",
                  window=3, palette="viridis", rotation=0, figsize=(8, 5), alpha=0.85,
                  kde=True, normalize=False, show=True):
        frame = self._require_data()
        kind = self._plot_type_key(plot_type)
        x_col = self._plot_column(x)
        y_col = self._plot_column(y)
        hue_col = self._plot_column(hue)
        size_col = self._plot_column(size)
        bins = max(1, int(bins or 20))
        top_n = int(top_n) if top_n else None
        window = max(1, int(window or 3))
        rotation = int(rotation or 0)
        alpha = float(alpha)
        palette = palette or "viridis"
        kde = self._plot_bool(kde)
        normalize = self._plot_bool(normalize)

        if kind == "pairplot":
            selected = self._resolve_columns(columns)
            if len(selected) < 2:
                raise ValueError("Pair Plot requires at least two columns.")
            plot_columns = list(dict.fromkeys(selected + ([hue_col] if hue_col else [])))
            grid = sns.pairplot(frame[plot_columns], vars=selected, hue=hue_col, palette=palette if hue_col else None, corner=True)
            figure = grid.figure
            if title:
                figure.suptitle(title)
            figure.tight_layout(rect=(0, 0, 1, 0.96))
            if show:
                self.show_plot_window(figure, title or "Pair Plot")
            return figure, None

        if kind == "auto":
            if x_col and y_col:
                if self._is_numeric_plot_column(x_col) and self._is_numeric_plot_column(y_col):
                    kind = "scatter"
                elif self._is_numeric_plot_column(y_col):
                    kind = "box"
                elif hue_col:
                    kind = "grouped_bar"
                else:
                    kind = "bar"
            elif x_col:
                kind = "histogram" if self._is_numeric_plot_column(x_col) else "bar"
            elif y_col:
                kind = "histogram"
            else:
                raise ValueError("Select at least one variable for a quick plot.")

        width, height = self._parse_figsize(figsize)
        figure, axes = plt.subplots(figsize=(width, height))
        hue_palette = palette if hue_col else None

        if kind == "histogram":
            column = x_col or y_col
            sns.histplot(data=frame, x=column, hue=hue_col, bins=bins, kde=kde, stat="density" if normalize else "count", palette=hue_palette, alpha=alpha, ax=axes)
        elif kind == "kde":
            column = x_col or y_col
            sns.kdeplot(data=frame, x=column, hue=hue_col, fill=True, palette=hue_palette, ax=axes)
        elif kind in ("box", "violin", "strip"):
            if not y_col:
                if not self._is_numeric_plot_column(x_col):
                    raise ValueError(f"{PLOT_TYPES[kind]} requires a numeric Y variable.")
                plot_kwargs = {"data": frame, "y": x_col, "ax": axes}
            else:
                plot_kwargs = {"data": frame, "x": x_col, "y": y_col, "hue": hue_col, "palette": hue_palette, "ax": axes}
            if kind == "box":
                sns.boxplot(**plot_kwargs)
            elif kind == "violin":
                sns.violinplot(**plot_kwargs)
            else:
                sns.stripplot(**plot_kwargs, alpha=alpha)
        elif kind in ("bar", "horizontal_bar"):
            if not x_col:
                raise ValueError("Select a category column for the bar chart.")
            summary = self._aggregate_plot_frame(frame, x_col, y_col, hue_col, agg, top_n)
            chart_kind = "barh" if kind == "horizontal_bar" else "bar"
            if summary.shape[1] == 1:
                summary.iloc[:, 0].plot(kind=chart_kind, ax=axes, color="steelblue", alpha=alpha, rot=rotation if chart_kind == "bar" else 0)
            else:
                summary.plot(kind=chart_kind, ax=axes, stacked=False, alpha=alpha, rot=rotation if chart_kind == "bar" else 0)
        elif kind == "pie":
            if not x_col:
                raise ValueError("Select a label column for the pie chart.")
            summary = self._aggregate_plot_frame(frame, x_col, y_col, None, "sum", top_n)
            values = summary.iloc[:, 0].dropna()
            if values.empty:
                raise ValueError("No values are available for the pie chart.")
            axes.pie(values, labels=[str(label) for label in values.index], autopct="%1.1f%%", startangle=90)
            axes.set_ylabel("")
        elif kind in ("grouped_bar", "stacked_bar"):
            if not x_col or not y_col or not hue_col:
                raise ValueError(f"{PLOT_TYPES[kind]} requires X, Y, and Group variables.")
            table = frame.dropna(subset=[x_col, y_col, hue_col]).groupby([x_col, hue_col])[y_col].agg(agg).unstack(hue_col)
            if top_n:
                table = table.head(top_n)
            table.plot(kind="bar", stacked=(kind == "stacked_bar"), ax=axes, alpha=alpha, rot=rotation)
        elif kind in ("scatter", "bubble"):
            if not x_col or not y_col:
                raise ValueError("Scatter and bubble charts require X and Y variables.")
            if kind == "bubble" and not size_col:
                raise ValueError("Bubble charts require a Size variable.")
            sns.scatterplot(data=frame, x=x_col, y=y_col, hue=hue_col, size=size_col, sizes=(80, 700) if size_col else None, palette=hue_palette, alpha=alpha, ax=axes)
        elif kind == "line":
            if not y_col:
                raise ValueError("Line plots require a Y variable.")
            if x_col:
                working = frame.sort_values(x_col)
                x_arg = x_col
            else:
                working = frame.reset_index()
                x_arg = working.columns[0]
            sns.lineplot(data=working, x=x_arg, y=y_col, hue=hue_col, palette=hue_palette, ax=axes)
        elif kind == "area":
            if not x_col or not y_col:
                raise ValueError("Area plots require X and Y variables.")
            series = frame.groupby(x_col)[y_col].agg(agg)
            if top_n:
                series = series.head(top_n)
            series.plot(kind="area", ax=axes, alpha=alpha, rot=rotation)
        elif kind == "heatmap":
            selected = self._numeric_plot_columns(columns)
            correlation = frame[selected].corr(numeric_only=True)
            sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, ax=axes)
        elif kind == "pareto":
            if not x_col:
                raise ValueError("Pareto charts require a category column.")
            summary = self._aggregate_plot_frame(frame, x_col, y_col, None, "sum" if y_col else "count", top_n)
            values = summary.iloc[:, 0].dropna().sort_values(ascending=False)
            if values.empty:
                raise ValueError("No values are available for the Pareto chart.")
            values.plot(kind="bar", ax=axes, color="steelblue", rot=rotation)
            cumulative = values.cumsum() / values.sum() * 100
            secondary_axes = axes.twinx()
            secondary_axes.plot(range(len(values)), cumulative.values, color="darkorange", marker="o")
            secondary_axes.set_ylabel("Cumulative %")
            secondary_axes.set_ylim(0, 105)
        elif kind == "time_series":
            if not y_col:
                raise ValueError("Time series plots require a value column.")
            if x_col:
                working = frame.sort_values(x_col)
                x_values = working[x_col]
            else:
                working = frame
                x_values = working.index
            axes.plot(x_values, working[y_col], marker="o", label=y_col)
            if hue_col:
                for category in working[hue_col].dropna().unique():
                    subset = working[working[hue_col] == category]
                    subset_x = subset[x_col] if x_col else subset.index
                    axes.plot(subset_x, subset[y_col], marker="o", label=str(category))
            axes.legend()
        elif kind == "moving_average":
            if not y_col:
                raise ValueError("Moving average plots require a value column.")
            if x_col:
                working = frame.sort_values(x_col)
                x_values = working[x_col]
            else:
                working = frame
                x_values = working.index
            axes.plot(x_values, working[y_col], label="Actual")
            axes.plot(x_values, working[y_col].rolling(window=window).mean(), label=f"{window}-period Moving Average", linestyle="dashed")
            axes.legend()
        else:
            raise ValueError(f"Plot type '{kind}' is not implemented.")

        title = title or f"{PLOT_TYPES.get(kind, kind)}" + (f" - {x_col}" if x_col else "") + (f" vs {y_col}" if y_col else "")
        return self._finish_plot(figure, axes, title, xlabel or x_col, ylabel or y_col, show)

    def plot(self, columns, kind=None, show=True):
        selected = self._resolve_columns(columns)
        options = {"show": show}
        if selected:
            options["x"] = selected[0]
        if len(selected) > 1:
            options["y"] = selected[1]
        if len(selected) > 2:
            options["hue"] = selected[2]
        plot_kind = self._plot_type_key(kind) if kind else "auto"
        if plot_kind in ("heatmap", "pairplot"):
            options["columns"] = selected
            options.pop("x", None)
            options.pop("y", None)
            options.pop("hue", None)
        return self.visualize(plot_kind, **options)

    def histogram(self, column, **kwargs):
        kwargs.setdefault("x", column)
        return self.visualize("histogram", **kwargs)

    def kde_plot(self, column, **kwargs):
        kwargs.setdefault("x", column)
        return self.visualize("kde", **kwargs)

    def bar_chart(self, x_value, y_value=None, **kwargs):
        kwargs.setdefault("x", x_value)
        kwargs.setdefault("y", y_value)
        return self.visualize("bar", **kwargs)

    def horizontal_bar(self, x_value, y_value=None, **kwargs):
        kwargs.setdefault("x", x_value)
        kwargs.setdefault("y", y_value)
        return self.visualize("horizontal_bar", **kwargs)

    def pie_chart(self, labels, values=None, **kwargs):
        kwargs.setdefault("x", labels)
        kwargs.setdefault("y", values)
        return self.visualize("pie", **kwargs)

    def scatter_plot(self, x_value, y_value, **kwargs):
        kwargs.setdefault("x", x_value)
        kwargs.setdefault("y", y_value)
        return self.visualize("scatter", **kwargs)

    def bubble_chart(self, x_value, y_value, size_value, **kwargs):
        kwargs.setdefault("x", x_value)
        kwargs.setdefault("y", y_value)
        kwargs.setdefault("size", size_value)
        return self.visualize("bubble", **kwargs)

    def line_plot(self, y_value, x_value=None, **kwargs):
        kwargs.setdefault("x", x_value)
        kwargs.setdefault("y", y_value)
        return self.visualize("line", **kwargs)

    def box_plot(self, x_value, y_value=None, **kwargs):
        kwargs.setdefault("x", x_value)
        kwargs.setdefault("y", y_value)
        return self.visualize("box", **kwargs)

    def violin_plot(self, x_value, y_value=None, **kwargs):
        kwargs.setdefault("x", x_value)
        kwargs.setdefault("y", y_value)
        return self.visualize("violin", **kwargs)

    def strip_plot(self, x_value, y_value=None, **kwargs):
        kwargs.setdefault("x", x_value)
        kwargs.setdefault("y", y_value)
        return self.visualize("strip", **kwargs)

    def area_plot(self, x_value, y_value, **kwargs):
        kwargs.setdefault("x", x_value)
        kwargs.setdefault("y", y_value)
        return self.visualize("area", **kwargs)

    def grouped_bar(self, x_value, y_value, group_value, **kwargs):
        kwargs.setdefault("x", x_value)
        kwargs.setdefault("y", y_value)
        kwargs.setdefault("hue", group_value)
        return self.visualize("grouped_bar", **kwargs)

    def stacked_bar(self, x_value, y_value, group_value, **kwargs):
        kwargs.setdefault("x", x_value)
        kwargs.setdefault("y", y_value)
        kwargs.setdefault("hue", group_value)
        return self.visualize("stacked_bar", **kwargs)

    def heatmap(self, columns=None, **kwargs):
        return self.visualize("heatmap", columns=columns, **kwargs)

    def pairplot(self, columns=None, **kwargs):
        return self.visualize("pairplot", columns=columns, **kwargs)

    def pareto(self, x_value, y_value=None, **kwargs):
        kwargs.setdefault("x", x_value)
        kwargs.setdefault("y", y_value)
        return self.visualize("pareto", **kwargs)

    def time_series(self, y_value, x_value=None, **kwargs):
        kwargs.setdefault("x", x_value)
        kwargs.setdefault("y", y_value)
        return self.visualize("time_series", **kwargs)

    def moving_average_plot(self, y_value, x_value=None, **kwargs):
        kwargs.setdefault("x", x_value)
        kwargs.setdefault("y", y_value)
        return self.visualize("moving_average", **kwargs)

    def show(self, value, title="Result"):
        if isinstance(value, pd.Series):
            value = value.to_frame("Value")
        elif isinstance(value, (list, tuple, dict)) and not isinstance(value, dict):
            value = pd.DataFrame({"Value": list(value)})
        elif not isinstance(value, (pd.DataFrame, dict)):
            value = pd.DataFrame({"Value": [value]})
        self.show_report_window(value, title)
        return value

    def show_plot(self, figure, title="Plot"):
        self.show_plot_window(figure, title)
        return figure

    def _domain_params(self, **values):
        return {key: value for key, value in values.items() if value is not None}

    def generate_domain_demo(self, domain, preset, rows=200, seed=42, apply=True, show=True, confirm=True):
        frame = domain_analytics.simulate(domain, preset, rows=rows, seed=seed)
        if apply:
            if confirm and self.data is not None and not messagebox.askyesno("Replace Data", "Generating domain demo data will replace the current data grid. Continue?"):
                return frame
            self.set_df(frame)
            if show:
                label = str(preset).replace("_", " ").title()
                messagebox.showinfo("Success", f"{label} demo generated with {len(frame)} rows.")
        return frame

    def perform_domain_analysis(self, domain, technique):
        domain_key = str(domain).strip().lower()
        technique_key = str(technique).strip().lower().replace(" ", "_").replace("-", "_")
        title = domain_analytics.ANALYSIS_TITLES.get((domain_key, technique_key), f"{domain_key} {technique_key}".title())
        fields = []
        for key, label, default in domain_analytics.parameter_fields(domain_key, technique_key, self.data):
            fields.append((key, label, tk.StringVar(value=default)))
        values = self._ask_field_dialog(title, fields)
        if not values:
            return None
        try:
            return self.domain_analysis(domain_key, technique_key, frame=self.data, show=True, **values)
        except Exception as error:
            messagebox.showerror("Analysis Failed", str(error))
            return None

    def domain_analysis(self, domain, technique, frame=None, show=False, **params):
        result = domain_analytics.analyze(domain, technique, frame if frame is not None else self._require_data(), **params)
        if show:
            self.show_report_window(result["report"], f"{result['title']} - Report", figures=result["figures"])
        return result

    def finance_demo(self, preset="credit_risk", rows=200, seed=42, apply=False):
        return self.generate_domain_demo("finance", preset, rows=rows, seed=seed, apply=apply, show=False, confirm=False)

    def hrm_demo(self, preset="attrition", rows=200, seed=42, apply=False):
        return self.generate_domain_demo("hrm", preset, rows=rows, seed=seed, apply=apply, show=False, confirm=False)

    def credit_risk(self, target=None, features=None, model="gradient_boosting", test_size=0.25, seed=42, show=False, frame=None):
        params = self._domain_params(target=target, features=features, model=model, test_size=test_size, seed=seed)
        return self.domain_analysis("finance", "credit_risk", frame=frame, show=show, **params)

    def fraud_detection(self, features=None, target=None, method="isolation_forest", contamination=0.08, top_n=20, seed=42, show=False, frame=None):
        params = self._domain_params(features=features, target=target, model=method, contamination=contamination, top_n=top_n, seed=seed)
        return self.domain_analysis("finance", "fraud_detection", frame=frame, show=show, **params)

    def portfolio_risk(self, returns=None, confidence=0.95, weights="", show=False, frame=None):
        params = self._domain_params(returns=returns, confidence=confidence, weights=weights)
        return self.domain_analysis("finance", "portfolio_risk", frame=frame, show=show, **params)

    def attrition(self, target=None, features=None, model="gradient_boosting", test_size=0.25, seed=42, show=False, frame=None):
        params = self._domain_params(target=target, features=features, model=model, test_size=test_size, seed=seed)
        return self.domain_analysis("hrm", "attrition", frame=frame, show=show, **params)

    def pay_equity(self, salary=None, group=None, role=None, performance=None, show=False, frame=None):
        params = self._domain_params(salary=salary, group=group, role=role, performance=performance)
        return self.domain_analysis("hrm", "pay_equity", frame=frame, show=show, **params)

    def employee_segmentation(self, features=None, clusters=3, seed=42, show=False, frame=None):
        params = self._domain_params(features=features, clusters=clusters, seed=seed)
        return self.domain_analysis("hrm", "employee_segmentation", frame=frame, show=show, **params)

    def it_operations_demo(self, preset="incidents", rows=200, seed=42, apply=False):
        return self.generate_domain_demo("it_operations", preset, rows=rows, seed=seed, apply=apply, show=False, confirm=False)

    def international_demo(self, preset="trade", rows=200, seed=42, apply=False):
        return self.generate_domain_demo("international", preset, rows=rows, seed=seed, apply=apply, show=False, confirm=False)

    def fair_lending(self, decision=None, group=None, features=None, target=None, show=False, frame=None):
        params = self._domain_params(decision=decision, group=group, features=features, target=target)
        return self.domain_analysis("finance", "fair_lending", frame=frame, show=show, **params)

    def demand_forecast(self, value=None, period=None, periods=6, model="holt_winters", seasonal_periods=12, show=False, frame=None):
        params = self._domain_params(value=value, period=period, periods=periods, model=model, seasonal_periods=seasonal_periods)
        return self.domain_analysis("finance", "demand_forecast", frame=frame, show=show, **params)

    def workforce_forecast(self, value=None, period=None, periods=6, model="holt_winters", seasonal_periods=12, show=False, frame=None):
        params = self._domain_params(value=value, period=period, periods=periods, model=model, seasonal_periods=seasonal_periods)
        return self.domain_analysis("hrm", "workforce_forecast", frame=frame, show=show, **params)

    def predictive_maintenance(self, features=None, target=None, model="isolation_forest", contamination=0.08, top_n=20, seed=42, show=False, frame=None):
        params = self._domain_params(features=features, target=target, model=model, contamination=contamination, top_n=top_n, seed=seed)
        return self.domain_analysis("it_operations", "predictive_maintenance", frame=frame, show=show, **params)

    def incident_priority(self, target=None, features=None, model="gradient_boosting", test_size=0.25, seed=42, show=False, frame=None):
        params = self._domain_params(target=target, features=features, model=model, test_size=test_size, seed=seed)
        return self.domain_analysis("it_operations", "incident_priority", frame=frame, show=show, **params)

    def sla_breach(self, target=None, features=None, model="gradient_boosting", test_size=0.25, seed=42, show=False, frame=None):
        params = self._domain_params(target=target, features=features, model=model, test_size=test_size, seed=seed)
        return self.domain_analysis("it_operations", "sla_breach", frame=frame, show=show, **params)

    def capacity_forecast(self, value=None, period=None, periods=6, model="holt_winters", seasonal_periods=12, show=False, frame=None):
        params = self._domain_params(value=value, period=period, periods=periods, model=model, seasonal_periods=seasonal_periods)
        return self.domain_analysis("it_operations", "capacity_forecast", frame=frame, show=show, **params)

    def inventory_reorder(self, demand=None, lead_time=7, service_level=0.95, scenarios=1000, seed=42, show=False, frame=None):
        params = self._domain_params(demand=demand, lead_time=lead_time, service_level=service_level, scenarios=scenarios, seed=seed)
        return self.domain_analysis("it_operations", "inventory_reorder", frame=frame, show=show, **params)

    def supplier_risk(self, features=None, group=None, clusters=3, seed=42, show=False, frame=None):
        params = self._domain_params(features=features, group=group, clusters=clusters, seed=seed)
        return self.domain_analysis("it_operations", "supplier_risk", frame=frame, show=show, **params)

    def process_cycle(self, case=None, activity=None, timestamp=None, top_n=10, show=False, frame=None):
        params = self._domain_params(case=case, activity=activity, timestamp=timestamp, top_n=top_n)
        return self.domain_analysis("it_operations", "process_cycle", frame=frame, show=show, **params)

    def country_segmentation(self, features=None, clusters=3, seed=42, show=False, frame=None):
        params = self._domain_params(features=features, clusters=clusters, seed=seed)
        return self.domain_analysis("international", "country_segmentation", frame=frame, show=show, **params)

    def export_forecast(self, value=None, period=None, filter_column=None, filter_value=None, periods=6, model="holt_winters", seasonal_periods=12, show=False, frame=None):
        params = self._domain_params(value=value, period=period, filter_column=filter_column, filter_value=filter_value, periods=periods, model=model, seasonal_periods=seasonal_periods)
        return self.domain_analysis("international", "export_forecast", frame=frame, show=show, **params)

    def fx_risk(self, returns=None, confidence=0.95, shocks=None, weights=None, show=False, frame=None):
        params = self._domain_params(returns=returns, confidence=confidence, shocks=shocks, weights=weights)
        return self.domain_analysis("international", "fx_risk", frame=frame, show=show, **params)

    def trade_anomaly(self, features=None, target=None, model="isolation_forest", contamination=0.08, top_n=20, seed=42, show=False, frame=None):
        params = self._domain_params(features=features, target=target, model=model, contamination=contamination, top_n=top_n, seed=seed)
        return self.domain_analysis("international", "trade_anomaly", frame=frame, show=show, **params)

    def gravity_model(self, flow=None, exporter_gdp=None, importer_gdp=None, distance=None, dummy=None, show=False, frame=None):
        params = self._domain_params(flow=flow, exporter_gdp=exporter_gdp, importer_gdp=importer_gdp, distance=distance, dummy=dummy)
        return self.domain_analysis("international", "gravity_model", frame=frame, show=show, **params)

    def tariff_impact(self, price=None, demand=None, elasticity=None, tariff=0.10, scenarios=500, seed=42, show=False, frame=None):
        params = self._domain_params(price=price, demand=demand, elasticity=elasticity, tariff=tariff, scenarios=scenarios, seed=seed)
        return self.domain_analysis("international", "tariff_impact", frame=frame, show=show, **params)

    def console_help(self):
        return textwrap.dedent("""
            SADSA Python Console API

            Console variables:
              app                 this application object
              df                  DataFrame currently shown in the data preview
              pd, np, plt, sns    pandas, numpy, matplotlib.pyplot, seaborn
              stats, sm           scipy.stats, statsmodels.api
              os, sys, Path       filesystem and Python runtime helpers
              api_help()          print this help

            Data:
              app.load_csv(path)            app.save_csv(path)
              app.set_df(frame)             app.get_df()
              app.insert_row(position, count=1, values=None)
              app.insert_column(position, name=None, values=None)
              app.edit_cell(row, column, value)
              app.rename_column(column, new_name)
              app.delete_row_at(index)      app.delete_column(column)
              app.copy_column_names()
              app.head(n)                   app.tail(n)
              app.describe()                app.info_text()
              app.missing()                 app.value_counts(column)
              app.groupby_agg(by, agg)      app.pivot(index, columns, values)
              app.correlation(columns)
              app.compute_column(name, expr)  -> creates or overwrites name
              app.evaluate_expression(expr)   -> previews computed values
              app.recode_column(column, mapping)
              app.treat_missing(column, method)
              app.treat_missing_advanced(column, method, predictors)
              app.simulate_missing_data(columns, rate, mechanism, predictor)
              app.test_missing_significance(columns, group_column, alpha)
              app.missingness_report()
              app.standardize_column(column, method)
              app.decompose_columns(columns, method)
              app.simulate(specs, rows, seed)
              app.simulate_multivariate_normal(names, means, covariance, samples)
              app.generate_ml_demo('svm', rows=200, apply=True)
              app.open_ml_simulations()

            Demo data:
              Transactional simulations:
                app.generate_basket(...)
                app.generate_long_transactions(...)
                app.generate_sequence_log(...)
                app.generate_profit_basket(...)
              Customer and campaign simulations:
                app.generate_marketing_log(...)

            Marketing analytics:
              Transactional analysis:
                app.transaction_matrix(...)
                app.frequent_itemsets(min_support=0.1, algorithm='Apriori')
                app.closed_itemsets(...)
                app.maximal_itemsets(...)
                app.association_rules(min_confidence=0.6)
                app.market_basket(...)
                app.sequential_patterns(min_support=0.2)
              Customer analytics:
                app.rfm()
                app.customer_lifetime_value(retention=0.95, horizon=12)
                app.churn_prediction(features=['Age', 'TenureMonths'])
              Campaign and pricing analytics:
                app.campaign_response(segment_column='CampaignChannel')
                app.price_elasticity(price_column='Price', quantity_column='Quantity')
                app.profit_weighted_rules()
                app.recommendations()

            Statistics:
              app.frequencies(columns)      app.summary_stats(columns)
              app.dataset_properties(show=True)
              app.ttest(col1, col2)         app.chisquare(col1, col2)
              app.normality(columns)        app.anova(dependent, independent)
              app.manova(dependents, independent)
              app.efa(columns)              app.cfa(model_description)
              app.canonical_correlation(set1, set2)
              app.regression(dependent, independent, kind='ols')

            Clustering and time series:
              app.kmeans(columns, n_clusters, show=True)
              app.hierarchical(columns, show=True)
              app.dbscan(columns, eps=0.5, min_samples=5, show=True)
              app.gaussian_mixture(columns, n_components=3, show=True)
              app.pca(columns, n_components=2, show=True)
              app.stationarity(column)
              app.seasonal_decomposition(column, period=12)
              app.holt_winters(column)
              app.moving_average(column, window)
              app.simulate_time_series(preset='holt_winters', rows=96)
              app.open_time_series_simulation('seasonal')

            Machine learning:
              app.classify('random_forest', 'target', ['x1', 'x2'], show=True)
              app.logistic_regression(...)
              app.decision_tree(...)
              app.random_forest(...)
              app.naive_bayes(...)
              app.knn(...)
              app.svm(...)
              app.neural_network(...)
              result = app.classify('decision_tree', 'target', ['x1', 'x2'])
              result['test_set'], result['confusion_matrix'], result['figures']

            Business domain analytics:
              app.finance_demo('credit_risk', rows=200, apply=True)
              app.hrm_demo('attrition', rows=200, apply=True)
              app.it_operations_demo('incidents', apply=True)
              app.international_demo('trade', apply=True)
              app.credit_risk(show=True)
              app.fraud_detection(method='local_outlier', show=True)
              app.portfolio_risk(confidence=0.99, show=True)
              app.fair_lending(show=True)
              app.demand_forecast(periods=12, show=True)
              app.attrition(model='logistic', show=True)
              app.pay_equity(show=True)
              app.employee_segmentation(clusters=4, show=True)
              app.workforce_forecast(show=True)
              app.predictive_maintenance(show=True)
              app.incident_priority(show=True)
              app.sla_breach(show=True)
              app.capacity_forecast(show=True)
              app.inventory_reorder(service_level=0.99, show=True)
              app.supplier_risk(clusters=4, show=True)
              app.process_cycle(show=True)
              app.country_segmentation(show=True)
              app.export_forecast(filter_value='USA', show=True)
              app.fx_risk(shocks='-15, -10, -5, 5, 10', show=True)
              app.trade_anomaly(show=True)
              app.gravity_model(show=True)
              app.tariff_impact(tariff=0.15, show=True)
              result = app.domain_analysis('finance', 'credit_risk')
              result['report'], result['figures']

            Output and plots:
              app.show(frame, title)
              app.plot(columns, kind='scatter')
              app.visualize(plot_type, x=..., y=..., hue=..., size=...)
              app.histogram(column, bins=20)
              app.kde_plot(column)
              app.bar_chart(x, y, agg='sum')
              app.horizontal_bar(x, y)
              app.pie_chart(labels, values)
              app.grouped_bar(x, y, group)
              app.stacked_bar(x, y, group)
              app.scatter_plot(x, y, hue=group)
              app.bubble_chart(x, y, size)
              app.line_plot(y, x)
              app.box_plot(x, y)
              app.violin_plot(x, y)
              app.strip_plot(x, y)
              app.area_plot(x, y)
              app.heatmap(columns)
              app.pairplot(columns, hue=group)
              app.pareto(x, y)
              app.time_series(y, x)
              app.moving_average_plot(y, x, window=3)
              app.show_plot(figure, title)

            Shortcuts:
              F5 run script, Ctrl+Enter run selection/current line,
              Ctrl+N new script, Ctrl+O open script, Ctrl+S save,
              Ctrl+plus / Ctrl+minus zoom.
        """).strip()

    def help_api(self):
        print(self.console_help())


class ConsoleOutput:
    def __init__(self, console, tag="output"):
        self.console = console
        self.tag = tag

    def write(self, text):
        if text:
            self.console.write_output(text, self.tag)

    def flush(self):
        pass


class PythonConsoleWindow(tk.Toplevel):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        self.title("SADSA Python Console")
        self.geometry("1120x740")
        self.minsize(820, 520)
        self.font_size = 11
        self.history = []
        self.history_index = 0
        self.pending_lines = []
        self.script_path = None
        self.script_dirty = False
        self.namespace = {
            "app": app,
            "df": app.data,
            "pd": pd,
            "np": np,
            "plt": plt,
            "sns": sns,
            "os": os,
            "sys": sys,
            "math": math,
            "statistics": statistics,
            "random": random,
            "json": json,
            "re": re,
            "io": io,
            "textwrap": textwrap,
            "stats": stats,
            "sm": sm,
            "Path": Path,
            "datetime": dt,
            "datetime_cls": datetime,
            "timedelta": timedelta,
            "api_help": app.console_help,
            "help_api": app.help_api
        }
        self._build_menu()
        self._build_interface()
        self._apply_fonts()
        self._write_banner()
        self.protocol("WM_DELETE_WINDOW", self.close)

    def _build_menu(self):
        menu_bar = tk.Menu(self)
        self.config(menu=menu_bar)
        file_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New Script", accelerator="Ctrl+N", command=self.new_script)
        file_menu.add_command(label="Open Script...", accelerator="Ctrl+O", command=self.open_script)
        file_menu.add_command(label="Save Script", accelerator="Ctrl+S", command=self.save_script)
        file_menu.add_command(label="Save Script As...", accelerator="Ctrl+Shift+S", command=self.save_script_as)
        file_menu.add_separator()
        file_menu.add_command(label="Run Script", accelerator="F5", command=self.run_script)
        file_menu.add_separator()
        file_menu.add_command(label="Close", command=self.close)
        edit_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Clear Console", command=self.clear_console)
        edit_menu.add_command(label="Clear Script", command=self.clear_script)
        edit_menu.add_separator()
        edit_menu.add_command(label="Cut", command=lambda: self._editor_event("<<Cut>>"))
        edit_menu.add_command(label="Copy", command=lambda: self._editor_event("<<Copy>>"))
        edit_menu.add_command(label="Paste", command=lambda: self._editor_event("<<Paste>>"))
        edit_menu.add_command(label="Select All", command=lambda: self._editor_event("<<SelectAll>>"))
        view_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Zoom In", accelerator="Ctrl+plus", command=self.zoom_in)
        view_menu.add_command(label="Zoom Out", accelerator="Ctrl+minus", command=self.zoom_out)
        view_menu.add_command(label="Reset Zoom", command=self.reset_zoom)
        help_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Python Console Help", command=self.help_api)
        self.bind("<Control-n>", lambda event: self.new_script())
        self.bind("<Control-o>", lambda event: self.open_script())
        self.bind("<Control-s>", lambda event: self.save_script())
        self.bind("<Control-S>", lambda event: self.save_script_as())
        self.bind("<F5>", lambda event: self.run_script())
        self.bind("<Control-Return>", lambda event: self.run_selection())
        self.bind("<Control-plus>", lambda event: self.zoom_in())
        self.bind("<Control-equal>", lambda event: self.zoom_in())
        self.bind("<Control-minus>", lambda event: self.zoom_out())

    def _build_interface(self):
        main_frame = ttk.Frame(self)
        main_frame.pack(fill="both", expand=True)
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill="both", expand=True)
        self.status_var = tk.StringVar(value="Ready")
        console_tab = ttk.Frame(self.notebook)
        editor_tab = ttk.Frame(self.notebook)
        self.notebook.add(console_tab, text="Console")
        self.notebook.add(editor_tab, text="Script Editor")
        self._build_console_tab(console_tab)
        self._build_editor_tab(editor_tab)
        ttk.Label(self, textvariable=self.status_var, anchor="w").pack(side="bottom", fill="x")

    def _build_console_tab(self, parent):
        output_frame = ttk.Frame(parent)
        output_frame.pack(fill="both", expand=True)
        output_frame.columnconfigure(0, weight=1)
        output_frame.rowconfigure(0, weight=1)
        self.output_scrollbar_y = ttk.Scrollbar(output_frame, orient="vertical")
        self.output_scrollbar_x = ttk.Scrollbar(output_frame, orient="horizontal")
        self.output_text = tk.Text(output_frame, wrap="none", state="disabled", bg="#101216", fg="#e8e8e8", insertbackground="#ffffff")
        self.output_scrollbar_y.configure(command=self.output_text.yview)
        self.output_scrollbar_x.configure(command=self.output_text.xview)
        self.output_text.configure(yscrollcommand=self.output_scrollbar_y.set, xscrollcommand=self.output_scrollbar_x.set)
        self.output_text.grid(row=0, column=0, sticky="nsew")
        self.output_scrollbar_y.grid(row=0, column=1, sticky="ns")
        self.output_scrollbar_x.grid(row=1, column=0, sticky="ew")
        self.output_text.tag_configure("prompt", foreground="#7bd88f")
        self.output_text.tag_configure("input", foreground="#e8e8e8")
        self.output_text.tag_configure("output", foreground="#d7e6ff")
        self.output_text.tag_configure("error", foreground="#ff8a8a")
        self.output_text.tag_configure("info", foreground="#ffd479")
        input_frame = ttk.Frame(parent)
        input_frame.pack(fill="x", padx=6, pady=6)
        self.prompt_label = tk.Label(input_frame, text=">>>", foreground="#7bd88f")
        self.prompt_label.pack(side="left")
        self.input_entry = tk.Entry(input_frame, bg="#1b1f27", fg="#ffffff", insertbackground="#ffffff")
        self.input_entry.pack(side="left", fill="x", expand=True, padx=6)
        self.input_entry.bind("<Return>", self.execute_input)
        self.input_entry.bind("<Up>", self.history_up)
        self.input_entry.bind("<Down>", self.history_down)
        tk.Button(input_frame, text="Run", command=self.execute_input).pack(side="left", padx=(0, 4))
        tk.Button(input_frame, text="Clear", command=self.clear_console).pack(side="left")

    def _build_editor_tab(self, parent):
        toolbar = ttk.Frame(parent)
        toolbar.pack(fill="x", padx=4, pady=4)
        for label, command in (
            ("New", self.new_script),
            ("Open", self.open_script),
            ("Save", self.save_script),
            ("Save As", self.save_script_as),
            ("Run F5", self.run_script),
            ("Run Selection", self.run_selection),
            ("Zoom -", self.zoom_out),
            ("Zoom +", self.zoom_in),
            ("Reset Zoom", self.reset_zoom)
        ):
            tk.Button(toolbar, text=label, command=command).pack(side="left", padx=2)
        editor_body = ttk.Frame(parent)
        editor_body.pack(fill="both", expand=True)
        self.line_numbers = tk.Text(editor_body, width=5, state="disabled", bg="#f0f0f0", fg="#666666", relief="flat")
        self.line_numbers.pack(side="left", fill="y")
        self.script_scrollbar = ttk.Scrollbar(editor_body)
        self.script_scrollbar.pack(side="right", fill="y")
        self.script_text = tk.Text(editor_body, wrap="none", undo=True, bg="#1b1f27", fg="#e8e8e8", insertbackground="#ffffff")
        self.script_text.pack(side="left", fill="both", expand=True)
        self.script_text.configure(yscrollcommand=self._sync_editor_scroll)
        self.script_scrollbar.configure(command=self._scroll_editor)
        self.script_text.bind("<KeyPress>", self._mark_dirty)
        self.script_text.bind("<KeyRelease>", self._update_line_numbers)
        self.script_text.bind("<Button-1>", self._update_line_numbers)
        self.script_text.bind("<MouseWheel>", self._update_line_numbers)
        self._update_line_numbers()

    def _write_banner(self):
        self.write_output("SADSA Python Console\n", "info")
        self.write_output("Ready variables: app, df, pd, np, plt, sns, stats, sm, os, Path\n")
        self.write_output("Examples:\n")
        self.write_output("  df.head()\n")
        self.write_output("  app.frequent_itemsets(min_support=0.05)\n")
        self.write_output("  app.association_rules(min_confidence=0.6)\n")
        self.write_output("  app.rfm()\n")
        self.write_output("  app.customer_lifetime_value()\n")
        self.write_output("  app.plot(['gender', 'education'])\n")
        self.write_output("  api_help()\n\n")

    def set_df(self, frame):
        self.namespace["df"] = frame

    def write_output(self, text, tag="output"):
        self.output_text.configure(state="normal")
        self.output_text.insert(tk.END, text, tag)
        self.output_text.configure(state="disabled")
        self.output_text.see(tk.END)

    def _write_prompt(self, prompt, text):
        self.write_output(f"{prompt} ", "prompt")
        self.write_output(text + "\n", "input")

    def execute_input(self, event=None):
        raw = self.input_entry.get()
        if not raw.strip():
            return
        self.history.append(raw)
        self.history_index = len(self.history)
        if self.pending_lines:
            self.pending_lines.append(raw)
            source = "\n".join(self.pending_lines)
            self._write_prompt("...", raw)
        else:
            self.pending_lines = [raw]
            source = raw
            self._write_prompt(">>>", raw)
        self.input_entry.delete(0, tk.END)
        try:
            compiled = codeop.compile_command(source, "<console>", "exec")
        except SyntaxError:
            self.pending_lines = []
            self.prompt_label.configure(text=">>>")
            self._execute_code(source)
            return
        if compiled is None:
            self.prompt_label.configure(text="... ")
            return
        self.pending_lines = []
        self.prompt_label.configure(text=">>>")
        self._execute_code(source)

    def _execute_code(self, source, name="<console>"):
        self.status_var.set("Running...")
        self.update_idletasks()
        try:
            tree = ast.parse(source, filename=name)
            single_expression = len(tree.body) == 1 and isinstance(tree.body[0], ast.Expr)
            with contextlib.redirect_stdout(ConsoleOutput(self, "output")), contextlib.redirect_stderr(ConsoleOutput(self, "error")):
                if single_expression:
                    expression = ast.Expression(body=tree.body[0].value)
                    ast.fix_missing_locations(expression)
                    value = eval(compile(expression, name, "eval"), self.namespace)
                    if value is not None:
                        print(repr(value))
                else:
                    exec(compile(tree, name, "exec"), self.namespace)
        except SystemExit:
            self.write_output("Execution stopped.\n", "info")
        except Exception:
            traceback.print_exc(file=ConsoleOutput(self, "error"))
        finally:
            self._sync_after_execution()
            self.status_var.set("Ready")

    def _sync_after_execution(self):
        frame = self.namespace.get("df")
        if isinstance(frame, pd.DataFrame) and frame is not self.app.data:
            self.app.data = frame
            self.app.file_path = None
            self.app.display_data()

    def history_up(self, event=None):
        if not self.history:
            return "break"
        self.history_index = max(0, self.history_index - 1)
        self.input_entry.delete(0, tk.END)
        self.input_entry.insert(0, self.history[self.history_index])
        return "break"

    def history_down(self, event=None):
        if not self.history:
            return "break"
        self.history_index = min(len(self.history), self.history_index + 1)
        self.input_entry.delete(0, tk.END)
        if self.history_index < len(self.history):
            self.input_entry.insert(0, self.history[self.history_index])
        return "break"

    def run_script(self, source=None):
        if source is None:
            source = self.script_text.get("1.0", tk.END)
        if not source.strip():
            self.write_output("No script content to run.\n", "info")
            return
        self.notebook.select(0)
        target = self.script_path or "<script>"
        self._write_prompt("#", f"Running {target}")
        self._execute_code(source, target)

    def run_selection(self, event=None):
        try:
            source = self.script_text.get("sel.first", "sel.last")
        except tk.TclError:
            source = self.script_text.get("insert linestart", "insert lineend") + "\n"
        self.run_script(source)
        return "break"

    def new_script(self, event=None):
        if not self._confirm_unsaved():
            return
        self.script_text.delete("1.0", tk.END)
        self.script_path = None
        self.script_dirty = False
        self.notebook.select(1)
        self._update_line_numbers()
        self.status_var.set("New script")

    def open_script(self, event=None):
        if not self._confirm_unsaved():
            return
        path = filedialog.askopenfilename(filetypes=[("Python Files", "*.py"), ("Text Files", "*.txt"), ("All Files", "*.*")])
        if not path:
            return
        with open(path, "r", encoding="utf-8") as handle:
            content = handle.read()
        self.script_text.delete("1.0", tk.END)
        self.script_text.insert("1.0", content)
        self.script_path = path
        self.script_dirty = False
        self.notebook.select(1)
        self._update_line_numbers()
        self.status_var.set(f"Opened {path}")

    def save_script(self, event=None):
        if not self.script_path:
            return self.save_script_as()
        with open(self.script_path, "w", encoding="utf-8") as handle:
            handle.write(self.script_text.get("1.0", tk.END))
        self.script_dirty = False
        self.status_var.set(f"Saved {self.script_path}")
        return self.script_path

    def save_script_as(self, event=None):
        path = filedialog.asksaveasfilename(defaultextension=".py", filetypes=[("Python Files", "*.py"), ("Text Files", "*.txt"), ("All Files", "*.*")])
        if not path:
            return None
        self.script_path = path
        return self.save_script()

    def _confirm_unsaved(self):
        if not self.script_dirty:
            return True
        return messagebox.askyesno("Unsaved Script", "Save changes before continuing?", parent=self)

    def clear_console(self):
        self.output_text.configure(state="normal")
        self.output_text.delete("1.0", tk.END)
        self.output_text.configure(state="disabled")
        self.pending_lines = []
        self.prompt_label.configure(text=">>>")
        self.status_var.set("Console cleared")

    def clear_script(self):
        if not self._confirm_unsaved():
            return
        self.script_text.delete("1.0", tk.END)
        self.script_path = None
        self.script_dirty = False
        self._update_line_numbers()
        self.status_var.set("Script cleared")

    def zoom_in(self, event=None):
        self.font_size = min(28, self.font_size + 1)
        self._apply_fonts()
        self.status_var.set(f"Font size {self.font_size}")
        return "break"

    def zoom_out(self, event=None):
        self.font_size = max(7, self.font_size - 1)
        self._apply_fonts()
        self.status_var.set(f"Font size {self.font_size}")
        return "break"

    def reset_zoom(self, event=None):
        self.font_size = 11
        self._apply_fonts()
        self.status_var.set("Font size reset")
        return "break"

    def _apply_fonts(self):
        font = ("Consolas", self.font_size)
        self.output_text.configure(font=font)
        self.input_entry.configure(font=font)
        self.script_text.configure(font=font)
        self.line_numbers.configure(font=font)
        self.prompt_label.configure(font=font)

    def _editor_event(self, sequence):
        self.notebook.select(1)
        self.script_text.event_generate(sequence)

    def _mark_dirty(self, event=None):
        self.script_dirty = True

    def _update_line_numbers(self, event=None):
        count = int(self.script_text.index("end-1c").split(".")[0])
        numbers = "\n".join(str(index) for index in range(1, count + 1))
        self.line_numbers.configure(state="normal")
        self.line_numbers.delete("1.0", tk.END)
        self.line_numbers.insert("1.0", numbers)
        self.line_numbers.configure(state="disabled")
        line, column = self.script_text.index("insert").split(".")
        self.status_var.set(f"Ln {line}, Col {int(column) + 1}")

    def _update_status(self, event=None):
        line, column = self.script_text.index("insert").split(".")
        self.status_var.set(f"Ln {line}, Col {int(column) + 1}")

    def _sync_editor_scroll(self, first, last):
        self.script_scrollbar.set(first, last)
        self.line_numbers.yview_moveto(first)

    def _scroll_editor(self, *args):
        self.script_text.yview(*args)
        self.line_numbers.yview(*args)

    def help_api(self):
        self.notebook.select(0)
        self.app.help_api()

    def close(self):
        if not self._confirm_unsaved():
            return
        if self.app.console is self:
            self.app.console = None
        self.destroy()

class PlotBuilderDialog(tk.Toplevel):
    def __init__(self, master, app, plot_type="auto"):
        super().__init__(master)
        self.app = app
        self.title("SADSA Plot Builder")
        self.geometry("1080x720")
        self.minsize(900, 600)
        initial_key = self.app._plot_type_key(plot_type)
        self.plot_type_var = tk.StringVar(value=PLOT_TYPES[initial_key])
        self.vars = {}
        self.option_vars = {}
        self.visible_columns = []
        self.status_var = tk.StringVar(value="Choose variables and options, then click Plot.")
        self.search_var = tk.StringVar(value="")
        self._build_interface()
        self._refresh_fields()
        self.lift()
        self.focus_set()

    def _current_key(self):
        label = self.plot_type_var.get()
        for key, text in PLOT_TYPES.items():
            if text == label:
                return key
        return "auto"

    def _build_interface(self):
        main_frame = ttk.Frame(self)
        main_frame.pack(fill="both", expand=True)
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side="left", fill="y", padx=10, pady=10)
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side="right", fill="both", expand=True, padx=(0, 10), pady=10)

        setup_frame = ttk.LabelFrame(left_frame, text="Plot Setup", padding=10)
        setup_frame.pack(fill="x")
        ttk.Label(setup_frame, text="Plot type").grid(row=0, column=0, sticky="w", pady=5)
        type_combo = ttk.Combobox(setup_frame, textvariable=self.plot_type_var, values=list(PLOT_TYPES.values()), state="readonly", width=32)
        type_combo.grid(row=0, column=1, sticky="ew", pady=5)
        type_combo.bind("<<ComboboxSelected>>", self._refresh_fields)
        setup_frame.columnconfigure(1, weight=1)

        self.fields_frame = ttk.LabelFrame(left_frame, text="Variables", padding=10)
        self.fields_frame.pack(fill="x", pady=10)

        options_frame = ttk.LabelFrame(left_frame, text="Options", padding=10)
        options_frame.pack(fill="x")
        self.option_vars = {
            "title": tk.StringVar(value=""),
            "xlabel": tk.StringVar(value=""),
            "ylabel": tk.StringVar(value=""),
            "bins": tk.StringVar(value="20"),
            "top_n": tk.StringVar(value="10"),
            "agg": tk.StringVar(value="mean"),
            "window": tk.StringVar(value="3"),
            "palette": tk.StringVar(value="viridis"),
            "rotation": tk.StringVar(value="0"),
            "figsize": tk.StringVar(value="8, 5"),
            "alpha": tk.StringVar(value="0.85"),
            "kde": tk.BooleanVar(value=True),
            "normalize": tk.BooleanVar(value=False),
            "show": tk.BooleanVar(value=True)
        }
        option_rows = [
            ("Title", "title", "entry"),
            ("X label", "xlabel", "entry"),
            ("Y label", "ylabel", "entry"),
            ("Bins", "bins", "entry"),
            ("Top N", "top_n", "entry"),
            ("Aggregation", "agg", "agg"),
            ("Window", "window", "entry"),
            ("Palette", "palette", "entry"),
            ("Rotation", "rotation", "entry"),
            ("Figure size", "figsize", "entry"),
            ("Alpha", "alpha", "entry")
        ]
        for row, (label, key, kind) in enumerate(option_rows):
            ttk.Label(options_frame, text=label).grid(row=row, column=0, sticky="w", pady=3)
            if kind == "agg":
                widget = ttk.Combobox(options_frame, textvariable=self.option_vars[key], values=["mean", "sum", "count", "median", "min", "max", "std"], state="readonly")
            else:
                widget = ttk.Entry(options_frame, textvariable=self.option_vars[key])
            widget.grid(row=row, column=1, sticky="ew", pady=3)
        options_frame.columnconfigure(1, weight=1)
        toggle_row = len(option_rows)
        ttk.Checkbutton(options_frame, text="KDE overlay", variable=self.option_vars["kde"]).grid(row=toggle_row, column=0, sticky="w", pady=3)
        ttk.Checkbutton(options_frame, text="Normalize", variable=self.option_vars["normalize"]).grid(row=toggle_row, column=1, sticky="w", pady=3)
        ttk.Checkbutton(options_frame, text="Show in window", variable=self.option_vars["show"]).grid(row=toggle_row + 1, column=0, sticky="w", pady=3)

        button_frame = ttk.Frame(left_frame)
        button_frame.pack(fill="x", pady=10)
        ttk.Button(button_frame, text="Plot", command=self.on_plot).pack(side="left", padx=(0, 5))
        ttk.Button(button_frame, text="Reset", command=self.reset_options).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Close", command=self.destroy).pack(side="left", padx=5)

        columns_frame = ttk.LabelFrame(right_frame, text="Available Columns", padding=10)
        columns_frame.pack(fill="both", expand=True)
        ttk.Label(columns_frame, text="Search").pack(anchor="w")
        search_entry = ttk.Entry(columns_frame, textvariable=self.search_var)
        search_entry.pack(fill="x", pady=(0, 6))
        self.search_var.trace_add("write", lambda *args: self._filter_columns())
        list_frame = ttk.Frame(columns_frame)
        list_frame.pack(fill="both", expand=True)
        list_scrollbar = ttk.Scrollbar(list_frame, orient="vertical")
        list_scrollbar.pack(side="right", fill="y")
        self.column_listbox = tk.Listbox(list_frame, exportselection=False, yscrollcommand=list_scrollbar.set)
        self.column_listbox.pack(side="left", fill="both", expand=True)
        list_scrollbar.configure(command=self.column_listbox.yview)
        self.column_listbox.bind("<Double-1>", lambda event: self._assign_to("x"))

        assign_frame = ttk.Frame(columns_frame)
        assign_frame.pack(fill="x", pady=8)
        for label, role in (("Set X", "x"), ("Set Y", "y"), ("Set Group", "hue"), ("Set Size", "size"), ("Add to Columns", "columns")):
            ttk.Button(assign_frame, text=label, command=lambda role=role: self._assign_to(role)).pack(side="left", padx=2)

        ttk.Label(self, textvariable=self.status_var, anchor="w").pack(side="bottom", fill="x")

    def _field_specs(self):
        key = self._current_key()
        column_specs = {
            "auto": [("x", "X variable", True, "column"), ("y", "Y variable", False, "column"), ("hue", "Group / Hue", False, "column")],
            "histogram": [("x", "Value column", True, "column"), ("hue", "Group / Hue", False, "column")],
            "kde": [("x", "Value column", True, "column"), ("hue", "Group / Hue", False, "column")],
            "box": [("y", "Numeric value", True, "column"), ("x", "Category", False, "column"), ("hue", "Group / Hue", False, "column")],
            "violin": [("y", "Numeric value", True, "column"), ("x", "Category", False, "column"), ("hue", "Group / Hue", False, "column")],
            "strip": [("y", "Numeric value", True, "column"), ("x", "Category", False, "column"), ("hue", "Group / Hue", False, "column")],
            "bar": [("x", "Category", True, "column"), ("y", "Value", False, "column"), ("hue", "Group / Hue", False, "column")],
            "horizontal_bar": [("x", "Category", True, "column"), ("y", "Value", False, "column"), ("hue", "Group / Hue", False, "column")],
            "pie": [("x", "Labels", True, "column"), ("y", "Values", False, "column")],
            "grouped_bar": [("x", "Category", True, "column"), ("y", "Value", True, "column"), ("hue", "Group", True, "column")],
            "stacked_bar": [("x", "Category", True, "column"), ("y", "Value", True, "column"), ("hue", "Group", True, "column")],
            "scatter": [("x", "X variable", True, "column"), ("y", "Y variable", True, "column"), ("hue", "Group / Hue", False, "column")],
            "bubble": [("x", "X variable", True, "column"), ("y", "Y variable", True, "column"), ("size", "Size variable", True, "column"), ("hue", "Group / Hue", False, "column")],
            "line": [("y", "Value", True, "column"), ("x", "X variable", False, "column"), ("hue", "Group / Hue", False, "column")],
            "area": [("x", "X variable", True, "column"), ("y", "Value", True, "column")],
            "heatmap": [("columns", "Numeric columns", True, "columns")],
            "pairplot": [("columns", "Numeric columns", True, "columns"), ("hue", "Group / Hue", False, "column")],
            "pareto": [("x", "Category", True, "column"), ("y", "Value", False, "column")],
            "time_series": [("y", "Value", True, "column"), ("x", "Time variable", False, "column"), ("hue", "Group / Hue", False, "column")],
            "moving_average": [("y", "Value", True, "column"), ("x", "Time variable", False, "column")]
        }
        return column_specs.get(key, column_specs["auto"])

    def _refresh_fields(self, event=None):
        for child in self.fields_frame.winfo_children():
            child.destroy()
        self.fields_frame.columnconfigure(1, weight=1)
        columns = list(self.app.data.columns)
        numeric_columns = [column for column in columns if pd.api.types.is_numeric_dtype(self.app.data[column])]
        for row, (key, label, required, kind) in enumerate(self._field_specs()):
            variable = self.vars.setdefault(key, tk.StringVar(value=""))
            display_label = f"{label}{' *' if required else ''}"
            ttk.Label(self.fields_frame, text=display_label).grid(row=row, column=0, sticky="w", pady=4)
            if kind == "columns":
                if not variable.get() and numeric_columns:
                    variable.set(", ".join(numeric_columns[:6]))
                widget = ttk.Entry(self.fields_frame, textvariable=variable)
            else:
                widget = ttk.Combobox(self.fields_frame, textvariable=variable, values=[""] + columns, state="readonly")
            widget.grid(row=row, column=1, sticky="ew", pady=4)
        self.status_var.set(f"Configured: {self.plot_type_var.get()}")
        self._filter_columns()

    def _filter_columns(self):
        query = self.search_var.get().strip().lower()
        self.column_listbox.delete(0, tk.END)
        self.visible_columns = []
        for column in self.app.data.columns:
            if query and query not in str(column).lower():
                continue
            dtype = "numeric" if pd.api.types.is_numeric_dtype(self.app.data[column]) else "categorical"
            self.column_listbox.insert(tk.END, f"{column}  [{dtype}]")
            self.visible_columns.append(column)

    def _selected_column(self):
        selection = self.column_listbox.curselection()
        if not selection:
            messagebox.showinfo("No Column Selected", "Select a column from the Available Columns list first.", parent=self)
            return None
        return self.visible_columns[selection[0]]

    def _assign_to(self, role):
        column = self._selected_column()
        if not column:
            return
        specs = {spec[0] for spec in self._field_specs()}
        if role not in specs:
            messagebox.showinfo("Role Not Available", f"'{role}' is not used by this plot type.", parent=self)
            return
        if role == "columns":
            current = [item.strip() for item in self.vars[role].get().split(",") if item.strip()]
            if column not in current:
                current.append(column)
            self.vars[role].set(", ".join(current))
        else:
            self.vars[role].set(column)
        self.status_var.set(f"Set {role} = {column}")

    def _numeric_option(self, key, default=0):
        try:
            return float(self.option_vars[key].get())
        except ValueError:
            return default

    def _integer_option(self, key, default=0):
        try:
            return int(float(self.option_vars[key].get()))
        except ValueError:
            return default

    def apply(self):
        parameters = {"plot_type": self._current_key(), "show": self.option_vars["show"].get()}
        for key, label, required, kind in self._field_specs():
            value = self.vars[key].get().strip()
            if required and not value:
                raise ValueError(f"{label} is required for {PLOT_TYPES[self._current_key()]}.")
            parameters[key] = value or None
        parameters["title"] = self.option_vars["title"].get().strip() or None
        parameters["xlabel"] = self.option_vars["xlabel"].get().strip() or None
        parameters["ylabel"] = self.option_vars["ylabel"].get().strip() or None
        parameters["bins"] = max(1, self._integer_option("bins", 20))
        parameters["top_n"] = self._integer_option("top_n", 10) or None
        parameters["agg"] = self.option_vars["agg"].get()
        parameters["window"] = max(1, self._integer_option("window", 3))
        parameters["palette"] = self.option_vars["palette"].get().strip() or "viridis"
        parameters["rotation"] = self._integer_option("rotation", 0)
        parameters["figsize"] = self.option_vars["figsize"].get().strip() or "8, 5"
        parameters["alpha"] = min(1.0, max(0.1, self._numeric_option("alpha", 0.85)))
        parameters["kde"] = self.option_vars["kde"].get()
        parameters["normalize"] = self.option_vars["normalize"].get()
        figure, axes = self.app.visualize(**parameters)
        self.status_var.set(f"Created {PLOT_TYPES[self._current_key()]}")
        return figure, axes

    def on_plot(self):
        try:
            self.apply()
        except Exception as error:
            messagebox.showerror("Plot Failed", str(error), parent=self)

    def reset_options(self):
        self.option_vars["title"].set("")
        self.option_vars["xlabel"].set("")
        self.option_vars["ylabel"].set("")
        self.option_vars["bins"].set("20")
        self.option_vars["top_n"].set("10")
        self.option_vars["agg"].set("mean")
        self.option_vars["window"].set("3")
        self.option_vars["palette"].set("viridis")
        self.option_vars["rotation"].set("0")
        self.option_vars["figsize"].set("8, 5")
        self.option_vars["alpha"].set("0.85")
        self.option_vars["kde"].set(True)
        self.option_vars["normalize"].set(False)
        self.option_vars["show"].set(True)
        self.vars.clear()
        self._refresh_fields()
        self.status_var.set("Options reset")

class DocumentationWindow(tk.Toplevel):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        self.search_position = "1.0"
        self.status_var = tk.StringVar(value="Documentation loaded. Use Find or download buttons below.")
        self.title(f"{APP_SHORT_NAME} Documentation - {APP_VERSION}")
        self.geometry("1000x720")
        self.minsize(820, 560)
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill="both", expand=True)
        toolbar = ttk.Frame(main_frame)
        toolbar.pack(fill="x", pady=(0, 8))
        ttk.Label(toolbar, text="Find").pack(side="left", padx=(0, 5))
        self.search_var = tk.StringVar(value="")
        search_entry = ttk.Entry(toolbar, textvariable=self.search_var, width=42)
        search_entry.pack(side="left", padx=5)
        search_entry.bind("<Return>", lambda event: self.find())
        ttk.Button(toolbar, text="Find", command=self.find).pack(side="left", padx=5)
        for label, file_format in (("TXT", "txt"), ("Markdown", "md"), ("DOCX", "docx"), ("PDF", "pdf")):
            ttk.Button(toolbar, text=label, command=lambda file_format=file_format: self.download(file_format)).pack(side="left", padx=5)
        text_frame = ttk.Frame(main_frame)
        text_frame.pack(fill="both", expand=True)
        text_scrollbar = ttk.Scrollbar(text_frame, orient="vertical")
        text_scrollbar.pack(side="right", fill="y")
        self.text_widget = tk.Text(text_frame, wrap="word", yscrollcommand=text_scrollbar.set)
        self.text_widget.pack(side="left", fill="both", expand=True)
        text_scrollbar.configure(command=self.text_widget.yview)
        self.text_widget.insert("1.0", self.app.documentation_text())
        self.text_widget.configure(state="disabled")
        self.text_widget.tag_configure("match", background="yellow")
        ttk.Label(self, textvariable=self.status_var, anchor="w").pack(side="bottom", fill="x")

    def find(self):
        query = self.search_var.get().strip()
        if not query:
            self.status_var.set("Enter text to find.")
            return
        self.text_widget.tag_remove("match", "1.0", tk.END)
        match_start = self.text_widget.search(query, self.search_position, tk.END, nocase=True)
        if not match_start:
            match_start = self.text_widget.search(query, "1.0", tk.END, nocase=True)
            if not match_start:
                self.status_var.set(f"Text not found: {query}")
                return
        match_end = f"{match_start}+{len(query)}c"
        self.text_widget.tag_add("match", match_start, match_end)
        self.text_widget.see(match_start)
        self.search_position = match_end
        self.status_var.set(f"Found: {query}")

    def download(self, file_format):
        try:
            path = self.app.download_documentation(file_format)
            if path:
                self.status_var.set(f"Saved {file_format.upper()} documentation: {path}")
        except Exception as error:
            messagebox.showerror("Download Failed", str(error), parent=self)

class DocumentationPDF(FPDF):
    def __init__(self, app):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.app = app
        self.set_auto_page_break(auto=True, margin=18)
        self.set_margins(18, 18, 18)
        self.content_width = self.w - self.l_margin - self.r_margin

    def safe(self, value):
        return self.app._pdf_safe_text(value)

    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Arial", "I", 8)
        self.set_text_color(100, 100, 100)
        self.cell(self.content_width, 7, self.safe(f"{APP_NAME} | {APP_VERSION}"), align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.set_text_color(100, 100, 100)
        self.cell(0, 10, self.safe(f"Page {self.page_no()}"), align="C")
        self.set_text_color(0, 0, 0)

    def section_heading(self, text, level=1):
        sizes = {1: 15, 2: 12, 3: 10}
        heights = {1: 12, 2: 10, 3: 8}
        size = sizes.get(level, 10)
        if self.get_y() + heights.get(level, 8) > self.page_break_trigger:
            self.add_page()
        self.set_font("Arial", "B", size)
        self.multi_cell(self.content_width, size / 2 + 2, self.safe(text), align="L", new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def detail_row(self, label, value):
        if self.get_y() + 14 > self.page_break_trigger:
            self.add_page()
        x = self.get_x()
        y = self.get_y()
        self.set_font("Arial", "B", 9)
        self.set_fill_color(234, 241, 248)
        self.cell(34, 6, self.safe(label), border=1, align="L", fill=True, new_x="RIGHT", new_y="TOP")
        self.set_xy(x + 34, y)
        self.set_font("Arial", size=9)
        self.multi_cell(self.content_width - 34, 6, self.safe(value), border=1, align="L", new_x="LMARGIN", new_y="NEXT")

    def code_block(self, label, code):
        if self.get_y() + 20 > self.page_break_trigger:
            self.add_page()
        self.set_font("Arial", "B", 9)
        self.set_text_color(31, 78, 121)
        self.cell(self.content_width, 6, self.safe(label), align="L", new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        self.set_font("Courier", size=8)
        self.set_fill_color(245, 245, 245)
        self.multi_cell(self.content_width, 5, self.safe(code), border=1, align="L", fill=True, new_x="LMARGIN", new_y="NEXT")
        self.set_font("Arial", size=9)

class AdvancedImputationDialog(tk.Toplevel):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        self.title("Advanced Missing Data Imputation")
        self.geometry("720x440")
        self.minsize(620, 380)
        self.status_var = tk.StringVar(value="Select a target column, method, and predictors.")
        columns = list(self.app.data.columns)
        numeric_columns = [column for column in columns if pd.api.types.is_numeric_dtype(self.app.data[column])]
        missing_columns = [column for column in columns if self.app.data[column].isna().any()]
        target = missing_columns[0] if missing_columns else (numeric_columns[0] if numeric_columns else (columns[0] if columns else ""))
        predictors = [column for column in numeric_columns if column != target]
        main_frame = ttk.Frame(self, padding=12)
        main_frame.pack(fill="both", expand=True)
        self.target_var = tk.StringVar(value=target)
        self.method_var = tk.StringVar(value="Regression")
        self.predictor_var = tk.StringVar(value=", ".join(predictors))
        self.show_var = tk.BooleanVar(value=True)
        ttk.Label(main_frame, text="Target column").grid(row=0, column=0, sticky="w", pady=6)
        ttk.Combobox(main_frame, textvariable=self.target_var, values=columns, state="readonly", width=58).grid(row=0, column=1, sticky="ew", pady=6)
        ttk.Label(main_frame, text="Imputation method").grid(row=1, column=0, sticky="w", pady=6)
        ttk.Combobox(main_frame, textvariable=self.method_var, values=["Regression", "Random Forest", "KNN", "Iterative"], state="readonly", width=58).grid(row=1, column=1, sticky="ew", pady=6)
        ttk.Label(main_frame, text="Predictors (comma-separated)").grid(row=2, column=0, sticky="nw", pady=6)
        ttk.Entry(main_frame, textvariable=self.predictor_var, width=64).grid(row=2, column=1, sticky="ew", pady=6)
        ttk.Label(main_frame, text="Regression and Iterative require numeric data; Random Forest also supports categorical targets.").grid(row=3, column=1, sticky="w", pady=(0, 8))
        ttk.Checkbutton(main_frame, text="Show report window", variable=self.show_var).grid(row=4, column=1, sticky="w")
        main_frame.columnconfigure(1, weight=1)
        button_frame = ttk.Frame(self, padding=(12, 0, 12, 12))
        button_frame.pack(fill="x")
        ttk.Button(button_frame, text="Cancel", command=self.destroy).pack(side="right")
        ttk.Button(button_frame, text="Impute", command=self.apply).pack(side="right", padx=6)
        ttk.Label(self, textvariable=self.status_var, anchor="w").pack(side="bottom", fill="x")

    def apply(self):
        try:
            predictors = [value.strip() for value in self.predictor_var.get().split(",") if value.strip()]
            self.app.treat_missing_advanced(self.target_var.get(), method=self.method_var.get(), predictors=predictors, show=self.show_var.get())
            self.status_var.set("Imputation applied to the data grid.")
            self.destroy()
        except Exception as error:
            messagebox.showerror("Imputation Failed", str(error), parent=self)

class MissingDataSimulationDialog(tk.Toplevel):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        self.title("Simulate Missing Data")
        self.geometry("760x460")
        self.minsize(660, 400)
        self.status_var = tk.StringVar(value="MCAR is random; MAR depends on a predictor; MNAR depends on the column itself.")
        columns = list(self.app.data.columns)
        numeric_columns = [column for column in columns if pd.api.types.is_numeric_dtype(self.app.data[column])]
        default_columns = numeric_columns if numeric_columns else columns
        predictor = next((column for column in columns if column not in default_columns), default_columns[0] if default_columns else "")
        main_frame = ttk.Frame(self, padding=12)
        main_frame.pack(fill="both", expand=True)
        self.column_var = tk.StringVar(value=", ".join(default_columns))
        self.rate_var = tk.StringVar(value="0.2")
        self.mechanism_var = tk.StringVar(value="MCAR")
        self.predictor_var = tk.StringVar(value=predictor)
        self.seed_var = tk.StringVar(value="42")
        self.apply_var = tk.BooleanVar(value=True)
        self.show_var = tk.BooleanVar(value=True)
        ttk.Label(main_frame, text="Columns (comma-separated)").grid(row=0, column=0, sticky="w", pady=6)
        ttk.Entry(main_frame, textvariable=self.column_var, width=64).grid(row=0, column=1, sticky="ew", pady=6)
        ttk.Label(main_frame, text="Missing rate").grid(row=1, column=0, sticky="w", pady=6)
        ttk.Entry(main_frame, textvariable=self.rate_var, width=20).grid(row=1, column=1, sticky="w", pady=6)
        ttk.Label(main_frame, text="Mechanism").grid(row=2, column=0, sticky="w", pady=6)
        ttk.Combobox(main_frame, textvariable=self.mechanism_var, values=["MCAR", "MAR", "MNAR"], state="readonly", width=20).grid(row=2, column=1, sticky="w", pady=6)
        ttk.Label(main_frame, text="MAR predictor column").grid(row=3, column=0, sticky="w", pady=6)
        ttk.Combobox(main_frame, textvariable=self.predictor_var, values=columns, width=58).grid(row=3, column=1, sticky="ew", pady=6)
        ttk.Label(main_frame, text="Seed").grid(row=4, column=0, sticky="w", pady=6)
        ttk.Entry(main_frame, textvariable=self.seed_var, width=20).grid(row=4, column=1, sticky="w", pady=6)
        ttk.Checkbutton(main_frame, text="Apply to data grid", variable=self.apply_var).grid(row=5, column=1, sticky="w", pady=4)
        ttk.Checkbutton(main_frame, text="Show report window", variable=self.show_var).grid(row=6, column=1, sticky="w", pady=4)
        main_frame.columnconfigure(1, weight=1)
        button_frame = ttk.Frame(self, padding=(12, 0, 12, 12))
        button_frame.pack(fill="x")
        ttk.Button(button_frame, text="Cancel", command=self.destroy).pack(side="right")
        ttk.Button(button_frame, text="Simulate", command=self.apply).pack(side="right", padx=6)
        ttk.Label(self, textvariable=self.status_var, anchor="w").pack(side="bottom", fill="x")

    def apply(self):
        try:
            columns = [value.strip() for value in self.column_var.get().split(",") if value.strip()]
            before_missing = int(self.app.data[columns].isna().sum().sum())
            frame = self.app.simulate_missing_data(
                columns=columns,
                rate=float(self.rate_var.get()),
                mechanism=self.mechanism_var.get(),
                predictor=self.predictor_var.get(),
                seed=int(float(self.seed_var.get())),
                apply=self.apply_var.get(),
                show=self.show_var.get()
            )
            added = int(frame[columns].isna().sum().sum()) - before_missing
            self.status_var.set(f"Missing values simulated for {len(columns)} columns; added {max(added, 0)} cells.")
            self.destroy()
        except Exception as error:
            messagebox.showerror("Simulation Failed", str(error), parent=self)

class MissingnessTestDialog(tk.Toplevel):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        self.title("Missingness Significance Test")
        self.geometry("760x440")
        self.minsize(660, 380)
        self.status_var = tk.StringVar(value="Without a group column, the test screens missingness against all other observed variables.")
        columns = list(self.app.data.columns)
        missing_columns = [column for column in columns if self.app.data[column].isna().any()]
        group_options = ["None"] + [column for column in columns if not pd.api.types.is_numeric_dtype(self.app.data[column]) or self.app.data[column].nunique(dropna=True) <= 10]
        main_frame = ttk.Frame(self, padding=12)
        main_frame.pack(fill="both", expand=True)
        self.column_var = tk.StringVar(value=", ".join(missing_columns if missing_columns else columns))
        self.group_var = tk.StringVar(value="None")
        self.alpha_var = tk.StringVar(value="0.05")
        self.show_var = tk.BooleanVar(value=True)
        ttk.Label(main_frame, text="Columns to test (comma-separated)").grid(row=0, column=0, sticky="w", pady=6)
        ttk.Entry(main_frame, textvariable=self.column_var, width=64).grid(row=0, column=1, sticky="ew", pady=6)
        ttk.Label(main_frame, text="Group column").grid(row=1, column=0, sticky="w", pady=6)
        ttk.Combobox(main_frame, textvariable=self.group_var, values=group_options, width=58).grid(row=1, column=1, sticky="ew", pady=6)
        ttk.Label(main_frame, text="Alpha").grid(row=2, column=0, sticky="w", pady=6)
        ttk.Entry(main_frame, textvariable=self.alpha_var, width=20).grid(row=2, column=1, sticky="w", pady=6)
        ttk.Checkbutton(main_frame, text="Show report window", variable=self.show_var).grid(row=3, column=1, sticky="w", pady=4)
        main_frame.columnconfigure(1, weight=1)
        button_frame = ttk.Frame(self, padding=(12, 0, 12, 12))
        button_frame.pack(fill="x")
        ttk.Button(button_frame, text="Cancel", command=self.destroy).pack(side="right")
        ttk.Button(button_frame, text="Run Test", command=self.apply).pack(side="right", padx=6)
        ttk.Label(self, textvariable=self.status_var, anchor="w").pack(side="bottom", fill="x")

    def apply(self):
        try:
            columns = [value.strip() for value in self.column_var.get().split(",") if value.strip()]
            group_column = self.group_var.get()
            report = self.app.test_missing_significance(
                columns=columns,
                group_column=None if group_column == "None" else group_column,
                alpha=float(self.alpha_var.get()),
                show=self.show_var.get()
            )
            self.status_var.set(report["Interpretation"])
            self.destroy()
        except Exception as error:
            messagebox.showerror("Test Failed", str(error), parent=self)

class TimeSeriesSimulationDialog(tk.Toplevel):
    PRESETS = (
        ("Stationary Series", "stationary"),
        ("Non-Stationary Trend", "non_stationary"),
        ("Seasonal Series", "seasonal"),
        ("Trend + Seasonal (Holt-Winters)", "holt_winters"),
        ("Moving-Average Friendly", "moving_average"),
        ("Random Walk", "random_walk"),
        ("Custom Series", "custom")
    )
    TIME_MODES = (("Date range", "date"), ("Numeric index", "numeric"), ("Period labels", "period"))
    FREQUENCIES = (("Hourly", "H"), ("Daily", "D"), ("Business day", "B"), ("Weekly", "W"), ("Monthly", "M"), ("Quarterly", "Q"), ("Yearly", "Y"))
    TREND_TYPES = (("None", "none"), ("Linear", "linear"), ("Quadratic", "quadratic"))
    SEASONAL_MODELS = (("Additive", "additive"), ("Multiplicative", "multiplicative"))
    NOISE_TYPES = (("Normal", "normal"), ("Uniform", "uniform"), ("None", "none"))

    def __init__(self, master, app, preset="custom"):
        super().__init__(master)
        self.app = app
        self.title("Time Series Data Simulation")
        self.geometry("900x680")
        self.minsize(780, 560)
        self.status_var = tk.StringVar(value="Values are prefilled from the current data; adjust any option, then click Simulate.")
        defaults = self.app.time_series_simulation_defaults(preset)
        columns = list(self.app.data.columns) if self.app.data is not None else []
        self.vars = {}
        self.random_walk_var = tk.BooleanVar(value=False)
        self.apply_var = tk.BooleanVar(value=True)
        self.show_var = tk.BooleanVar(value=True)
        self.preset_var = tk.StringVar(value=self._display_for(defaults["preset"], self.PRESETS))
        main_frame = ttk.Frame(self, padding=12)
        main_frame.pack(fill="both", expand=True)
        ttk.Label(main_frame, text="Simulation preset").grid(row=0, column=0, sticky="w", pady=6)
        preset_combo = ttk.Combobox(main_frame, textvariable=self.preset_var, values=[label for label, _ in self.PRESETS], state="readonly", width=52)
        preset_combo.grid(row=0, column=1, sticky="ew", pady=6)
        preset_combo.bind("<<ComboboxSelected>>", self._apply_preset)
        main_frame.columnconfigure(1, weight=1)
        notebook = ttk.Notebook(main_frame)
        notebook.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(8, 0))
        main_frame.rowconfigure(1, weight=1)
        time_tab = ttk.Frame(notebook, padding=10)
        data_tab = ttk.Frame(notebook, padding=10)
        notebook.add(time_tab, text="Time Variable")
        notebook.add(data_tab, text="Data Variable")
        self._build_time_tab(time_tab, columns)
        self._build_data_tab(data_tab, columns)
        self._set_defaults(defaults)
        button_frame = ttk.Frame(self, padding=(12, 0, 12, 12))
        button_frame.pack(fill="x")
        ttk.Button(button_frame, text="Cancel", command=self.destroy).pack(side="right")
        ttk.Button(button_frame, text="Simulate", command=self.apply).pack(side="right", padx=6)
        ttk.Label(self, textvariable=self.status_var, anchor="w").pack(side="bottom", fill="x")

    def _add_entry(self, parent, label, key, row, width=52):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=5)
        variable = tk.StringVar(value="")
        self.vars[key] = variable
        ttk.Entry(parent, textvariable=variable, width=width).grid(row=row, column=1, sticky="ew", pady=5)
        parent.columnconfigure(1, weight=1)

    def _add_combo(self, parent, label, key, options, row, width=50, readonly=True):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=5)
        variable = tk.StringVar(value="")
        self.vars[key] = variable
        state = "readonly" if readonly else "normal"
        ttk.Combobox(parent, textvariable=variable, values=[label for label, _ in options], state=state, width=width).grid(row=row, column=1, sticky="ew", pady=5)
        parent.columnconfigure(1, weight=1)

    def _build_time_tab(self, parent, columns):
        self._add_combo(parent, "Time variable", "time_column", tuple((column, column) for column in columns), 0, readonly=False)
        self._add_combo(parent, "Time mode", "time_mode", self.TIME_MODES, 1)
        self._add_entry(parent, "Start date / value", "start", 2)
        self._add_combo(parent, "Frequency", "frequency", self.FREQUENCIES, 3)
        self._add_entry(parent, "Numeric step", "time_step", 4, width=20)
        self._add_entry(parent, "Rows", "rows", 5, width=20)
        ttk.Label(parent, text="Date range uses the frequency above; numeric index uses the start and step; period labels use P1, P2, ...").grid(row=6, column=1, sticky="w", pady=(8, 0))

    def _build_data_tab(self, parent, columns):
        self._add_combo(parent, "Data variable", "value_column", tuple((column, column) for column in columns), 0, readonly=False)
        self._add_combo(parent, "Trend type", "trend_type", self.TREND_TYPES, 1)
        self._add_entry(parent, "Base level", "base", 2, width=20)
        self._add_entry(parent, "Trend slope", "trend", 3, width=20)
        self._add_combo(parent, "Seasonal model", "seasonal_model", self.SEASONAL_MODELS, 4)
        self._add_entry(parent, "Seasonal amplitude", "seasonality", 5, width=20)
        self._add_entry(parent, "Seasonal period", "period", 6, width=20)
        self._add_combo(parent, "Noise type", "noise_type", self.NOISE_TYPES, 7)
        self._add_entry(parent, "Noise magnitude", "noise", 8, width=20)
        self._add_entry(parent, "Seed", "seed", 9, width=20)
        ttk.Checkbutton(parent, text="Random walk drift", variable=self.random_walk_var).grid(row=10, column=1, sticky="w", pady=4)
        ttk.Checkbutton(parent, text="Apply to data grid", variable=self.apply_var).grid(row=11, column=1, sticky="w", pady=4)
        ttk.Checkbutton(parent, text="Show report window", variable=self.show_var).grid(row=12, column=1, sticky="w", pady=4)

    def _display_for(self, value, options):
        for label, key in options:
            if str(key).strip().lower() == str(value).strip().lower():
                return label
            if str(key).strip().upper() == str(value).strip().upper():
                return label
        return str(value)

    def _key_for(self, value, options):
        for label, key in options:
            if str(label).strip().lower() == str(value).strip().lower():
                return key
        return str(value)

    def _current_preset(self):
        return self._key_for(self.preset_var.get(), self.PRESETS)

    def _set_defaults(self, defaults):
        display_values = {
            "preset": self._display_for(defaults["preset"], self.PRESETS),
            "time_mode": self._display_for(defaults["time_mode"], self.TIME_MODES),
            "frequency": self._display_for(defaults["frequency"], self.FREQUENCIES),
            "trend_type": self._display_for(defaults["trend_type"], self.TREND_TYPES),
            "seasonal_model": self._display_for(defaults["seasonal_model"], self.SEASONAL_MODELS),
            "noise_type": self._display_for(defaults["noise_type"], self.NOISE_TYPES)
        }
        for key, variable in self.vars.items():
            variable.set(str(display_values.get(key, defaults.get(key, ""))))
        self.random_walk_var.set(bool(defaults["random_walk"]))
        self.apply_var.set(bool(defaults["apply"]))
        self.show_var.set(bool(defaults["show"]))

    def _apply_preset(self, event=None):
        self._set_defaults(self.app.time_series_simulation_defaults(self._current_preset()))
        self.status_var.set("Preset values applied. Adjust any option, then click Simulate.")

    def apply(self):
        try:
            frame = self.app.simulate_time_series(
                time_column=self.vars["time_column"].get(),
                value_column=self.vars["value_column"].get(),
                rows=int(float(self.vars["rows"].get())),
                start=self.vars["start"].get(),
                frequency=self._key_for(self.vars["frequency"].get(), self.FREQUENCIES),
                time_mode=self._key_for(self.vars["time_mode"].get(), self.TIME_MODES),
                time_step=float(self.vars["time_step"].get()),
                base=float(self.vars["base"].get()),
                trend_type=self._key_for(self.vars["trend_type"].get(), self.TREND_TYPES),
                trend=float(self.vars["trend"].get()),
                seasonal_model=self._key_for(self.vars["seasonal_model"].get(), self.SEASONAL_MODELS),
                seasonality=float(self.vars["seasonality"].get()),
                period=int(float(self.vars["period"].get())),
                noise_type=self._key_for(self.vars["noise_type"].get(), self.NOISE_TYPES),
                noise=float(self.vars["noise"].get()),
                random_walk=self.random_walk_var.get(),
                seed=int(float(self.vars["seed"].get())),
                apply=self.apply_var.get(),
                show=self.show_var.get(),
                preset=self._current_preset()
            )
            self.status_var.set(f"Generated {len(frame)} time-series rows.")
            self.destroy()
        except Exception as error:
            messagebox.showerror("Simulation Failed", str(error), parent=self)

class TransformationSimulationDialog(tk.Toplevel):
    def __init__(self, master, app, mode, defaults):
        super().__init__(master)
        self.app = app
        self.mode = mode
        self.vars = {}
        self.specs_text = None
        self.covariance_text = None
        self.status_var = tk.StringVar(value="Values are inferred from the current data; edit if needed, then click OK.")
        self.title("Data Simulations" if mode == "data" else "Multivariate Normal Distribution")
        self.geometry("780x560")
        self.minsize(680, 460)
        main_frame = ttk.Frame(self, padding=12)
        main_frame.pack(fill="both", expand=True)
        if mode == "data":
            self._build_data_interface(main_frame, defaults)
        else:
            self._build_mvn_interface(main_frame, defaults)
        button_frame = ttk.Frame(self, padding=(12, 0, 12, 12))
        button_frame.pack(fill="x")
        ttk.Button(button_frame, text="Cancel", command=self.destroy).pack(side="right")
        ttk.Button(button_frame, text="OK", command=self.apply).pack(side="right", padx=6)
        ttk.Label(self, textvariable=self.status_var, anchor="w").pack(side="bottom", fill="x")

    def _add_entry(self, parent, label, key, value, row):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=5)
        variable = tk.StringVar(value=str(value))
        self.vars[key] = variable
        ttk.Entry(parent, textvariable=variable, width=64).grid(row=row, column=1, sticky="ew", pady=5)
        parent.columnconfigure(1, weight=1)

    def _add_text(self, parent, label, row, content, height):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="nw", pady=5)
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=1, sticky="nsew", pady=5)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        text_widget = tk.Text(frame, height=height, wrap="none")
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        text_widget.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        text_widget.insert("1.0", content)
        return text_widget

    def _build_data_interface(self, parent, defaults):
        frame = ttk.LabelFrame(parent, text="Simulation Settings", padding=10)
        frame.pack(fill="both", expand=True)
        self._add_entry(frame, "Rows", "rows", defaults["rows"], 0)
        self._add_entry(frame, "Seed", "seed", defaults["seed"], 1)
        self.specs_text = self._add_text(frame, "Variable specifications\nName | Type | Details", 2, defaults["specifications"], 10)
        ttk.Label(frame, text="Types: int, float, normal, category. Details: start,stop for numbers; comma-separated labels for category.").grid(row=3, column=1, sticky="w", pady=(6, 0))
        frame.rowconfigure(2, weight=1)

    def _build_mvn_interface(self, parent, defaults):
        frame = ttk.LabelFrame(parent, text="Multivariate Normal Settings", padding=10)
        frame.pack(fill="both", expand=True)
        self._add_entry(frame, "Variable names", "names", defaults["names"], 0)
        self._add_entry(frame, "Means", "means", defaults["means"], 1)
        self._add_entry(frame, "Samples", "samples", defaults["samples"], 2)
        self._add_entry(frame, "Seed", "seed", defaults["seed"], 3)
        self.covariance_text = self._add_text(frame, "Covariance matrix\nUse semicolons for rows", 4, defaults["covariance"], 7)
        ttk.Label(frame, text="Example: 1, 0.5; 0.5, 1").grid(row=5, column=1, sticky="w", pady=(6, 0))
        frame.rowconfigure(4, weight=1)

    def apply(self):
        try:
            if self.mode == "data":
                self._apply_data_simulation()
            else:
                self._apply_multivariate_normal()
        except Exception as error:
            messagebox.showerror("Simulation Failed", str(error), parent=self)

    def _append_frame(self, generated):
        frame = pd.DataFrame(generated)
        if self.app.data is not None:
            frame = pd.concat([self.app.data.reset_index(drop=True), frame.reset_index(drop=True)], axis=1)
        self.app.data = frame
        self.app.display_data()
        return frame

    def _apply_data_simulation(self):
        rows = max(1, int(float(self.vars["rows"].get())))
        seed = int(float(self.vars["seed"].get()))
        rng = np.random.default_rng(seed)
        generated = {}
        for raw_line in self.specs_text.get("1.0", tk.END).splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parts = [part.strip() for part in line.split("|")]
            if len(parts) < 3:
                raise ValueError(f"Invalid specification: {line}. Use Name | Type | Details.")
            name = parts[0]
            kind = parts[1].lower()
            details = " | ".join(parts[2:]).strip()
            if not name:
                raise ValueError("Variable names cannot be blank.")
            if kind in ("int", "integer", "float", "uniform", "normal", "gaussian"):
                bounds = [value for value in re.split(r"[,;\s]+", details) if value]
                if len(bounds) < 2:
                    raise ValueError(f"Numeric variable '{name}' needs two values: start, stop.")
                first_value = float(bounds[0])
                second_value = float(bounds[1])
                if kind in ("int", "integer"):
                    low = int(first_value)
                    high = int(second_value)
                    if high <= low:
                        high = low + 1
                    generated[name] = rng.integers(low, high, rows)
                elif kind in ("normal", "gaussian"):
                    generated[name] = rng.normal(first_value, max(abs(second_value), 1e-9), rows)
                else:
                    if second_value <= first_value:
                        second_value = first_value + 1
                    generated[name] = rng.uniform(first_value, second_value, rows)
            elif kind in ("category", "categorical", "choice"):
                choices = [choice.strip() for choice in re.split(r"[,;]+", details) if choice.strip()]
                if not choices:
                    raise ValueError(f"Category variable '{name}' needs at least one label.")
                generated[name] = rng.choice(choices, rows)
            else:
                raise ValueError(f"Unsupported type '{parts[1]}' for '{name}'.")
        if not generated:
            raise ValueError("Add at least one variable specification.")
        frame = self._append_frame(generated)
        self.status_var.set(f"Generated {frame.shape[0]} rows and {len(generated)} simulated variables.")
        messagebox.showinfo("Success", "Data simulated successfully!", parent=self)

    def _apply_multivariate_normal(self):
        names = [name.strip() for name in self.vars["names"].get().split(",") if name.strip()]
        if not names:
            raise ValueError("Enter at least one variable name.")
        means = np.asarray([float(value.strip()) for value in self.vars["means"].get().split(",") if value.strip()], dtype=float)
        if len(means) != len(names):
            raise ValueError("The number of means must match the number of variable names.")
        covariance = self.app._parse_covariance_text(self.covariance_text.get("1.0", tk.END), len(names))
        if not np.allclose(covariance, covariance.T):
            raise ValueError("The covariance matrix must be symmetric.")
        eigenvalues = np.linalg.eigvalsh(covariance)
        if np.min(eigenvalues) < -1e-8:
            raise ValueError("The covariance matrix must be positive semi-definite.")
        samples = max(1, int(float(self.vars["samples"].get())))
        seed = int(float(self.vars["seed"].get()))
        rng = np.random.default_rng(seed)
        values = rng.multivariate_normal(means, covariance, samples)
        frame = pd.DataFrame(values, columns=names)
        if self.app.data is not None:
            frame = pd.concat([self.app.data.reset_index(drop=True), frame.reset_index(drop=True)], axis=1)
        self.app.data = frame
        self.app.display_data()
        self.status_var.set(f"Generated {samples} rows from a {len(names)}-variable multivariate normal distribution.")
        messagebox.showinfo("Success", "Multivariate normal data appended to the data grid.", parent=self)

class MLSimulationDialog(tk.Toplevel):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        self.title("Machine Learning Simulations")
        self.geometry("860x560")
        self.minsize(720, 480)
        self.status_var = tk.StringVar(value="Select a technique to replace the data grid with a demo dataset.")
        main_frame = ttk.Frame(self, padding=12)
        main_frame.pack(fill="both", expand=True)
        options = self.app._ml_simulation_options()
        self._build_group(main_frame, "Supervised Learning", options[:7], 0)
        self._build_group(main_frame, "Unsupervised Learning", options[7:], 1)
        button_frame = ttk.Frame(self, padding=(12, 0, 12, 12))
        button_frame.pack(fill="x")
        ttk.Button(button_frame, text="Close", command=self.destroy).pack(side="right")
        ttk.Label(self, textvariable=self.status_var, anchor="w").pack(side="bottom", fill="x")

    def _build_group(self, parent, title, options, column):
        frame = ttk.LabelFrame(parent, text=title, padding=10)
        frame.grid(row=0, column=column, sticky="nsew", padx=6, pady=6)
        parent.columnconfigure(column, weight=1)
        parent.rowconfigure(0, weight=1)
        for index, (label, key) in enumerate(options):
            row = index // 2
            column_index = index % 2
            ttk.Button(frame, text=label, command=lambda key=key, label=label: self.generate(key, label)).grid(row=row, column=column_index, sticky="ew", padx=4, pady=4)
            frame.columnconfigure(column_index, weight=1)

    def generate(self, key, label):
        try:
            frame = self.app.generate_ml_demo(key)
            self.status_var.set(f"Generated {label}: {frame.shape[0]} rows x {frame.shape[1]} columns")
        except Exception as error:
            messagebox.showerror("Simulation Failed", str(error), parent=self)

# Run the application
root = tk.Tk()
root.state("zoomed")  # Ensure window is maximized on startup
app = EditableCSVApp(root)
root.mainloop()
