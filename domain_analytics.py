import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import statsmodels.api as sm
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


DOMAIN_LABELS = {
    "finance": "Finance",
    "hrm": "HRM",
    "it_operations": "IT & Operations",
    "international": "International Business",
}

DOMAIN_SIMULATIONS = {
    "finance": [
        ("Credit Risk Dataset", "credit_risk"),
        ("Fraudulent Transactions", "fraud"),
        ("Portfolio Returns", "portfolio"),
        ("Loan Applications", "applications"),
        ("Collections History", "collections"),
    ],
    "hrm": [
        ("Employee Attrition", "attrition"),
        ("Compensation & Reviews", "compensation"),
        ("Headcount History", "headcount"),
    ],
    "it_operations": [
        ("Incident Tickets", "incidents"),
        ("Sensor Telemetry", "sensors"),
        ("Capacity Metrics", "capacity"),
        ("Supplier Register", "suppliers"),
        ("Process Event Log", "events"),
    ],
    "international": [
        ("Trade Flows", "trade"),
        ("FX Returns", "fx"),
        ("Tariff Scenarios", "tariff"),
    ],
}

DOMAIN_ANALYSES = {
    "finance": [
        ("Credit Risk Scoring", "credit_risk"),
        ("Fraud / Anomaly Detection", "fraud_detection"),
        ("Portfolio Risk (VaR)", "portfolio_risk"),
        ("Fair Lending Audit", "fair_lending"),
        ("Credit Demand Forecast", "demand_forecast"),
    ],
    "hrm": [
        ("Attrition Prediction", "attrition"),
        ("Pay Equity Analysis", "pay_equity"),
        ("Employee Segmentation", "employee_segmentation"),
        ("Workforce Demand Forecast", "workforce_forecast"),
    ],
    "it_operations": [
        ("Predictive Maintenance", "predictive_maintenance"),
        ("Incident Priority Classification", "incident_priority"),
        ("SLA Breach Prediction", "sla_breach"),
        ("Capacity Forecasting", "capacity_forecast"),
        ("Inventory Reorder Simulation", "inventory_reorder"),
        ("Supplier Risk Network", "supplier_risk"),
        ("Process Cycle Mining", "process_cycle"),
    ],
    "international": [
        ("Country Market Segmentation", "country_segmentation"),
        ("Export Demand Forecast", "export_forecast"),
        ("FX Exposure Risk", "fx_risk"),
        ("Trade Anomaly Detection", "trade_anomaly"),
        ("Gravity Model of Trade", "gravity_model"),
        ("Tariff Impact Simulation", "tariff_impact"),
    ],
}

ANALYSIS_TITLES = {
    ("finance", "credit_risk"): "Credit Risk Scoring",
    ("finance", "fraud_detection"): "Fraud / Anomaly Detection",
    ("finance", "portfolio_risk"): "Portfolio Risk (VaR)",
    ("finance", "fair_lending"): "Fair Lending Audit",
    ("finance", "demand_forecast"): "Credit Demand Forecast",
    ("hrm", "attrition"): "Employee Attrition Prediction",
    ("hrm", "pay_equity"): "Pay Equity Analysis",
    ("hrm", "employee_segmentation"): "Employee Segmentation",
    ("hrm", "workforce_forecast"): "Workforce Demand Forecast",
    ("it_operations", "predictive_maintenance"): "Predictive Maintenance",
    ("it_operations", "incident_priority"): "Incident Priority Classification",
    ("it_operations", "sla_breach"): "SLA Breach Prediction",
    ("it_operations", "capacity_forecast"): "Capacity Forecasting",
    ("it_operations", "inventory_reorder"): "Inventory Reorder Simulation",
    ("it_operations", "supplier_risk"): "Supplier Risk Network",
    ("it_operations", "process_cycle"): "Process Cycle Mining",
    ("international", "country_segmentation"): "Country Market Segmentation",
    ("international", "export_forecast"): "Export Demand Forecast",
    ("international", "fx_risk"): "FX Exposure Risk",
    ("international", "trade_anomaly"): "Trade Anomaly Detection",
    ("international", "gravity_model"): "Gravity Model of Trade",
    ("international", "tariff_impact"): "Tariff Impact Simulation",
}

PARAMETER_SPECS = {
    "finance": {
        "credit_risk": [
            ("target", "Default column", "Default"),
            ("features", "Feature columns (comma separated)", "Age, AnnualIncome, CreditScore, DebtToIncome, LoanAmount, CreditHistoryYears, EmploymentYears"),
            ("model", "Model (logistic / gradient_boosting)", "gradient_boosting"),
            ("test_size", "Test fraction", "0.25"),
            ("seed", "Random seed", "42"),
        ],
        "fraud_detection": [
            ("features", "Numeric feature columns", "Amount, HourOfDay, DistanceFromHome, PreviousFlags, AccountAgeMonths"),
            ("target", "Optional fraud label column", "IsFraud"),
            ("model", "Method (isolation_forest / local_outlier / one_class_svm)", "isolation_forest"),
            ("contamination", "Expected anomaly rate", "0.08"),
            ("top_n", "Anomalies to display", "20"),
            ("seed", "Random seed", "42"),
        ],
        "portfolio_risk": [
            ("returns", "Return columns (comma separated)", "Return_A, Return_B, Return_C, Return_D"),
            ("confidence", "Confidence level (0.90 - 0.99)", "0.95"),
            ("weights", "Optional portfolio weights", ""),
        ],
        "fair_lending": [
            ("decision", "Approval decision column", "Approved"),
            ("group", "Protected group column", "Gender"),
            ("features", "Control feature columns (optional)", "AnnualIncome, CreditScore, DebtToIncome"),
            ("target", "Optional outcome column (default)", "Default"),
        ],
        "demand_forecast": [
            ("value", "Value column", "Collections"),
            ("period", "Date or period column", "Date"),
            ("periods", "Forecast periods", "6"),
            ("model", "Model (holt_winters / arima)", "holt_winters"),
            ("seasonal_periods", "Seasonal periods", "12"),
        ],
    },
    "hrm": {
        "attrition": [
            ("target", "Attrition column", "Attrition"),
            ("features", "Feature columns (comma separated)", "Age, TenureYears, MonthlyIncome, JobSatisfaction, PerformanceRating, OvertimeHours, DistanceFromHome"),
            ("model", "Model (logistic / gradient_boosting)", "gradient_boosting"),
            ("test_size", "Test fraction", "0.25"),
            ("seed", "Random seed", "42"),
        ],
        "pay_equity": [
            ("salary", "Salary column", "Salary"),
            ("group", "Group column (gender / category)", "Gender"),
            ("role", "Role or department column (optional)", "Role"),
            ("performance", "Performance columns (optional)", "TenureYears, PerformanceRating"),
        ],
        "employee_segmentation": [
            ("features", "Numeric feature columns", "Age, TenureYears, MonthlyIncome, JobSatisfaction, PerformanceRating"),
            ("clusters", "Number of clusters", "3"),
            ("seed", "Random seed", "42"),
        ],
        "workforce_forecast": [
            ("value", "Headcount column", "Headcount"),
            ("period", "Date or period column", "Date"),
            ("periods", "Forecast periods", "6"),
            ("model", "Model (holt_winters / arima)", "holt_winters"),
            ("seasonal_periods", "Seasonal periods", "12"),
        ],
    },
    "it_operations": {
        "predictive_maintenance": [
            ("features", "Sensor feature columns", "Temperature, Vibration, Pressure, RuntimeHours"),
            ("target", "Optional failure label", "Failure"),
            ("model", "Method (isolation_forest / local_outlier / one_class_svm)", "isolation_forest"),
            ("contamination", "Expected anomaly rate", "0.08"),
            ("top_n", "Anomalies to display", "20"),
            ("seed", "Random seed", "42"),
        ],
        "incident_priority": [
            ("target", "Priority column", "Priority"),
            ("features", "Feature columns", "AffectedUsers, AgeHours, CategoryCode, AssignmentGroupCode"),
            ("model", "Model (logistic / gradient_boosting)", "gradient_boosting"),
            ("test_size", "Test fraction", "0.25"),
            ("seed", "Random seed", "42"),
        ],
        "sla_breach": [
            ("target", "SLA breach column", "SLA_Breach"),
            ("features", "Feature columns", "AffectedUsers, AgeHours, CategoryCode, PriorityCode"),
            ("model", "Model (logistic / gradient_boosting)", "gradient_boosting"),
            ("test_size", "Test fraction", "0.25"),
            ("seed", "Random seed", "42"),
        ],
        "capacity_forecast": [
            ("value", "Demand column", "Demand"),
            ("period", "Date or period column", "Date"),
            ("periods", "Forecast periods", "6"),
            ("model", "Model (holt_winters / arima)", "holt_winters"),
            ("seasonal_periods", "Seasonal periods", "12"),
        ],
        "inventory_reorder": [
            ("demand", "Daily demand column", "Demand"),
            ("lead_time", "Lead time in days", "7"),
            ("service_level", "Service level (0.85 - 0.99)", "0.95"),
            ("scenarios", "Monte Carlo scenarios", "1000"),
            ("seed", "Random seed", "42"),
        ],
        "supplier_risk": [
            ("features", "Risk feature columns", "Spend, DefectRate, LeadDays, OnTimeRate"),
            ("group", "Network grouping column", "Country"),
            ("clusters", "Number of risk clusters", "3"),
            ("seed", "Random seed", "42"),
        ],
        "process_cycle": [
            ("case", "Case id column", "CaseID"),
            ("activity", "Activity column", "Activity"),
            ("timestamp", "Timestamp column", "Timestamp"),
            ("top_n", "Transitions to display", "10"),
        ],
    },
    "international": {
        "country_segmentation": [
            ("features", "Macro indicator columns", "ExporterGDP, ImporterGDP, DistanceKm, TradeValue"),
            ("clusters", "Number of market clusters", "3"),
            ("seed", "Random seed", "42"),
        ],
        "export_forecast": [
            ("value", "Export value column", "TradeValue"),
            ("period", "Date or period column", "Date"),
            ("filter_column", "Optional country column", "Country"),
            ("filter_value", "Optional country value", ""),
            ("periods", "Forecast periods", "6"),
            ("model", "Model (holt_winters / arima)", "holt_winters"),
            ("seasonal_periods", "Seasonal periods", "12"),
        ],
        "fx_risk": [
            ("returns", "FX return columns", "Return_USD_EUR, Return_USD_GBP, Return_USD_JPY, Return_USD_INR"),
            ("confidence", "Confidence level (0.90 - 0.99)", "0.95"),
            ("shocks", "Stress shocks in percent", "-10, -5, 5, 10"),
            ("weights", "Optional exposure weights", ""),
        ],
        "trade_anomaly": [
            ("features", "Numeric trade feature columns", "ExporterGDP, ImporterGDP, DistanceKm, TradeValue"),
            ("target", "Optional anomaly label", ""),
            ("model", "Method (isolation_forest / local_outlier / one_class_svm)", "isolation_forest"),
            ("contamination", "Expected anomaly rate", "0.08"),
            ("top_n", "Anomalies to display", "20"),
            ("seed", "Random seed", "42"),
        ],
        "gravity_model": [
            ("flow", "Trade flow column", "TradeValue"),
            ("exporter_gdp", "Exporter GDP column", "ExporterGDP"),
            ("importer_gdp", "Importer GDP column", "ImporterGDP"),
            ("distance", "Distance column", "DistanceKm"),
            ("dummy", "Optional common currency dummy", "CommonCurrency"),
        ],
        "tariff_impact": [
            ("price", "Base price column", "BasePrice"),
            ("demand", "Demand column", "DemandUnits"),
            ("elasticity", "Price elasticity column", "PriceElasticity"),
            ("tariff", "Tariff rate (e.g. 0.10)", "0.10"),
            ("scenarios", "Monte Carlo scenarios", "500"),
            ("seed", "Random seed", "42"),
        ],
    },
}


def _key(value):
    return str(value or "").strip().lower().replace(" ", "_").replace("-", "_").replace("/", "_")


def _split_columns(value):
    if isinstance(value, (list, tuple, set, pd.Index)):
        return [str(column).strip() for column in value if str(column).strip()]
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def _number(value, default, minimum=None, maximum=None, integer=False):
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        number = float(default)
    if minimum is not None:
        number = max(minimum, number)
    if maximum is not None:
        number = min(maximum, number)
    return int(round(number)) if integer else number


def _columns(frame, columns, label="columns"):
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing {label}: {', '.join(missing)}")
    return frame[columns].copy()


def _numeric_columns(frame, exclude=()):
    excluded = set(exclude)
    return [column for column in frame.select_dtypes(include=[np.number]).columns if column not in excluded]


def _find_column(frame, keywords, fallback=None):
    lowered = {str(column).lower(): column for column in frame.columns}
    for keyword in keywords:
        for name, column in lowered.items():
            if keyword in name:
                return column
    return fallback if fallback in frame.columns else None


def _positive_label(values):
    labels = list(pd.unique(values))
    for candidate in ("Yes", "yes", "1", 1, True, "True", "Default", "Fraud", "Attrition", "Churn"):
        for label in labels:
            if str(label).strip().lower() == str(candidate).strip().lower():
                return label
    return sorted(labels, key=str)[-1]


def _probability(model, features, positive):
    if not hasattr(model, "predict_proba"):
        return None
    probabilities = model.predict_proba(features)
    classes = list(model.classes_)
    if positive not in classes:
        return None
    return probabilities[:, classes.index(positive)]


def _feature_effects(model, feature_names):
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        values = np.asarray(model.coef_, dtype=float)
        if values.ndim > 1:
            values = values[0]
    else:
        values = np.zeros(len(feature_names))
    effects = pd.DataFrame({"Feature": feature_names, "Effect": values})
    effects["Absolute Effect"] = effects["Effect"].abs()
    return effects.sort_values("Absolute Effect", ascending=False).reset_index(drop=True)[["Feature", "Effect", "Absolute Effect"]]


def _confusion_frame(actual, predicted):
    labels = sorted(set(actual.astype(str)) | set(predicted.astype(str)))
    matrix = confusion_matrix(actual.astype(str), predicted.astype(str), labels=labels)
    frame = pd.DataFrame(matrix, index=[f"Actual {label}" for label in labels], columns=[f"Predicted {label}" for label in labels])
    frame["Actual Total"] = frame.sum(axis=1)
    totals = frame.sum(axis=0)
    frame.loc["Predicted Total"] = totals
    frame.loc["Predicted Total", "Actual Total"] = int(matrix.sum())
    return frame, matrix, labels


def _figure_from_matrix(matrix, row_labels, column_labels, title):
    figure, axis = plt.subplots(figsize=(7, 5))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(column_labels)))
    axis.set_yticks(range(len(row_labels)))
    axis.set_xticklabels(column_labels, rotation=30, ha="right")
    axis.set_yticklabels(row_labels)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_title(title)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, str(int(matrix[row, column])), ha="center", va="center", color="darkblue")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    return figure


def _build_classification(frame, params, domain, technique):
    target = str(params.get("target") or "").strip()
    features = _split_columns(params.get("features"))
    if not target or not features:
        raise ValueError("A target column and at least one feature column are required.")
    subset = _columns(frame, [target] + features, "columns").dropna()
    if len(subset) < 10:
        raise ValueError("At least 10 complete rows are required for a train/test analysis.")
    encoded = pd.get_dummies(subset[features], drop_first=False).fillna(0)
    target_values = subset[target]
    if target_values.nunique() < 2:
        raise ValueError(f"Column '{target}' must contain at least two classes.")
    positive = _positive_label(target_values)
    seed = _number(params.get("seed"), 42, integer=True)
    test_size = _number(params.get("test_size"), 0.25, 0.1, 0.5)
    counts = target_values.value_counts()
    stratify = target_values if counts.min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        encoded, target_values, test_size=test_size, random_state=seed, stratify=stratify
    )
    model_key = _key(params.get("model"))
    if model_key in ("logistic", "logisticregression", "lr"):
        model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
        model_name = "Logistic Regression"
    else:
        model = GradientBoostingClassifier(n_estimators=120, random_state=seed)
        model_name = "Gradient Boosting"
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probability = _probability(model, X_test, positive)
    binary_actual = (y_test == positive).astype(int)
    summary_rows = [
        ("Model", model_name),
        ("Target", target),
        ("Positive class", str(positive)),
        ("Training rows", len(X_train)),
        ("Test rows", len(X_test)),
        ("Features used", encoded.shape[1]),
        ("Accuracy", round(float(accuracy_score(y_test, predictions)), 4)),
        ("Precision", round(float(precision_score(binary_actual, (predictions == positive).astype(int), zero_division=0)), 4)),
        ("Recall", round(float(recall_score(binary_actual, (predictions == positive).astype(int), zero_division=0)), 4)),
        ("F1 Score", round(float(f1_score(binary_actual, (predictions == positive).astype(int), zero_division=0)), 4)),
    ]
    if probability is not None and binary_actual.nunique() == 2:
        summary_rows.extend([
            ("ROC AUC", round(float(roc_auc_score(binary_actual, probability)), 4)),
            ("Average Precision", round(float(average_precision_score(binary_actual, probability)), 4)),
        ])
    summary = pd.DataFrame(summary_rows, columns=["Metric", "Value"])
    report_frame = pd.DataFrame(classification_report(y_test, predictions, output_dict=True, zero_division=0)).transpose().reset_index()
    report_frame = report_frame.rename(columns={"index": "Class"})
    confusion, matrix, labels = _confusion_frame(y_test, predictions)
    effects = _feature_effects(model, list(encoded.columns))
    predictions_frame = pd.DataFrame({
        "Row": X_test.index,
        "Actual": y_test.astype(str).values,
        "Predicted": pd.Series(predictions).astype(str).values,
        "Probability_Positive": np.round(probability, 4) if probability is not None else np.nan,
        "Correct": np.asarray(y_test) == np.asarray(predictions),
    })
    figures = [
        _figure_from_matrix(matrix, [f"Actual {label}" for label in labels], [f"Predicted {label}" for label in labels], f"{ANALYSIS_TITLES[(domain, technique)]} - Confusion Matrix")
    ]
    top_effects = effects.head(12).iloc[::-1]
    effect_figure, effect_axis = plt.subplots(figsize=(7, 5))
    effect_axis.barh(top_effects["Feature"], top_effects["Effect"], color="#4c78a8")
    effect_axis.axvline(0, color="black", linewidth=0.8)
    effect_axis.set_title("Model Feature Effects")
    effect_axis.set_xlabel("Coefficient or importance")
    effect_figure.tight_layout()
    figures.append(effect_figure)
    if probability is not None:
        probability_figure, probability_axis = plt.subplots(figsize=(7, 4))
        for label in sorted(set(y_test.astype(str))):
            mask = y_test.astype(str) == label
            probability_axis.hist(probability[mask], bins=15, alpha=0.6, label=str(label))
        probability_axis.axvline(0.5, color="black", linestyle="--", linewidth=1)
        probability_axis.set_xlabel(f"Probability of {positive}")
        probability_axis.set_ylabel("Count")
        probability_axis.set_title("Predicted Probability by Actual Class")
        probability_axis.legend()
        probability_figure.tight_layout()
        figures.append(probability_figure)
        if binary_actual.nunique() == 2:
            false_positive, true_positive, _ = roc_curve(binary_actual, probability)
            roc_figure, roc_axis = plt.subplots(figsize=(6, 5))
            roc_axis.plot(false_positive, true_positive, marker=".", label="Model")
            roc_axis.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
            roc_axis.set_xlabel("False positive rate")
            roc_axis.set_ylabel("True positive rate")
            roc_axis.set_title("ROC Curve")
            roc_axis.legend()
            roc_figure.tight_layout()
            figures.append(roc_figure)
    report = {
        "Summary": summary,
        "Classification Report": report_frame,
        "Confusion Matrix": confusion,
        "Feature Effects": effects,
        "Test Set Predictions": predictions_frame,
    }
    if technique == "attrition" and probability is not None:
        bands = pd.cut(probability, bins=[-0.01, 0.33, 0.66, 1.01], labels=["Low", "Medium", "High"])
        report["Attrition Risk Bands"] = pd.DataFrame({"Risk Band": bands.value_counts().index.astype(str), "Employees": bands.value_counts().values})
    return {"report": report, "figures": figures}


def _build_fraud(frame, params):
    features = _split_columns(params.get("features"))
    target = str(params.get("target") or "").strip()
    if target in frame.columns and target in features:
        features = [feature for feature in features if feature != target]
    if not features:
        features = _numeric_columns(frame, exclude=[target])
    if not features:
        raise ValueError("At least one numeric feature column is required.")
    subset = _columns(frame, features, "feature columns").dropna()
    if len(subset) < 10:
        raise ValueError("At least 10 complete rows are required for anomaly detection.")
    contamination = _number(params.get("contamination"), 0.08, 0.01, 0.5)
    seed = _number(params.get("seed"), 42, integer=True)
    scaled = StandardScaler().fit_transform(subset)
    method = _key(params.get("model"))
    if method in ("local_outlier", "localoutlierfactor", "lof"):
        neighbours = min(20, max(2, len(subset) - 1))
        detector = LocalOutlierFactor(n_neighbors=neighbours, contamination=contamination)
        flags = detector.fit_predict(scaled) < 0
        scores = -detector.negative_outlier_factor_
        method_name = "Local Outlier Factor"
    elif method in ("one_class_svm", "oneclasssvm", "ocsvm"):
        detector = OneClassSVM(nu=contamination, gamma="scale")
        flags = detector.fit_predict(scaled) < 0
        scores = -detector.decision_function(scaled)
        method_name = "One-Class SVM"
    else:
        detector = IsolationForest(contamination=contamination, random_state=seed)
        flags = detector.fit_predict(scaled) < 0
        scores = -detector.score_samples(scaled)
        method_name = "Isolation Forest"
    threshold = float(np.quantile(scores, 1 - contamination))
    scored = subset.copy()
    scored.insert(0, "Row", subset.index)
    scored["Anomaly Score"] = np.round(scores, 4)
    scored["Flagged"] = flags
    top_n = _number(params.get("top_n"), 20, 1, 200, integer=True)
    flagged_frame = scored.loc[scored["Flagged"]].sort_values("Anomaly Score", ascending=False).head(top_n)
    summary_rows = [
        ("Method", method_name),
        ("Rows scored", len(scored)),
        ("Features used", len(features)),
        ("Expected anomaly rate", contamination),
        ("Flagged rows", int(flags.sum())),
        ("Flagged percentage", round(float(flags.mean() * 100), 2)),
        ("Score threshold", round(threshold, 4)),
    ]
    if target in frame.columns:
        actual = frame.loc[scored["Row"], target].astype(str)
        positive = _positive_label(actual)
        actual_flags = actual == positive
        summary_rows.extend([
            ("Labelled rows", len(actual)),
            ("Label precision", round(float(precision_score(actual_flags, flags, zero_division=0)), 4)),
            ("Label recall", round(float(recall_score(actual_flags, flags, zero_division=0)), 4)),
            ("Label F1", round(float(f1_score(actual_flags, flags, zero_division=0)), 4)),
        ])
    summary = pd.DataFrame(summary_rows, columns=["Metric", "Value"])
    score_figure, score_axis = plt.subplots(figsize=(7, 4))
    score_axis.hist(scores, bins=25, color="#e45756", alpha=0.75)
    score_axis.axvline(threshold, color="black", linestyle="--", label="Threshold")
    score_axis.set_xlabel("Anomaly score")
    score_axis.set_ylabel("Count")
    score_axis.set_title("Anomaly Score Distribution")
    score_axis.legend()
    score_figure.tight_layout()
    figures = [score_figure]
    if len(features) >= 2:
        scatter_figure, scatter_axis = plt.subplots(figsize=(7, 5))
        colours = np.where(flags, "#e45756", "#4c78a8")
        scatter_axis.scatter(subset.iloc[:, 0], subset.iloc[:, 1], c=colours, alpha=0.7)
        scatter_axis.set_xlabel(features[0])
        scatter_axis.set_ylabel(features[1])
        scatter_axis.set_title("Flagged Observations")
        scatter_figure.tight_layout()
        figures.append(scatter_figure)
    count_figure, count_axis = plt.subplots(figsize=(5, 4))
    count_axis.bar(["Normal", "Flagged"], [len(scored) - int(flags.sum()), int(flags.sum())], color=["#4c78a8", "#e45756"])
    count_axis.set_ylabel("Rows")
    count_axis.set_title("Detection Counts")
    count_figure.tight_layout()
    figures.append(count_figure)
    return {"report": {"Summary": summary, "Flagged Observations": flagged_frame, "Scored Data": scored}, "figures": figures}


def _build_portfolio(frame, params):
    returns_columns = _split_columns(params.get("returns"))
    if not returns_columns:
        returns_columns = _numeric_columns(frame)
    if not returns_columns:
        raise ValueError("At least one numeric return column is required.")
    data = _columns(frame, returns_columns, "return columns").dropna()
    if len(data) < 10:
        raise ValueError("At least 10 complete return observations are required.")
    confidence = _number(params.get("confidence"), 0.95, 0.9, 0.99)
    weight_values = _split_columns(params.get("weights"))
    if weight_values:
        if len(weight_values) != len(returns_columns):
            raise ValueError("The number of weights must match the number of return columns.")
        weights = np.asarray(weight_values, dtype=float)
    else:
        weights = np.repeat(1.0 / len(returns_columns), len(returns_columns))
    if not np.isfinite(weights).all() or weights.sum() == 0:
        raise ValueError("Portfolio weights must be finite and not all zero.")
    weights = weights / weights.sum()
    portfolio = data.to_numpy(dtype=float) @ weights
    annualising = np.sqrt(252)
    quantile = float(np.quantile(portfolio, 1 - confidence))
    tail = portfolio[portfolio <= quantile]
    wealth = np.cumprod(1 + portfolio)
    drawdowns = wealth / np.maximum.accumulate(wealth) - 1
    covariance = data.cov().to_numpy(dtype=float)
    volatility = float(np.std(portfolio, ddof=1))
    contributions = weights * (covariance @ weights) / volatility if volatility else np.zeros_like(weights)
    summary_rows = [
        ("Confidence level", confidence),
        ("Observations", len(portfolio)),
        ("Mean daily return", round(float(np.mean(portfolio)), 6)),
        ("Daily volatility", round(volatility, 6)),
        ("Annualised return", round(float(np.mean(portfolio) * 252), 4)),
        ("Annualised volatility", round(float(volatility * annualising), 4)),
        ("Sharpe ratio (rf=0)", round(float(np.mean(portfolio) / volatility), 4) if volatility else np.nan),
        (f"Historical VaR {confidence:.0%}", round(float(-quantile), 6)),
        (f"Historical CVaR {confidence:.0%}", round(float(-tail.mean()), 6) if len(tail) else np.nan),
        ("Parametric VaR", round(float(-(np.mean(portfolio) + 1.6448536269514722 * volatility)), 6)),
        ("Maximum drawdown", round(float(drawdowns.min()), 6)),
        ("Worst day", round(float(portfolio.min()), 6)),
        ("Best day", round(float(portfolio.max()), 6)),
    ]
    contribution_frame = pd.DataFrame({
        "Asset": returns_columns,
        "Weight": np.round(weights, 4),
        "Risk Contribution": np.round(contributions, 6),
        "Risk Share": np.round(contributions / contributions.sum(), 4) if contributions.sum() else np.nan,
    })
    worst_days = pd.DataFrame({
        "Row": data.index[np.argsort(portfolio)[:10]],
        "Portfolio Return": np.round(np.sort(portfolio)[:10], 6),
    })
    figures = []
    wealth_figure, wealth_axis = plt.subplots(figsize=(8, 4))
    wealth_axis.plot(data.index, wealth, color="#4c78a8")
    wealth_axis.set_title("Cumulative Portfolio Value")
    wealth_axis.set_xlabel("Observation")
    wealth_axis.set_ylabel("Growth of 1 unit")
    wealth_figure.autofmt_xdate()
    wealth_figure.tight_layout()
    figures.append(wealth_figure)
    distribution_figure, distribution_axis = plt.subplots(figsize=(7, 4))
    distribution_axis.hist(portfolio, bins=30, color="#72b7b2", alpha=0.8)
    distribution_axis.axvline(quantile, color="#e45756", linestyle="--", label=f"VaR {confidence:.0%}")
    distribution_axis.set_xlabel("Portfolio return")
    distribution_axis.set_ylabel("Count")
    distribution_axis.set_title("Return Distribution and Value at Risk")
    distribution_axis.legend()
    distribution_figure.tight_layout()
    figures.append(distribution_figure)
    drawdown_figure, drawdown_axis = plt.subplots(figsize=(8, 3))
    drawdown_axis.fill_between(data.index, drawdowns, 0, color="#e45756", alpha=0.6)
    drawdown_axis.set_title("Portfolio Drawdown")
    drawdown_axis.set_xlabel("Observation")
    drawdown_axis.set_ylabel("Drawdown")
    drawdown_figure.autofmt_xdate()
    drawdown_figure.tight_layout()
    figures.append(drawdown_figure)
    correlation = data.corr()
    correlation_figure, correlation_axis = plt.subplots(figsize=(6, 5))
    image = correlation_axis.imshow(correlation.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
    correlation_axis.set_xticks(range(len(returns_columns)))
    correlation_axis.set_yticks(range(len(returns_columns)))
    correlation_axis.set_xticklabels(returns_columns, rotation=30, ha="right")
    correlation_axis.set_yticklabels(returns_columns)
    correlation_figure.colorbar(image, ax=correlation_axis, fraction=0.046, pad=0.04)
    correlation_axis.set_title("Asset Return Correlation")
    correlation_figure.tight_layout()
    figures.append(correlation_figure)
    return {
        "report": {
            "Summary": pd.DataFrame(summary_rows, columns=["Metric", "Value"]),
            "Risk Contributions": contribution_frame,
            "Correlation Matrix": correlation,
            "Worst Days": worst_days,
        },
        "figures": figures,
    }


def _build_pay_equity(frame, params):
    salary = str(params.get("salary") or "").strip()
    group = str(params.get("group") or "").strip()
    role = str(params.get("role") or "").strip()
    performance = [column for column in _split_columns(params.get("performance")) if column not in (salary, group, role)]
    columns = [column for column in [salary, group, role] + performance if column]
    if not salary or not group or any(column not in frame.columns for column in columns):
        raise ValueError("A valid salary column, group column and optional role/performance columns are required.")
    data = _columns(frame, columns).dropna()
    if len(data) < 10 or data[group].nunique() < 2:
        raise ValueError("At least 10 complete rows and two groups are required for pay equity analysis.")
    data = data[data[salary].apply(lambda value: isinstance(value, (int, float, np.integer, np.floating)))]
    if len(data) < 10:
        raise ValueError("The salary column must contain numeric values.")
    group_means = data.groupby(group)[salary].agg(["count", "mean", "median", "std"]).round(3)
    reference = sorted(group_means.index, key=str)[0]
    comparison = group_means.copy()
    comparison["Difference vs Reference"] = (comparison["mean"] - group_means.loc[reference, "mean"]).round(3)
    comparison["Percent vs Reference"] = ((comparison["mean"] / group_means.loc[reference, "mean"] - 1) * 100).round(2)
    predictors = data.drop(columns=[salary]).copy()
    predictors = pd.get_dummies(predictors, drop_first=True).astype(float)
    predictors = sm.add_constant(predictors, has_constant="add")
    model = sm.OLS(data[salary].astype(float), predictors).fit()
    intervals = model.conf_int()
    coefficients = pd.DataFrame({
        "Coefficient": model.params,
        "Std Error": model.bse,
        "P Value": model.pvalues,
        "CI Low": intervals[0],
        "CI High": intervals[1],
    }).round(5).reset_index().rename(columns={"index": "Term"})
    group_terms = coefficients[coefficients["Term"].astype(str).str.startswith(group + "_")]
    summary_rows = [
        ("Rows analysed", len(data)),
        ("Groups", data[group].nunique()),
        ("Reference group", str(reference)),
        ("Unadjusted mean pay gap", round(float(comparison["Difference vs Reference"].drop(reference).abs().max()), 3)),
        ("Adjusted group terms", len(group_terms)),
        ("Model R-squared", round(float(model.rsquared), 4)),
        ("Model Adj. R-squared", round(float(model.rsquared_adj), 4)),
    ]
    figures = []
    box_figure, box_axis = plt.subplots(figsize=(7, 4))
    ordered_groups = [str(value) for value in group_means.index]
    box_data = [data.loc[data[group] == value, salary].to_numpy(dtype=float) for value in group_means.index]
    box_axis.boxplot(box_data, tick_labels=ordered_groups)
    box_axis.set_ylabel(salary)
    box_axis.set_xlabel(group)
    box_axis.set_title("Salary Distribution by Group")
    box_figure.tight_layout()
    figures.append(box_figure)
    residual_figure, residual_axis = plt.subplots(figsize=(7, 4))
    residual_axis.hist(model.resid, bins=20, color="#4c78a8", alpha=0.8)
    residual_axis.axvline(0, color="black", linestyle="--")
    residual_axis.set_title("Model Residuals")
    residual_axis.set_xlabel("Residual")
    residual_figure.tight_layout()
    figures.append(residual_figure)
    plot_terms = coefficients.iloc[1:].head(12).iloc[::-1]
    coefficient_figure, coefficient_axis = plt.subplots(figsize=(8, 5))
    coefficient_axis.errorbar(
        plot_terms["Coefficient"],
        plot_terms["Term"],
        xerr=(plot_terms["CI High"] - plot_terms["CI Low"]).abs() / 2,
        fmt="o",
        color="#4c78a8",
        linestyle="none",
    )
    coefficient_axis.axvline(0, color="black", linewidth=0.8)
    coefficient_axis.set_title("Adjusted Coefficients with 95% Confidence Intervals")
    coefficient_figure.tight_layout()
    figures.append(coefficient_figure)
    return {
        "report": {
            "Summary": pd.DataFrame(summary_rows, columns=["Metric", "Value"]),
            "Group Comparison": comparison.reset_index().rename(columns={"index": group}),
            "Adjusted Model Coefficients": coefficients,
            "Group Effects": group_terms,
        },
        "figures": figures,
    }


def _build_segmentation(frame, params):
    features = _split_columns(params.get("features"))
    if not features:
        features = _numeric_columns(frame)
    if not features:
        raise ValueError("At least one numeric feature column is required.")
    data = _columns(frame, features, "feature columns").dropna()
    if len(data) < 10:
        raise ValueError("At least 10 complete rows are required for clustering.")
    seed = _number(params.get("seed"), 42, integer=True)
    clusters = _number(params.get("clusters"), 3, 2, min(10, len(data) - 1), integer=True)
    scaled = StandardScaler().fit_transform(data)
    model = KMeans(n_clusters=clusters, n_init=10, random_state=seed)
    labels = model.fit_predict(scaled)
    score = silhouette_score(scaled, labels) if clusters > 1 else np.nan
    profiles = data.assign(Cluster=labels).groupby("Cluster").mean(numeric_only=True).round(3)
    counts = pd.Series(labels).value_counts().sort_index().rename("Employees").rename_axis("Cluster").reset_index()
    assignments = data.copy()
    assignments.insert(0, "Row", data.index)
    assignments.insert(1, "Cluster", labels)
    component_count = min(2, scaled.shape[1])
    reduced = PCA(n_components=component_count, random_state=seed).fit_transform(scaled)
    scatter_figure, scatter_axis = plt.subplots(figsize=(7, 5))
    for cluster in range(clusters):
        mask = labels == cluster
        scatter_axis.scatter(reduced[mask, 0], reduced[mask, 1], label=f"Cluster {cluster}", alpha=0.75)
    scatter_axis.set_xlabel("PC1")
    scatter_axis.set_ylabel("PC2" if component_count > 1 else "")
    scatter_axis.set_title("Employee Segments (PCA projection)")
    scatter_axis.legend()
    scatter_figure.tight_layout()
    heatmap_figure, heatmap_axis = plt.subplots(figsize=(8, 4))
    image = heatmap_axis.imshow(profiles.to_numpy(), aspect="auto", cmap="viridis")
    heatmap_axis.set_xticks(range(len(profiles.columns)))
    heatmap_axis.set_yticks(range(len(profiles.index)))
    heatmap_axis.set_xticklabels(profiles.columns, rotation=30, ha="right")
    heatmap_axis.set_yticklabels([f"Cluster {index}" for index in profiles.index])
    heatmap_figure.colorbar(image, ax=heatmap_axis, fraction=0.046, pad=0.04)
    heatmap_axis.set_title("Cluster Profiles (standardised feature means are computed from original values)")
    heatmap_figure.tight_layout()
    elbow_values = range(1, min(9, len(data)) + 1)
    inertia = [KMeans(n_clusters=value, n_init=10, random_state=seed).fit(scaled).inertia_ for value in elbow_values]
    elbow_figure, elbow_axis = plt.subplots(figsize=(7, 4))
    elbow_axis.plot(list(elbow_values), inertia, marker="o", color="#4c78a8")
    elbow_axis.axvline(clusters, color="#e45756", linestyle="--", label="Selected k")
    elbow_axis.set_xlabel("Number of clusters")
    elbow_axis.set_ylabel("Inertia")
    elbow_axis.set_title("Elbow Curve")
    elbow_axis.legend()
    elbow_figure.tight_layout()
    return {
        "report": {
            "Summary": pd.DataFrame([
                ("Rows clustered", len(data)),
                ("Features used", len(features)),
                ("Clusters", clusters),
                ("Silhouette score", round(float(score), 4)),
            ], columns=["Metric", "Value"]),
            "Cluster Counts": counts,
            "Cluster Profiles": profiles.reset_index().rename(columns={"index": "Cluster"}),
            "Assignments": assignments.head(30),
        },
        "figures": [scatter_figure, heatmap_figure, elbow_figure],
    }


def _fit_series_forecast(series, steps, seasonal_periods, model_key):
    values = np.asarray(series.astype(float), dtype=float)
    if _key(model_key) == "arima":
        try:
            fit = ARIMA(values, order=(1, 0, 0), trend="c").fit()
            forecast = np.asarray(fit.forecast(steps=steps), dtype=float)
            intervals = np.asarray(fit.get_forecast(steps=steps).conf_int(alpha=0.05), dtype=float)
            return forecast, intervals[:, 0], intervals[:, 1], "ARIMA(1,0,0)"
        except Exception:
            pass
    for seasonal_mode in ("add", None):
        try:
            options = {"trend": "add"}
            if seasonal_mode == "add":
                options["seasonal"] = "add"
                options["seasonal_periods"] = max(2, min(int(seasonal_periods), max(2, len(values) // 2)))
            fit = ExponentialSmoothing(values, **options).fit()
            forecast = np.asarray(fit.forecast(steps), dtype=float)
            residuals = values - np.asarray(fit.fittedvalues, dtype=float)
            sigma = float(np.nanstd(residuals)) or float(np.std(np.diff(values))) if len(values) > 1 else 1.0
            width = 1.96 * sigma * np.sqrt(np.arange(1, steps + 1))
            return forecast, forecast - width, forecast + width, ("Holt-Winters" if seasonal_mode else "Holt's linear trend")
        except Exception:
            continue
    drift = float(np.mean(np.diff(values))) if len(values) > 1 else 0.0
    forecast = np.asarray([values[-1] + drift * (step + 1) for step in range(steps)], dtype=float)
    sigma = float(np.std(np.diff(values))) if len(values) > 1 else 1.0
    width = 1.96 * sigma * np.sqrt(np.arange(1, steps + 1))
    return forecast, forecast - width, forecast + width, "Naive drift fallback"


def _build_forecast(frame, params, domain, technique):
    value = str(params.get("value") or "").strip()
    period = str(params.get("period") or "").strip()
    filter_column = str(params.get("filter_column") or "").strip()
    filter_value = str(params.get("filter_value") or "").strip()
    data = frame.copy()
    if filter_column and filter_value and filter_column in data.columns:
        data = data[data[filter_column].astype(str).str.strip() == filter_value.strip()]
    if value not in data.columns:
        raise ValueError(f"Value column '{value}' was not found.")
    series = pd.to_numeric(data[value], errors="coerce").dropna()
    if len(series) < 24:
        raise ValueError("At least 24 complete observations are required for forecasting.")
    periods = _number(params.get("periods"), 6, 1, 36, integer=True)
    seasonal = _number(params.get("seasonal_periods"), 12, 2, max(2, len(series) // 2), integer=True)
    model_key = str(params.get("model") or "holt_winters").strip()
    split = max(12, len(series) - periods)
    train, test = series.iloc[:split], series.iloc[split:]
    backcast, _, _, model_name = _fit_series_forecast(train, len(test), seasonal, model_key)
    errors = np.asarray(test, dtype=float) - np.asarray(backcast, dtype=float)
    mape_values = np.abs(errors / np.where(np.asarray(test, dtype=float) == 0, np.nan, np.asarray(test, dtype=float))) * 100
    mape = float(np.nanmean(mape_values)) if np.isfinite(mape_values).any() else np.nan
    forecast, lower, upper, model_name = _fit_series_forecast(series, periods, seasonal, model_key)
    forecast_frame = pd.DataFrame({
        "Step": np.arange(1, periods + 1),
        "Forecast": np.round(forecast, 4),
        "Lower Bound": np.round(lower, 4),
        "Upper Bound": np.round(upper, 4),
    })
    backtest_frame = pd.DataFrame({
        "Row": test.index,
        "Actual": np.asarray(test, dtype=float),
        "Predicted": np.round(np.asarray(backcast, dtype=float), 4),
        "Error": np.round(errors, 4),
    })
    summary_rows = [
        ("Technique", ANALYSIS_TITLES[(domain, technique)]),
        ("Model", model_name),
        ("Observations", len(series)),
        ("Backtest rows", len(test)),
        ("Forecast periods", periods),
        ("MAE", round(float(mean_absolute_error(test, backcast)), 4)),
        ("RMSE", round(float(np.sqrt(mean_squared_error(test, backcast))), 4)),
        ("MAPE (%)", round(mape, 4) if np.isfinite(mape) else np.nan),
        ("Mean forecast", round(float(np.mean(forecast)), 4)),
        ("Forecast trend", round(float(forecast[-1] - forecast[0]), 4)),
    ]
    figures = []
    history_figure, history_axis = plt.subplots(figsize=(9, 5))
    history_axis.plot(series.index, series.to_numpy(), label="Actual", color="#4c78a8")
    history_axis.plot(test.index, backcast, linestyle="--", label="Backtest", color="#f58518")
    future_index = np.arange(len(series), len(series) + periods)
    history_axis.plot(future_index, forecast, marker="o", label="Forecast", color="#e45756")
    history_axis.fill_between(future_index, lower, upper, color="#e45756", alpha=0.2, label="95% interval")
    history_axis.axvline(len(series) - 0.5, color="grey", linestyle=":")
    history_axis.set_title(f"{ANALYSIS_TITLES[(domain, technique)]}: Actual, Backtest and Forecast")
    history_axis.set_xlabel("Observation" if not period else f"Observation ({period})")
    history_axis.set_ylabel(value)
    history_axis.legend()
    history_figure.tight_layout()
    figures.append(history_figure)
    residual_figure, residual_axis = plt.subplots(figsize=(7, 4))
    residual_axis.hist(errors, bins=15, color="#72b7b2", alpha=0.8)
    residual_axis.axvline(0, color="black", linestyle="--")
    residual_axis.set_title("Backtest Errors")
    residual_axis.set_xlabel("Error")
    residual_figure.tight_layout()
    figures.append(residual_figure)
    forecast_figure, forecast_axis = plt.subplots(figsize=(8, 4))
    forecast_axis.bar(forecast_frame["Step"], forecast_frame["Forecast"], color="#54a24b")
    forecast_axis.set_xlabel("Forecast step")
    forecast_axis.set_ylabel(value)
    forecast_axis.set_title("Forecast Values")
    forecast_figure.tight_layout()
    figures.append(forecast_figure)
    return {
        "report": {
            "Summary": pd.DataFrame(summary_rows, columns=["Metric", "Value"]),
            "Forecast": forecast_frame,
            "Backtest": backtest_frame,
            "Model": pd.DataFrame([("Name", model_name), ("Seasonal periods", seasonal), ("Value column", value)], columns=["Setting", "Value"]),
        },
        "figures": figures,
    }


def _binary_values(series):
    return series.astype(str).str.strip().str.lower().isin(["yes", "y", "1", "true", "approved", "approve", "pass"]).astype(int)


def _build_fair_lending(frame, params):
    decision = str(params.get("decision") or "").strip()
    group = str(params.get("group") or "").strip()
    target = str(params.get("target") or "").strip()
    controls = [column for column in _split_columns(params.get("features")) if column not in (decision, group, target)]
    columns = [column for column in [decision, group, target] + controls if column]
    if not decision or not group or any(column not in frame.columns for column in columns):
        raise ValueError("A valid approval decision column, group column and optional controls are required.")
    data = _columns(frame, columns).dropna()
    if len(data) < 20 or data[group].nunique() < 2:
        raise ValueError("At least 20 complete rows and two groups are required for a fair lending audit.")
    approved = _binary_values(data[decision])
    analysis = data.copy()
    analysis["_Approved"] = approved
    selection = analysis.groupby(group)["_Approved"].agg(["count", "mean"]).round(4)
    selection["Selection Rate (%)"] = (selection["mean"] * 100).round(2)
    selection = selection.rename(columns={"count": "Applicants", "mean": "Selection Rate"})
    highest = float(selection["Selection Rate"].max())
    lowest = float(selection["Selection Rate"].min())
    impact_ratio = round(lowest / highest, 4) if highest else np.nan
    summary_rows = [
        ("Applications", len(analysis)),
        ("Groups", analysis[group].nunique()),
        ("Overall approval rate", round(float(approved.mean()), 4)),
        ("Highest group approval rate", round(highest, 4)),
        ("Lowest group approval rate", round(lowest, 4)),
        ("Adverse impact ratio", impact_ratio),
        ("Four-fifths rule flag", "Review" if impact_ratio < 0.8 else "Pass"),
    ]
    figures = []
    selection_figure, selection_axis = plt.subplots(figsize=(8, 4))
    selection_axis.bar([str(index) for index in selection.index], selection["Selection Rate (%)"], color="#4c78a8")
    selection_axis.axhline(selection["Selection Rate (%)"].mean(), color="black", linestyle="--", label="Overall")
    selection_axis.set_ylabel("Approval rate (%)")
    selection_axis.set_title("Approval Rate by Group")
    selection_axis.legend()
    selection_figure.tight_layout()
    figures.append(selection_figure)
    coefficients = pd.DataFrame()
    if controls:
        predictors = pd.get_dummies(analysis[[group] + controls], drop_first=True).astype(float)
        predictors = sm.add_constant(predictors, has_constant="add")
        model = sm.OLS(approved.astype(float), predictors).fit()
        intervals = model.conf_int()
        coefficients = pd.DataFrame({
            "Coefficient": model.params,
            "P Value": model.pvalues,
            "CI Low": intervals[0],
            "CI High": intervals[1],
        }).round(5).reset_index().rename(columns={"index": "Term"})
        summary_rows.append(("Adjusted model R-squared", round(float(model.rsquared), 4)))
        plot_terms = coefficients[coefficients["Term"].astype(str).str.startswith(group + "_")].head(10)
        if not plot_terms.empty:
            coefficient_figure, coefficient_axis = plt.subplots(figsize=(8, 4))
            coefficient_axis.errorbar(
                plot_terms["Coefficient"],
                plot_terms["Term"],
                xerr=(plot_terms["CI High"] - plot_terms["CI Low"]).abs() / 2,
                fmt="o",
                color="#e45756",
                linestyle="none",
            )
            coefficient_axis.axvline(0, color="black", linewidth=0.8)
            coefficient_axis.set_title("Adjusted Group Effects")
            coefficient_figure.tight_layout()
            figures.append(coefficient_figure)
    outcome = pd.DataFrame()
    if target in frame.columns:
        analysis["_Outcome"] = _binary_values(analysis[target])
        outcome = analysis.groupby(group)["_Outcome"].agg(["count", "mean"]).round(4).rename(columns={"count": "Rows", "mean": "Outcome Rate"})
        outcome_figure, outcome_axis = plt.subplots(figsize=(8, 4))
        outcome_axis.bar([str(index) for index in outcome.index], outcome["Outcome Rate"] * 100, color="#f58518")
        outcome_axis.set_ylabel("Outcome rate (%)")
        outcome_axis.set_title(f"Outcome Rate by Group ({target})")
        outcome_figure.tight_layout()
        figures.append(outcome_figure)
    report = {
        "Summary": pd.DataFrame(summary_rows, columns=["Metric", "Value"]),
        "Group Approval Rates": selection.reset_index().rename(columns={"index": group}),
        "Adjusted Coefficients": coefficients if not coefficients.empty else pd.DataFrame({"Term": ["No controls supplied"], "Coefficient": [np.nan]}),
    }
    if not outcome.empty:
        report["Group Outcome Rates"] = outcome.reset_index().rename(columns={"index": group})
    return {"report": report, "figures": figures}


def _build_inventory(frame, params):
    demand = str(params.get("demand") or "").strip()
    if demand not in frame.columns:
        raise ValueError(f"Demand column '{demand}' was not found.")
    series = pd.to_numeric(frame[demand], errors="coerce").dropna()
    if len(series) < 20:
        raise ValueError("At least 20 demand observations are required.")
    lead_time = _number(params.get("lead_time"), 7, 1, 365, integer=True)
    service_level = _number(params.get("service_level"), 0.95, 0.85, 0.99)
    scenarios = _number(params.get("scenarios"), 1000, 100, 100000, integer=True)
    seed = _number(params.get("seed"), 42, integer=True)
    mean_daily = float(series.mean())
    sigma_daily = float(series.std(ddof=1))
    z_score = float(stats.norm.ppf(service_level))
    safety_stock = z_score * sigma_daily * np.sqrt(lead_time)
    reorder_point = mean_daily * lead_time + safety_stock
    rng = np.random.default_rng(seed)
    lead_demand = np.maximum(rng.normal(mean_daily * lead_time, sigma_daily * np.sqrt(lead_time), scenarios), 0)
    shortage = np.maximum(lead_demand - reorder_point, 0)
    stockout_probability = float(np.mean(lead_demand > reorder_point))
    sensitivity = pd.DataFrame({
        "Service Level": [0.85, 0.90, 0.95, 0.975, 0.99],
        "Reorder Point": [mean_daily * lead_time + float(stats.norm.ppf(level)) * sigma_daily * np.sqrt(lead_time) for level in (0.85, 0.90, 0.95, 0.975, 0.99)],
    }).round(3)
    scenario_frame = pd.DataFrame({"Scenario": np.arange(1, scenarios + 1), "Lead Time Demand": np.round(lead_demand, 3), "Shortage": np.round(shortage, 3)}).head(30)
    figures = []
    distribution_figure, distribution_axis = plt.subplots(figsize=(8, 4))
    distribution_axis.hist(lead_demand, bins=35, color="#4c78a8", alpha=0.8)
    distribution_axis.axvline(reorder_point, color="#e45756", linestyle="--", label="Reorder point")
    distribution_axis.set_xlabel("Demand during lead time")
    distribution_axis.set_ylabel("Scenarios")
    distribution_axis.set_title("Lead-Time Demand and Reorder Point")
    distribution_axis.legend()
    distribution_figure.tight_layout()
    figures.append(distribution_figure)
    shortage_figure, shortage_axis = plt.subplots(figsize=(7, 4))
    shortage_axis.hist(shortage, bins=30, color="#e45756", alpha=0.8)
    shortage_axis.set_xlabel("Shortage")
    shortage_axis.set_ylabel("Scenarios")
    shortage_axis.set_title("Simulated Shortage Distribution")
    shortage_figure.tight_layout()
    figures.append(shortage_figure)
    sensitivity_figure, sensitivity_axis = plt.subplots(figsize=(7, 4))
    sensitivity_axis.plot(sensitivity["Service Level"], sensitivity["Reorder Point"], marker="o", color="#54a24b")
    sensitivity_axis.set_xlabel("Service level")
    sensitivity_axis.set_ylabel("Reorder point")
    sensitivity_axis.set_title("Service Level Sensitivity")
    sensitivity_figure.tight_layout()
    figures.append(sensitivity_figure)
    return {
        "report": {
            "Summary": pd.DataFrame([
                ("Observations", len(series)),
                ("Mean daily demand", round(mean_daily, 3)),
                ("Daily demand std. dev.", round(sigma_daily, 3)),
                ("Lead time (days)", lead_time),
                ("Service level", service_level),
                ("Cycle stock", round(mean_daily * lead_time, 3)),
                ("Safety stock", round(safety_stock, 3)),
                ("Reorder point", round(reorder_point, 3)),
                ("Simulated stockout probability", round(stockout_probability, 4)),
                ("Expected shortage", round(float(shortage.mean()), 3)),
                ("Scenarios", scenarios),
            ], columns=["Metric", "Value"]),
            "Service Level Sensitivity": sensitivity,
            "Scenario Sample": scenario_frame,
        },
        "figures": figures,
    }


def _build_supplier_risk(frame, params):
    features = _split_columns(params.get("features"))
    group = str(params.get("group") or "").strip()
    if not features:
        features = _numeric_columns(frame, exclude=[group])
    if not features or group not in frame.columns:
        raise ValueError("Risk features and a grouping column such as Country are required.")
    data = _columns(frame, features + [group]).dropna()
    if len(data) < 10:
        raise ValueError("At least 10 complete supplier records are required.")
    clusters = _number(params.get("clusters"), 3, 2, min(8, len(data) - 1), integer=True)
    seed = _number(params.get("seed"), 42, integer=True)
    scaled = StandardScaler().fit_transform(data[features])
    model = KMeans(n_clusters=clusters, n_init=10, random_state=seed)
    labels = model.fit_predict(scaled)
    score = silhouette_score(scaled, labels) if len(data) > clusters else np.nan
    profiles = data[features].assign(Cluster=labels).groupby("Cluster").mean(numeric_only=True).round(3)
    graph = nx.Graph()
    supplier_nodes = [f"Supplier {position + 1}" for position in range(len(data))]
    group_nodes = [f"{group}: {value}" for value in data[group].astype(str).unique()]
    for node, cluster in zip(supplier_nodes, labels):
        graph.add_node(node, cluster=int(cluster), kind="supplier")
    for node in group_nodes:
        graph.add_node(node, kind=group)
    for supplier, group_value in zip(supplier_nodes, data[group].astype(str)):
        graph.add_edge(supplier, f"{group}: {group_value}")
    degree = nx.degree_centrality(graph)
    betweenness = nx.betweenness_centrality(graph)
    centrality = pd.DataFrame({
        "Node": supplier_nodes,
        "Cluster": labels,
        "Degree Centrality": [round(degree[node], 4) for node in supplier_nodes],
        "Betweenness Centrality": [round(betweenness[node], 4) for node in supplier_nodes],
    })
    figures = []
    network_figure, network_axis = plt.subplots(figsize=(9, 6))
    positions = nx.spring_layout(graph, seed=seed)
    colours = ["#e45756" if graph.nodes[node].get("kind") == "supplier" else "#4c78a8" for node in graph.nodes]
    nx.draw(graph, positions, ax=network_axis, node_color=colours, node_size=180, with_labels=False, alpha=0.85)
    network_axis.set_title("Supplier Dependency Network")
    network_figure.tight_layout()
    figures.append(network_figure)
    profile_figure, profile_axis = plt.subplots(figsize=(8, 4))
    image = profile_axis.imshow(profiles.to_numpy(), aspect="auto", cmap="viridis")
    profile_axis.set_xticks(range(len(profiles.columns)))
    profile_axis.set_yticks(range(len(profiles.index)))
    profile_axis.set_xticklabels(profiles.columns, rotation=30, ha="right")
    profile_axis.set_yticklabels([f"Cluster {index}" for index in profiles.index])
    profile_figure.colorbar(image, ax=profile_axis, fraction=0.046, pad=0.04)
    profile_axis.set_title("Supplier Risk Profiles")
    profile_figure.tight_layout()
    figures.append(profile_figure)
    assignments = data[features].copy()
    assignments.insert(0, "Row", data.index)
    assignments.insert(1, "Cluster", labels)
    return {
        "report": {
            "Summary": pd.DataFrame([
                ("Suppliers", len(data)),
                ("Features", len(features)),
                ("Clusters", clusters),
                ("Silhouette score", round(float(score), 4)),
                ("Network nodes", graph.number_of_nodes()),
                ("Network edges", graph.number_of_edges()),
            ], columns=["Metric", "Value"]),
            "Cluster Profiles": profiles.reset_index().rename(columns={"index": "Cluster"}),
            "Network Centrality": centrality,
            "Assignments": assignments.head(30),
        },
        "figures": figures,
    }


def _build_process_cycle(frame, params):
    case = str(params.get("case") or "").strip()
    activity = str(params.get("activity") or "").strip()
    timestamp = str(params.get("timestamp") or "").strip()
    if any(column not in frame.columns for column in (case, activity, timestamp)):
        raise ValueError("Case id, activity and timestamp columns are required.")
    data = frame[[case, activity, timestamp]].dropna().copy()
    data[timestamp] = pd.to_datetime(data[timestamp], errors="coerce")
    data = data.dropna(subset=[timestamp]).sort_values([case, timestamp])
    if data[case].nunique() < 3 or len(data) < 10:
        raise ValueError("At least three cases and ten events are required for process mining.")
    durations = data.groupby(case)[timestamp].agg(["min", "max"])
    durations["Cycle Hours"] = ((durations["max"] - durations["min"]).dt.total_seconds() / 3600).round(3)
    transitions = []
    for _, events in data.groupby(case):
        steps = list(events[activity].astype(str))
        transitions.extend(zip(steps, steps[1:]))
    transition_counts = pd.Series([f"{first} -> {second}" for first, second in transitions]).value_counts()
    top_n = _number(params.get("top_n"), 10, 1, 50, integer=True)
    transition_frame = transition_counts.head(top_n).rename("Count").rename_axis("Transition").reset_index()
    dwell = data.groupby(activity)[timestamp].apply(lambda values: values.sort_values().diff().dt.total_seconds().div(3600).mean()).round(3).rename("Avg Dwell Hours")
    figures = []
    transition_figure, transition_axis = plt.subplots(figsize=(9, 5))
    transition_axis.barh(transition_frame["Transition"][::-1], transition_frame["Count"][::-1], color="#4c78a8")
    transition_axis.set_xlabel("Count")
    transition_axis.set_title("Most Frequent Activity Transitions")
    transition_figure.tight_layout()
    figures.append(transition_figure)
    cycle_figure, cycle_axis = plt.subplots(figsize=(8, 4))
    cycle_axis.hist(durations["Cycle Hours"], bins=15, color="#54a24b", alpha=0.8)
    cycle_axis.set_xlabel("Cycle time (hours)")
    cycle_axis.set_ylabel("Cases")
    cycle_axis.set_title("Case Cycle Times")
    cycle_figure.tight_layout()
    figures.append(cycle_figure)
    activity_figure, activity_axis = plt.subplots(figsize=(8, 4))
    activity_axis.bar(dwell.index.astype(str), dwell.to_numpy(), color="#f58518")
    activity_axis.tick_params(axis="x", rotation=30)
    activity_axis.set_ylabel("Hours")
    activity_axis.set_title("Average Dwell Time by Activity")
    activity_figure.tight_layout()
    figures.append(activity_figure)
    return {
        "report": {
            "Summary": pd.DataFrame([
                ("Cases", data[case].nunique()),
                ("Events", len(data)),
                ("Distinct activities", data[activity].nunique()),
                ("Mean cycle hours", round(float(durations["Cycle Hours"].mean()), 3)),
                ("Median cycle hours", round(float(durations["Cycle Hours"].median()), 3)),
                ("Distinct transitions", len(transition_counts)),
            ], columns=["Metric", "Value"]),
            "Top Transitions": transition_frame,
            "Cycle Times": durations.reset_index().rename(columns={case: "Case", "min": "Started", "max": "Completed"})[["Case", "Started", "Completed", "Cycle Hours"]].head(30),
            "Activity Dwell": dwell.reset_index().rename(columns={activity: "Activity"}),
        },
        "figures": figures,
    }


def _build_gravity(frame, params):
    flow = str(params.get("flow") or "").strip()
    exporter = str(params.get("exporter_gdp") or "").strip()
    importer = str(params.get("importer_gdp") or "").strip()
    distance = str(params.get("distance") or "").strip()
    dummy = str(params.get("dummy") or "").strip()
    columns = [flow, exporter, importer, distance] + ([dummy] if dummy in frame.columns else [])
    if any(column not in frame.columns for column in columns[:4]):
        raise ValueError("Flow, exporter GDP, importer GDP and distance columns are required.")
    data = frame[columns].copy()
    for column in columns[:4]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna()
    data = data[(data[flow] > 0) & (data[exporter] > 0) & (data[importer] > 0) & (data[distance] > 0)]
    if len(data) < 15:
        raise ValueError("At least 15 positive observations are required for a gravity model.")
    predictors = pd.DataFrame({
        "const": 1.0,
        "log_exporter_gdp": np.log(data[exporter]),
        "log_importer_gdp": np.log(data[importer]),
        "log_distance": np.log(data[distance]),
    })
    if dummy in data.columns:
        predictors["common_currency"] = pd.to_numeric(data[dummy], errors="coerce").fillna(0).astype(float)
    predictors = predictors.replace([np.inf, -np.inf], np.nan).dropna()
    target_values = np.log(data.loc[predictors.index, flow])
    model = sm.OLS(target_values, predictors).fit()
    intervals = model.conf_int()
    coefficients = pd.DataFrame({
        "Coefficient": model.params,
        "Std Error": model.bse,
        "P Value": model.pvalues,
        "CI Low": intervals[0],
        "CI High": intervals[1],
    }).round(5).reset_index().rename(columns={"index": "Term"})
    fitted = np.asarray(model.fittedvalues, dtype=float)
    residuals = np.asarray(model.resid, dtype=float)
    figures = []
    fitted_figure, fitted_axis = plt.subplots(figsize=(7, 5))
    fitted_axis.scatter(fitted, target_values, alpha=0.7, color="#4c78a8")
    fitted_axis.plot([target_values.min(), target_values.max()], [target_values.min(), target_values.max()], color="black", linestyle="--")
    fitted_axis.set_xlabel("Fitted log trade flow")
    fitted_axis.set_ylabel("Actual log trade flow")
    fitted_axis.set_title("Gravity Model Fit")
    fitted_figure.tight_layout()
    figures.append(fitted_figure)
    residual_figure, residual_axis = plt.subplots(figsize=(7, 4))
    residual_axis.hist(residuals, bins=15, color="#72b7b2", alpha=0.8)
    residual_axis.axvline(0, color="black", linestyle="--")
    residual_axis.set_title("Residuals")
    residual_figure.tight_layout()
    figures.append(residual_figure)
    coefficient_figure, coefficient_axis = plt.subplots(figsize=(8, 4))
    plot_terms = coefficients[coefficients["Term"] != "const"]
    coefficient_axis.bar(plot_terms["Term"], plot_terms["Coefficient"], color="#54a24b")
    coefficient_axis.axhline(0, color="black", linewidth=0.8)
    coefficient_axis.tick_params(axis="x", rotation=25)
    coefficient_axis.set_title("Trade Elasticities")
    coefficient_figure.tight_layout()
    figures.append(coefficient_figure)
    return {
        "report": {
            "Summary": pd.DataFrame([
                ("Observations", len(predictors)),
                ("R-squared", round(float(model.rsquared), 4)),
                ("Adjusted R-squared", round(float(model.rsquared_adj), 4)),
                ("Exporter GDP elasticity", round(float(model.params.get("log_exporter_gdp", np.nan)), 4)),
                ("Importer GDP elasticity", round(float(model.params.get("log_importer_gdp", np.nan)), 4)),
                ("Distance elasticity", round(float(model.params.get("log_distance", np.nan)), 4)),
            ], columns=["Metric", "Value"]),
            "Coefficients": coefficients,
            "Fitted Values": pd.DataFrame({"Row": predictors.index, "Actual Log Flow": np.round(target_values.to_numpy(), 4) if hasattr(target_values, "to_numpy") else np.round(target_values, 4), "Fitted Log Flow": np.round(fitted, 4)}).head(30),
        },
        "figures": figures,
    }


def _build_tariff(frame, params):
    price = str(params.get("price") or "").strip()
    demand = str(params.get("demand") or "").strip()
    elasticity = str(params.get("elasticity") or "").strip()
    if any(column not in frame.columns for column in (price, demand, elasticity)):
        raise ValueError("Price, demand and elasticity columns are required.")
    data = frame[[price, demand, elasticity]].dropna()
    data = data[pd.to_numeric(data[price], errors="coerce").notna() & pd.to_numeric(data[demand], errors="coerce").notna() & pd.to_numeric(data[elasticity], errors="coerce").notna()]
    if len(data) < 5:
        raise ValueError("At least five complete price, demand and elasticity records are required.")
    data = data.apply(pd.to_numeric, errors="coerce")
    tariff = _number(params.get("tariff"), 0.10, 0.0, 3.0)
    scenarios = _number(params.get("scenarios"), 500, 50, 100000, integer=True)
    seed = _number(params.get("seed"), 42, integer=True)
    rng = np.random.default_rng(seed)
    noise_scale = np.maximum(np.abs(data[elasticity].to_numpy()) * 0.15, 0.05)
    simulated_elasticity = rng.normal(data[elasticity].to_numpy(dtype=float)[None, :], np.asarray(noise_scale, dtype=float)[None, :], size=(scenarios, len(data)))
    simulated_elasticity = np.clip(simulated_elasticity, -4.0, 0.5)
    demand_change = simulated_elasticity * tariff
    new_demand = np.maximum(data[demand].to_numpy(dtype=float)[None, :] * (1 + demand_change), 0)
    new_price = data[price].to_numpy(dtype=float)[None, :] * (1 + tariff)
    revenue = new_price * new_demand
    baseline_revenue = float((data[price] * data[demand]).sum())
    mean_revenue = float(revenue.sum(axis=1).mean())
    mean_demand_change = float(demand_change.mean() * 100)
    scenarios_frame = pd.DataFrame({
        "Scenario": np.arange(1, scenarios + 1),
        "Mean Demand Change (%)": np.round(demand_change.mean(axis=1) * 100, 3),
        "Total Revenue": np.round(revenue.sum(axis=1), 2),
        "Revenue Change (%)": np.round((revenue.sum(axis=1) / baseline_revenue - 1) * 100, 3) if baseline_revenue else np.nan,
    }).head(30)
    product_impact = pd.DataFrame({
        "Row": data.index,
        "Base Price": data[price].round(2),
        "New Price": np.round(data[price] * (1 + tariff), 2),
        "Base Demand": data[demand].round(2),
        "Expected New Demand": np.round(data[demand] * (1 + data[elasticity] * tariff), 2).clip(lower=0),
        "Base Revenue": np.round(data[price] * data[demand], 2),
        "Expected Revenue": np.round(new_price[0] * np.maximum(data[demand] * (1 + data[elasticity] * tariff), 0), 2),
    }).head(30)
    figures = []
    demand_figure, demand_axis = plt.subplots(figsize=(8, 4))
    demand_axis.hist(demand_change.ravel() * 100, bins=35, color="#e45756", alpha=0.8)
    demand_axis.axvline(mean_demand_change, color="black", linestyle="--", label="Mean")
    demand_axis.set_xlabel("Demand change (%)")
    demand_axis.set_ylabel("Scenario-product pairs")
    demand_axis.set_title("Simulated Demand Response")
    demand_axis.legend()
    demand_figure.tight_layout()
    figures.append(demand_figure)
    revenue_figure, revenue_axis = plt.subplots(figsize=(8, 4))
    revenue_axis.hist(revenue.sum(axis=1), bins=30, color="#54a24b", alpha=0.8)
    revenue_axis.axvline(baseline_revenue, color="black", linestyle="--", label="Baseline revenue")
    revenue_axis.set_xlabel("Total revenue")
    revenue_axis.set_ylabel("Scenarios")
    revenue_axis.set_title("Revenue Distribution")
    revenue_axis.legend()
    revenue_figure.tight_layout()
    figures.append(revenue_figure)
    elasticity_figure, elasticity_axis = plt.subplots(figsize=(8, 4))
    elasticity_axis.bar(range(min(15, len(data))), data[elasticity].to_numpy()[:15], color="#4c78a8")
    elasticity_axis.set_xlabel("Product")
    elasticity_axis.set_ylabel("Price elasticity")
    elasticity_axis.set_title("Input Elasticities")
    elasticity_figure.tight_layout()
    figures.append(elasticity_figure)
    return {
        "report": {
            "Summary": pd.DataFrame([
                ("Products", len(data)),
                ("Tariff rate", tariff),
                ("Scenarios", scenarios),
                ("Baseline revenue", round(baseline_revenue, 2)),
                ("Mean simulated revenue", round(mean_revenue, 2)),
                ("Mean revenue change (%)", round((mean_revenue / baseline_revenue - 1) * 100, 3) if baseline_revenue else np.nan),
                ("Mean demand change (%)", round(mean_demand_change, 3)),
            ], columns=["Metric", "Value"]),
            "Product Impact": product_impact,
            "Scenario Sample": scenarios_frame,
        },
        "figures": figures,
    }


def _build_fx_risk(frame, params):
    base = _build_portfolio(frame, params)
    returns_columns = _split_columns(params.get("returns"))
    if not returns_columns:
        returns_columns = _numeric_columns(frame)
    data = _columns(frame, returns_columns, "return columns").dropna()
    weight_values = _split_columns(params.get("weights"))
    if weight_values and len(weight_values) == len(returns_columns):
        weights = np.asarray(weight_values, dtype=float)
    else:
        weights = np.repeat(1.0 / len(returns_columns), len(returns_columns))
    weights = weights / weights.sum()
    shocks = [float(value) for value in _split_columns(params.get("shocks")) if np.isfinite(float(value))]
    stress = pd.DataFrame({
        "FX Shock (%)": shocks,
        "Portfolio P&L (%)": np.round(np.asarray(shocks, dtype=float) * float(weights.sum()), 4) if shocks else [],
        "Stressed Value of 1": np.round(1 + np.asarray(shocks, dtype=float) / 100, 4) if shocks else [],
    })
    stress["Loss (%)"] = np.round(-stress["Portfolio P&L (%)"], 4)
    if shocks:
        stress_figure, stress_axis = plt.subplots(figsize=(8, 4))
        stress_axis.bar(stress["FX Shock (%)"].astype(str), stress["Portfolio P&L (%)"], color="#e45756")
        stress_axis.axhline(0, color="black", linewidth=0.8)
        stress_axis.set_xlabel("FX shock (%)")
        stress_axis.set_ylabel("Portfolio P&L (%)")
        stress_axis.set_title("FX Stress Scenarios")
        stress_figure.tight_layout()
        base["figures"].append(stress_figure)
    base["report"]["Stress Scenarios"] = stress
    return base


BUILDERS = {
    ("finance", "credit_risk"): lambda frame, params: _build_classification(frame, params, "finance", "credit_risk"),
    ("finance", "fraud_detection"): _build_fraud,
    ("finance", "portfolio_risk"): _build_portfolio,
    ("finance", "fair_lending"): lambda frame, params: _build_fair_lending(frame, params),
    ("finance", "demand_forecast"): lambda frame, params: _build_forecast(frame, params, "finance", "demand_forecast"),
    ("hrm", "attrition"): lambda frame, params: _build_classification(frame, params, "hrm", "attrition"),
    ("hrm", "pay_equity"): _build_pay_equity,
    ("hrm", "employee_segmentation"): _build_segmentation,
    ("hrm", "workforce_forecast"): lambda frame, params: _build_forecast(frame, params, "hrm", "workforce_forecast"),
    ("it_operations", "predictive_maintenance"): _build_fraud,
    ("it_operations", "incident_priority"): lambda frame, params: _build_classification(frame, params, "it_operations", "incident_priority"),
    ("it_operations", "sla_breach"): lambda frame, params: _build_classification(frame, params, "it_operations", "sla_breach"),
    ("it_operations", "capacity_forecast"): lambda frame, params: _build_forecast(frame, params, "it_operations", "capacity_forecast"),
    ("it_operations", "inventory_reorder"): _build_inventory,
    ("it_operations", "supplier_risk"): _build_supplier_risk,
    ("it_operations", "process_cycle"): _build_process_cycle,
    ("international", "country_segmentation"): _build_segmentation,
    ("international", "export_forecast"): lambda frame, params: _build_forecast(frame, params, "international", "export_forecast"),
    ("international", "fx_risk"): _build_fx_risk,
    ("international", "trade_anomaly"): _build_fraud,
    ("international", "gravity_model"): _build_gravity,
    ("international", "tariff_impact"): _build_tariff,
}


def simulate(domain, preset, rows=200, seed=42):
    domain_key = _key(domain)
    preset_key = _key(preset)
    if domain_key not in DOMAIN_SIMULATIONS:
        raise ValueError(f"Unknown domain '{domain}'. Use one of: {', '.join(DOMAIN_LABELS)}")
    if preset_key not in [key for _, key in DOMAIN_SIMULATIONS[domain_key]]:
        raise ValueError(f"Unknown simulation '{preset}' for domain '{domain_key}'.")
    rows = int(_number(rows, 200, 20, 10000, integer=True))
    random_state = int(_number(seed, 42, integer=True))
    rng = np.random.default_rng(random_state)
    index = np.arange(rows)
    if domain_key == "finance" and preset_key == "credit_risk":
        age = rng.integers(21, 70, rows)
        income = np.round(rng.lognormal(np.log(65000), 0.45, rows), 2)
        credit_score = np.clip(rng.normal(680, 75, rows), 300, 850).round(0)
        debt_ratio = np.clip(rng.normal(0.32, 0.12, rows), 0.02, 0.9).round(3)
        loan_amount = np.round(rng.uniform(3000, 60000, rows), 2)
        credit_history = rng.integers(0, 25, rows)
        employment = rng.integers(0, 30, rows)
        score = -3.2 + 0.008 * (credit_score - 650) - 4.5 * debt_ratio + 0.00004 * income - 0.035 * loan_amount / 1000 + 0.06 * credit_history - 0.04 * employment
        probabilities = 1 / (1 + np.exp(-score))
        frame = pd.DataFrame({
            "Age": age,
            "AnnualIncome": income,
            "CreditScore": credit_score,
            "DebtToIncome": debt_ratio,
            "LoanAmount": loan_amount,
            "CreditHistoryYears": credit_history,
            "EmploymentYears": employment,
            "Default": np.where(rng.random(rows) < probabilities, "Yes", "No"),
        })
    elif domain_key == "finance" and preset_key == "fraud":
        account_age = rng.integers(1, 120, rows)
        amount = np.round(rng.lognormal(np.log(120), 1.05, rows), 2)
        hour = rng.integers(0, 24, rows)
        distance = np.round(rng.exponential(35, rows), 1)
        previous_flags = rng.poisson(0.25, rows)
        fraud = rng.random(rows) < np.clip(0.02 + 0.006 * previous_flags + 0.000004 * amount + 0.0015 * distance, 0.02, 0.35)
        fraud_amount = np.round(amount * rng.uniform(1.5, 4.0, rows), 2)
        frame = pd.DataFrame({
            "TransactionID": [f"TXN{10000 + position}" for position in index],
            "Amount": np.where(fraud, fraud_amount, amount).round(2),
            "HourOfDay": np.where(fraud, rng.integers(0, 5, rows), hour),
            "DistanceFromHome": np.where(fraud, distance * rng.uniform(2, 6, rows), distance).round(1),
            "PreviousFlags": previous_flags + np.where(fraud, rng.integers(1, 4, rows), 0),
            "AccountAgeMonths": account_age,
            "IsFraud": np.where(fraud, "Yes", "No"),
        })
    elif domain_key == "finance" and preset_key == "portfolio":
        covariance = np.array([[0.00016, 0.00009, 0.00004, 0.00002], [0.00009, 0.00025, 0.00006, 0.00001], [0.00004, 0.00006, 0.00020, 0.00003], [0.00002, 0.00001, 0.00003, 0.00010]])
        returns = rng.multivariate_normal([0.0004, 0.0006, 0.0002, 0.0003], covariance, rows)
        frame = pd.DataFrame({
            "Date": pd.date_range("2024-01-01", periods=rows, freq="B"),
            "Return_A": np.round(returns[:, 0], 6),
            "Return_B": np.round(returns[:, 1], 6),
            "Return_C": np.round(returns[:, 2], 6),
            "Return_D": np.round(returns[:, 3], 6),
        })
    elif domain_key == "finance" and preset_key == "applications":
        income = np.round(rng.lognormal(np.log(72000), 0.42, rows), 2)
        credit_score = np.clip(rng.normal(690, 70, rows), 300, 850).round(0)
        debt_ratio = np.clip(rng.normal(0.30, 0.11, rows), 0.01, 0.9).round(3)
        loan_amount = np.round(rng.uniform(5000, 80000, rows), 2)
        groups = rng.choice(["Female", "Male", "Other"], rows, p=[0.35, 0.55, 0.10])
        approval_score = 0.008 * (credit_score - 650) - 4.0 * debt_ratio + 0.00003 * income - 0.02 * loan_amount / 1000 + 0.12 * np.where(groups == "Male", 1, 0)
        approved = rng.random(rows) < 1 / (1 + np.exp(-approval_score))
        defaults = rng.random(rows) < np.clip(0.05 + 0.12 * debt_ratio + 0.05 * (debt_ratio > 0.5), 0.01, 0.35)
        frame = pd.DataFrame({
            "ApplicationID": [f"APP{20000 + position}" for position in index],
            "Gender": groups,
            "AnnualIncome": income,
            "CreditScore": credit_score,
            "DebtToIncome": debt_ratio,
            "LoanAmount": loan_amount,
            "Approved": np.where(approved, "Yes", "No"),
            "Default": np.where(defaults, "Yes", "No"),
        })
    elif domain_key == "finance" and preset_key == "collections":
        date = pd.date_range("2023-01-01", periods=rows, freq="ME")
        drift = np.arange(rows) * 0.65
        seasonal = 16 * np.sin(2 * np.pi * np.arange(rows) / 12)
        collections = np.maximum(120 + drift + seasonal + rng.normal(0, 7, rows), 20)
        frame = pd.DataFrame({
            "Date": date,
            "Collections": np.round(collections, 2),
            "CampaignSpend": np.round(rng.uniform(2000, 9000, rows), 2),
            "Region": rng.choice(["North", "South", "East", "West"], rows),
        })
    elif domain_key == "hrm" and preset_key == "attrition":
        age = rng.integers(22, 62, rows)
        tenure = np.clip(rng.integers(0, 25, rows), 0, age - 20)
        income = np.round(25000 + tenure * 1800 + age * 220 + rng.normal(0, 4500, rows), 2)
        satisfaction = rng.integers(1, 6, rows)
        rating = rng.integers(2, 6, rows)
        overtime = rng.poisson(rng.uniform(2, 12), rows)
        distance = np.round(rng.exponential(18, rows), 1)
        risk = 0.08 + 0.16 * (satisfaction <= 2) + 0.12 * (overtime > 10) + 0.06 * (rating <= 2) - 0.015 * tenure + 0.004 * distance
        frame = pd.DataFrame({
            "EmployeeID": [f"EMP{1000 + position}" for position in index],
            "Age": age,
            "TenureYears": tenure,
            "MonthlyIncome": np.maximum(income / 12, 1000).round(2),
            "JobSatisfaction": satisfaction,
            "PerformanceRating": rating,
            "OvertimeHours": overtime,
            "DistanceFromHome": distance,
            "Attrition": np.where(rng.random(rows) < np.clip(risk, 0.02, 0.6), "Yes", "No"),
        })
    elif domain_key == "hrm" and preset_key == "compensation":
        role_options = ["Analyst", "Engineer", "Manager"]
        roles = rng.choice(role_options, rows)
        genders = rng.choice(["Female", "Male"], rows)
        role_base = np.select([roles == "Analyst", roles == "Engineer", roles == "Manager"], [62000, 88000, 125000], default=70000)
        tenure = rng.integers(0, 18, rows)
        rating = rng.integers(2, 6, rows)
        certifications = rng.poisson(1.2, rows)
        salary = role_base + tenure * 2100 + rating * 1800 + certifications * 1200 + np.where(genders == "Male", 2600, 0) + rng.normal(0, 5000, rows)
        frame = pd.DataFrame({
            "EmployeeID": [f"EMP{2000 + position}" for position in index],
            "Gender": genders,
            "Role": roles,
            "TenureYears": tenure,
            "PerformanceRating": rating,
            "CertificationCount": certifications,
            "Salary": np.round(np.maximum(salary, 30000), 2),
        })
    elif domain_key == "hrm" and preset_key == "headcount":
        date = pd.date_range("2024-01-01", periods=rows, freq="ME")
        department = rng.choice(["Operations", "Sales", "Engineering"], rows)
        base = np.select([department == "Operations", department == "Sales", department == "Engineering"], [180, 120, 260], default=160)
        headcount = base + np.arange(rows) * 0.35 + 6 * np.sin(2 * np.pi * np.arange(rows) / 12) + rng.normal(0, 3, rows)
        frame = pd.DataFrame({"Date": date, "Department": department, "Headcount": np.round(headcount).astype(int)})
    elif domain_key == "it_operations" and preset_key == "incidents":
        affected_users = rng.poisson(np.exp(rng.normal(2.6, 0.9, rows))).astype(float)
        age_hours = np.round(rng.exponential(18, rows), 2)
        category_code = rng.integers(1, 7, rows)
        group_code = rng.integers(1, 10, rows)
        priority_score = 0.05 * affected_users + 0.08 * age_hours + 0.4 * category_code
        priority = np.select([priority_score > 12, priority_score > 6], ["P1", "P2"], default="P3")
        breach = (priority_score > 8) | (age_hours > 30)
        frame = pd.DataFrame({
            "TicketID": [f"INC{30000 + position}" for position in index],
            "CategoryCode": category_code,
            "AssignmentGroupCode": group_code,
            "AffectedUsers": affected_users,
            "AgeHours": age_hours,
            "Priority": priority,
            "PriorityCode": np.select([priority == "P1", priority == "P2"], [1, 2], default=3),
            "SLA_Breach": np.where(breach, "Yes", "No"),
        })
    elif domain_key == "it_operations" and preset_key == "sensors":
        machine = [f"MCH{100 + position % 12}" for position in index]
        runtime_hours = np.round(rng.uniform(200, 9000, rows), 1)
        temperature = np.round(62 + runtime_hours / 500 + rng.normal(0, 3, rows), 2)
        vibration = np.round(1.2 + runtime_hours / 4000 + rng.normal(0, 0.18, rows), 3)
        pressure = np.round(100 + rng.normal(0, 6, rows), 2)
        failure = rng.random(rows) < np.clip(0.02 + np.maximum(0, temperature - 72) * 0.05 + np.maximum(0, vibration - 1.8) * 0.35, 0.01, 0.45)
        frame = pd.DataFrame({
            "MachineID": machine,
            "Temperature": temperature,
            "Vibration": vibration,
            "Pressure": pressure,
            "RuntimeHours": runtime_hours,
            "Failure": np.where(failure, "Yes", "No"),
        })
    elif domain_key == "it_operations" and preset_key == "capacity":
        date = pd.date_range("2025-01-01", periods=rows, freq="D")
        workload = 220 + np.arange(rows) * 0.35 + 28 * np.sin(2 * np.pi * np.arange(rows) / 12) + rng.normal(0, 9, rows)
        frame = pd.DataFrame({"Date": date, "Region": rng.choice(["East", "West", "Central"], rows), "Demand": np.round(workload, 2)})
    elif domain_key == "it_operations" and preset_key == "suppliers":
        countries = rng.choice(["India", "China", "Germany", "USA", "Brazil", "Japan"], rows)
        frame = pd.DataFrame({
            "SupplierID": [f"SUP{4000 + position}" for position in index],
            "Country": countries,
            "Spend": np.round(rng.lognormal(np.log(800000), 0.7, rows), 2),
            "DefectRate": np.round(np.clip(rng.gamma(2, 0.012, rows), 0, 0.3), 4),
            "LeadDays": rng.integers(3, 60, rows),
            "OnTimeRate": np.round(np.clip(rng.normal(0.9, 0.08, rows), 0.4, 1.0), 4),
        })
    elif domain_key == "it_operations" and preset_key == "events":
        case_count = max(3, min(20, rows // 4))
        activities = ["Received", "Triaged", "Assigned", "In Progress", "Resolved", "Closed"]
        base_time = pd.Timestamp("2026-01-01")
        records = []
        for position in range(rows):
            case_index = position % case_count
            step = position // case_count
            start = base_time + pd.Timedelta(days=case_index, hours=int(rng.integers(0, 6)))
            records.append((f"CASE{5000 + case_index}", activities[step % len(activities)], start + pd.Timedelta(hours=int(step * 3 + rng.integers(0, 2)))))
        frame = pd.DataFrame(records, columns=["CaseID", "Activity", "Timestamp"])
    elif domain_key == "international" and preset_key == "trade":
        date = pd.date_range("2020-01-01", periods=rows, freq="ME")
        countries = rng.choice(["USA", "UK", "Germany", "India", "Japan", "Brazil"], rows)
        exporter_gdp = np.round(rng.uniform(800, 22000, rows), 1)
        importer_gdp = np.round(rng.uniform(400, 18000, rows), 1)
        distance = np.round(rng.uniform(300, 12000, rows), 1)
        common_currency = rng.integers(0, 2, rows)
        trade_value = np.round(np.exp(1.5 + 0.65 * np.log(exporter_gdp) + 0.55 * np.log(importer_gdp) - 0.35 * np.log(distance) + 0.25 * common_currency + rng.normal(0, 0.28, rows)), 2)
        frame = pd.DataFrame({
            "Date": date,
            "Country": countries,
            "Partner": rng.choice(["USA", "UK", "Germany", "India", "Japan", "Brazil"], rows),
            "TradeValue": trade_value,
            "ExporterGDP": exporter_gdp,
            "ImporterGDP": importer_gdp,
            "DistanceKm": distance,
            "CommonCurrency": common_currency,
            "BasePrice": np.round(rng.uniform(20, 500, rows), 2),
            "DemandUnits": np.round(rng.lognormal(np.log(2000), 0.5, rows), 1),
            "PriceElasticity": np.round(-np.abs(rng.normal(1.3, 0.35, rows)), 3),
        })
    elif domain_key == "international" and preset_key == "fx":
        covariance = np.array([[0.000025, 0.000012, 0.000008, 0.000005], [0.000012, 0.000030, 0.000007, 0.000004], [0.000008, 0.000007, 0.000045, 0.000003], [0.000005, 0.000004, 0.000003, 0.000020]])
        returns = rng.multivariate_normal([0.0001, 0.0002, -0.0001, 0.0003], covariance, rows)
        frame = pd.DataFrame({
            "Date": pd.date_range("2024-01-01", periods=rows, freq="B"),
            "Return_USD_EUR": np.round(returns[:, 0], 6),
            "Return_USD_GBP": np.round(returns[:, 1], 6),
            "Return_USD_JPY": np.round(returns[:, 2], 6),
            "Return_USD_INR": np.round(returns[:, 3], 6),
        })
    elif domain_key == "international" and preset_key == "tariff":
        frame = pd.DataFrame({
            "ProductID": [f"PRD{6000 + position}" for position in index],
            "Country": rng.choice(["USA", "UK", "Germany", "India", "Japan", "Brazil"], rows),
            "BasePrice": np.round(rng.uniform(15, 600, rows), 2),
            "DemandUnits": np.round(rng.lognormal(np.log(2500), 0.6, rows), 1),
            "PriceElasticity": np.round(-np.abs(rng.normal(1.4, 0.45, rows)), 3),
        })
    else:
        raise ValueError(f"Simulation '{preset}' for domain '{domain_key}' is not implemented.")
    frame.attrs["Domain Simulation"] = f"{domain_key}:{preset_key}"
    return frame


def default_parameters(domain, technique, frame=None):
    domain_key = _key(domain)
    technique_key = _key(technique)
    if (domain_key, technique_key) not in ANALYSIS_TITLES:
        raise ValueError(f"Unknown analysis '{technique}' for domain '{domain_key}'.")
    defaults = {key: str(default) for key, _, default in PARAMETER_SPECS[domain_key][technique_key]}
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return defaults
    columns = list(frame.columns)
    target_keywords = {
        ("finance", "credit_risk"): ["default", "bad_loan", "target"],
        ("finance", "fraud_detection"): ["fraud", "isfraud", "anomaly"],
        ("hrm", "attrition"): ["attrition", "churn", "left", "target"],
        ("it_operations", "incident_priority"): ["priority", "severity"],
        ("it_operations", "sla_breach"): ["sla_breach", "breach", "sla"],
        ("it_operations", "predictive_maintenance"): ["failure", "fault", "breakdown"],
        ("international", "trade_anomaly"): ["anomaly", "fraud", "label"],
    }
    if (domain_key, technique_key) in target_keywords:
        target = _find_column(frame, target_keywords[(domain_key, technique_key)])
        if target:
            defaults["target"] = target
            features = _numeric_columns(frame, exclude=[target])[:8]
            if features:
                defaults["features"] = ", ".join(features)
    if technique_key in ("credit_risk", "attrition", "incident_priority", "sla_breach"):
        if defaults.get("target") in frame.columns:
            features = _numeric_columns(frame, exclude=[defaults["target"]])[:8]
            if features:
                defaults["features"] = ", ".join(features)
    if technique_key in ("fraud_detection", "predictive_maintenance", "trade_anomaly"):
        target = defaults.get("target") if defaults.get("target") in frame.columns else None
        features = _numeric_columns(frame, exclude=[target] if target else [])[:8]
        if features:
            defaults["features"] = ", ".join(features)
        if target:
            defaults["target"] = target
    if technique_key in ("portfolio_risk", "fx_risk"):
        if technique_key == "fx_risk":
            returns = [column for column in frame.columns if any(word in str(column).lower() for word in ("return", "fx"))][:4] or _numeric_columns(frame)[:4]
        else:
            returns = [column for column in _numeric_columns(frame) if not any(word in str(column).lower() for word in ("id", "age", "year"))][:4]
        if returns:
            defaults["returns"] = ", ".join(returns)
    if technique_key == "fair_lending":
        decision = _find_column(frame, ["approved", "approval", "decision", "status"])
        group = _find_column(frame, ["gender", "group", "segment", "category"])
        target = _find_column(frame, ["default", "bad", "outcome", "loss"])
        for name, value in (("decision", decision), ("group", group), ("target", target)):
            if value:
                defaults[name] = value
        controls = _numeric_columns(frame, exclude=[value for value in (decision, group, target) if value])[:4]
        if controls:
            defaults["features"] = ", ".join(controls)
    if technique_key in ("demand_forecast", "workforce_forecast", "capacity_forecast", "export_forecast"):
        preferred = {
            "demand_forecast": ["collections", "demand", "applications", "volume"],
            "workforce_forecast": ["headcount", "employees", "staffing"],
            "capacity_forecast": ["demand", "cpu", "load", "utilization"],
            "export_forecast": ["tradevalue", "exports", "value", "demand"],
        }
        value = _find_column(frame, preferred[technique_key])
        if value:
            defaults["value"] = value
        period = _find_column(frame, ["date", "period", "month", "week", "time"])
        if period:
            defaults["period"] = period
        if technique_key == "export_forecast":
            group = _find_column(frame, ["country", "region", "market", "product"])
            if group:
                defaults["filter_column"] = group
    if technique_key == "pay_equity":
        salary = _find_column(frame, ["salary", "income", "compensation", "pay"])
        group = _find_column(frame, ["gender", "group", "category", "segment"])
        role = _find_column(frame, ["role", "department", "designation", "job"])
        performance = [column for column in _numeric_columns(frame, exclude=[salary]) if column != group][:3]
        if salary:
            defaults["salary"] = salary
        if group:
            defaults["group"] = group
        if role:
            defaults["role"] = role
        if performance:
            defaults["performance"] = ", ".join(performance)
    if technique_key in ("employee_segmentation", "country_segmentation", "supplier_risk"):
        excluded = [column for column in frame.columns if any(word in str(column).lower() for word in ("id", "case", "ticket"))]
        features = _numeric_columns(frame, exclude=excluded)[:6]
        if features:
            defaults["features"] = ", ".join(features)
        if technique_key == "supplier_risk":
            group = _find_column(frame, ["country", "region", "supplier", "category"])
            if group:
                defaults["group"] = group
    if technique_key == "inventory_reorder":
        demand = _find_column(frame, ["demand", "units", "volume", "sales"])
        if demand:
            defaults["demand"] = demand
    if technique_key == "process_cycle":
        case = _find_column(frame, ["caseid", "case", "ticketid", "orderid"])
        activity = _find_column(frame, ["activity", "event", "step"])
        timestamp = _find_column(frame, ["timestamp", "time", "date"])
        if case:
            defaults["case"] = case
        if activity:
            defaults["activity"] = activity
        if timestamp:
            defaults["timestamp"] = timestamp
    if technique_key == "gravity_model":
        mappings = {
            "flow": ["tradevalue", "trade", "flow", "exports"],
            "exporter_gdp": ["exportergdp", "exporter", "gdp"],
            "importer_gdp": ["importergdp", "importer", "partner_gdp"],
            "distance": ["distancekm", "distance", "km"],
            "dummy": ["commoncurrency", "currency", "dummy"],
        }
        for name, words in mappings.items():
            column = _find_column(frame, words)
            if column:
                defaults[name] = column
    if technique_key == "tariff_impact":
        mappings = {
            "price": ["baseprice", "price"],
            "demand": ["demandunits", "demand", "units"],
            "elasticity": ["priceelasticity", "elasticity"],
        }
        for name, words in mappings.items():
            column = _find_column(frame, words)
            if column:
                defaults[name] = column
    return defaults


def parameter_fields(domain, technique, frame=None):
    domain_key = _key(domain)
    technique_key = _key(technique)
    if (domain_key, technique_key) not in ANALYSIS_TITLES:
        raise ValueError(f"Unknown analysis '{technique}' for domain '{domain_key}'.")
    defaults = default_parameters(domain_key, technique_key, frame)
    return [(key, label, defaults[key]) for key, label, _ in PARAMETER_SPECS[domain_key][technique_key]]


def analyze(domain, technique, frame, **params):
    domain_key = _key(domain)
    technique_key = _key(technique)
    if domain_key not in DOMAIN_LABELS:
        raise ValueError(f"Unknown domain '{domain}'. Use one of: {', '.join(DOMAIN_LABELS)}")
    builder = BUILDERS.get((domain_key, technique_key))
    if builder is None:
        available = ", ".join(key for _, key in DOMAIN_ANALYSES[domain_key])
        raise ValueError(f"Unknown analysis '{technique}' for domain '{domain_key}'. Use one of: {available}")
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("A non-empty pandas DataFrame is required.")
    resolved = default_parameters(domain_key, technique_key, frame)
    resolved.update({key: value for key, value in params.items() if value is not None})
    result = builder(frame, resolved)
    return {
        "title": ANALYSIS_TITLES[(domain_key, technique_key)],
        "domain": domain_key,
        "technique": technique_key,
        "report": result["report"],
        "figures": result["figures"],
    }


def analyze_frame(domain, technique, frame, **params):
    return analyze(domain, technique, frame, **params)
