"""Data loading, segmentation and KPI helpers for the churn project."""
import numpy as np
import pandas as pd

REQUIRED = ["CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance",
            "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary", "Exited"]

AGE_ORDER = ["<30", "30-45", "46-60", "60+"]
CREDIT_ORDER = ["Low", "Medium", "High"]
TENURE_ORDER = ["New", "Mid-term", "Long-term"]
BALANCE_ORDER = ["Zero-balance", "Low-balance", "High-balance"]


def validate(df: pd.DataFrame) -> list:
    """Return list of problems found (empty list = OK)."""
    problems = [f"Missing column: {c}" for c in REQUIRED if c not in df.columns]
    if not problems:
        if df[REQUIRED].isna().any().any():
            problems.append("Dataset contains missing values in required columns.")
        if not set(df["Exited"].unique()) <= {0, 1}:
            problems.append("'Exited' must be 0/1.")
    return problems


def make_demo_data(n=10000, seed=42) -> pd.DataFrame:
    """Synthetic data with the same schema/patterns as the real bank dataset (for testing only)."""
    rng = np.random.default_rng(seed)
    geo = rng.choice(["France", "Spain", "Germany"], n, p=[0.50, 0.25, 0.25])
    age = np.clip(rng.normal(39, 10.5, n), 18, 92).astype(int)
    balance = np.where(
        (geo == "Germany") | (rng.random(n) > 0.45),
        rng.normal(120000, 30000, n), 0).clip(0)
    logit = (-2.0 + 0.065 * (age - 38).clip(0) + 0.8 * (geo == "Germany")
             - 0.9 * (rng.random(n) < 0.5) + 0.4 * (rng.random(n) < 0.3))
    active = rng.random(n) < 0.51
    logit = logit - 0.8 * active
    exited = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame({
        "RowNumber": np.arange(1, n + 1), "CustomerId": 15000000 + np.arange(n),
        "Surname": "Demo",
        "CreditScore": rng.normal(650, 97, n).clip(350, 850).astype(int),
        "Geography": geo, "Gender": rng.choice(["Male", "Female"], n, p=[0.55, 0.45]),
        "Age": age, "Tenure": rng.integers(0, 11, n), "Balance": balance.round(2),
        "NumOfProducts": rng.choice([1, 2, 3, 4], n, p=[0.5, 0.46, 0.03, 0.01]),
        "HasCrCard": (rng.random(n) < 0.7).astype(int),
        "IsActiveMember": active.astype(int),
        "EstimatedSalary": rng.uniform(10, 200000, n).round(2),
        "Exited": exited,
    })


def add_segments(df: pd.DataFrame, high_value_pct: float = 0.75) -> pd.DataFrame:
    """Add all segmentation columns required by the project brief."""
    d = df.copy()
    d["AgeGroup"] = pd.cut(d["Age"], [0, 29, 45, 60, 200],
                           labels=AGE_ORDER).astype(str)
    d["CreditBand"] = pd.cut(d["CreditScore"], [0, 579, 669, 1000],
                             labels=CREDIT_ORDER).astype(str)
    d["TenureGroup"] = pd.cut(d["Tenure"], [-1, 2, 6, 100],
                              labels=TENURE_ORDER).astype(str)
    nz = d.loc[d["Balance"] > 0, "Balance"]
    median_nz = nz.median() if len(nz) else 0
    d["BalanceSegment"] = np.select(
        [d["Balance"] == 0, d["Balance"] < median_nz],
        ["Zero-balance", "Low-balance"], default="High-balance")
    threshold = nz.quantile(high_value_pct) if len(nz) else np.inf
    d["HighValue"] = d["Balance"] >= threshold
    d.attrs["hv_threshold"] = float(threshold)
    return d


def churn_by(df, col, order=None):
    g = (df.groupby(col)["Exited"].agg(Customers="count", Churned="sum", ChurnRate="mean")
         .reset_index())
    g["ChurnRate"] = g["ChurnRate"] * 100
    g["ShareOfAllChurn"] = g["Churned"] / max(g["Churned"].sum(), 1) * 100
    if order:
        g[col] = pd.Categorical(g[col], order, ordered=True)
        g = g.sort_values(col)
    return g


def kpis(df: pd.DataFrame, full: pd.DataFrame) -> dict:
    """Project KPIs: segment churn rate, high-value churn ratio,
    geographic risk index, engagement drop indicator."""
    overall = df["Exited"].mean() * 100 if len(df) else 0
    base = full["Exited"].mean() * 100
    hv = df[df["HighValue"]]
    hv_rate = hv["Exited"].mean() * 100 if len(hv) else 0
    act = df[df["IsActiveMember"] == 1]["Exited"].mean() * 100
    ina = df[df["IsActiveMember"] == 0]["Exited"].mean() * 100
    return {
        "customers": len(df),
        "churned": int(df["Exited"].sum()),
        "churn_rate": overall,
        "hv_churn_rate": hv_rate,
        "hv_churn_ratio": hv_rate / overall if overall else 0,
        "revenue_at_risk": float(df.loc[df["Exited"] == 1, "Balance"].sum()),
        "engagement_gap": (ina / act) if act and not np.isnan(act) else 0,
        "geo_risk_index": {g: (v / base if base else 0) for g, v in
                           (full.groupby("Geography")["Exited"].mean() * 100).items()},
    }
