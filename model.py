"""Customer-level churn prediction: logistic regression (interpretable baseline)
and gradient boosting (non-linear challenger)."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUM = ["CreditScore", "Age", "Tenure", "Balance", "EstimatedSalary"]
CAT = ["Geography", "Gender", "NumOfProducts"]
BIN = ["HasCrCard", "IsActiveMember"]
SEED = 42


def make_xy(df: pd.DataFrame):
    X = df[NUM + CAT + BIN].copy()
    X["NumOfProducts"] = X["NumOfProducts"].astype(str)          # treat as category (non-monotonic effect)
    X["AgeSq"] = (X["Age"] - 40) ** 2 / 100                       # lets the linear model capture the age peak
    X["ZeroBalance"] = (X["Balance"] == 0).astype(int)
    return X, df["Exited"].astype(int)


def _pre():
    num = NUM + ["AgeSq"]
    return ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop=None), CAT),
        ("bin", "passthrough", BIN + ["ZeroBalance"]),
    ])


def build_models():
    logit = Pipeline([("pre", _pre()), ("clf", LogisticRegression(max_iter=2000, C=1.0))])
    gb = Pipeline([("pre", _pre()), ("clf", HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.06, max_iter=250, l2_regularization=1.0, random_state=SEED))])
    return {"Logistic regression": logit, "Gradient boosting": gb}


def evaluate(df: pd.DataFrame):
    """Train/test split (80/20, stratified) + 5-fold CV on the training set."""
    X, y = make_xy(df)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    res = {}
    for name, m in build_models().items():
        cvs = cross_val_score(m, Xtr, ytr, cv=cv, scoring="roc_auc")
        m.fit(Xtr, ytr)
        p = m.predict_proba(Xte)[:, 1]
        res[name] = dict(model=m, p=p, cv_auc=cvs.mean(), cv_sd=cvs.std(),
                         auc=roc_auc_score(yte, p), ap=average_precision_score(yte, p), brier=brier_score_loss(yte, p))
    return res, Xtr, Xte, ytr, yte


def fit_final(df: pd.DataFrame, name="Gradient boosting"):
    X, y = make_xy(df)
    m = build_models()[name]
    m.fit(X, y)
    return m


def risk_tier(p):
    return pd.cut(p, [-1, 0.2, 0.5, 2], labels=["Low", "Medium", "High"])
