"""Customer Segmentation & Churn Pattern Analytics in European Banking
Run:  streamlit run app.py
"""
import os
import pandas as pd
import plotly.express as px
import streamlit as st

import utils as U
import model as M

st.set_page_config(page_title="European Banking Churn Analytics", layout="wide")
st.markdown("""<style>
div[data-testid="stMetric"]{background:#F6F8FB;border-left:5px solid #1F3A5F;padding:12px 14px;border-radius:6px}
h1{color:#1F3A5F} h2,h3{color:#1F3A5F}</style>""", unsafe_allow_html=True)
st.title("Customer Segmentation & Churn Pattern Analytics")
st.caption("European Banking | France, Spain, Germany")

# ---------------------------------------------------------------- data
st.sidebar.header("Data")
up = st.sidebar.file_uploader("Upload dataset (CSV)", type="csv")
if up is not None:
    raw = pd.read_csv(up)
elif any(os.path.exists(f) for f in ("European_Bank.csv", "Churn_Modelling.csv")):
    raw = pd.read_csv(next(f for f in ("European_Bank.csv", "Churn_Modelling.csv") if os.path.exists(f)))
else:
    raw = U.make_demo_data()
    st.warning("Showing SYNTHETIC DEMO DATA. Upload the real dataset in the sidebar "
               "(or place European_Bank.csv next to app.py).")

problems = U.validate(raw)
if problems:
    st.error("Dataset problem(s):\n\n- " + "\n- ".join(problems))
    st.stop()

hv_pct = st.sidebar.slider("High-value = balance in top X% (of non-zero balances)",
                           50, 95, 75, 5) / 100
data = U.add_segments(raw, hv_pct)
hv_threshold = data.attrs["hv_threshold"]

# ---------------------------------------------------------------- filters
st.sidebar.header("Segment filters")
def pick(label, col, order=None):
    opts = order or sorted(data[col].unique())
    return st.sidebar.multiselect(label, opts, default=opts)

geo = pick("Geography", "Geography")
gen = pick("Gender", "Gender")
age = pick("Age group", "AgeGroup", U.AGE_ORDER)
cre = pick("Credit band", "CreditBand", U.CREDIT_ORDER)
ten = pick("Tenure group", "TenureGroup", U.TENURE_ORDER)
bal = pick("Balance segment", "BalanceSegment", U.BALANCE_ORDER)

df = data[data.Geography.isin(geo) & data.Gender.isin(gen) & data.AgeGroup.isin(age)
          & data.CreditBand.isin(cre) & data.TenureGroup.isin(ten)
          & data.BalanceSegment.isin(bal)]
if df.empty:
    st.info("No customers match the current filters.")
    st.stop()

k = U.kpis(df, data)

# ---------------------------------------------------------------- KPI row (dynamic)
c = st.columns(5)
c[0].metric("Customers", f"{k['customers']:,}")
c[1].metric("Churn rate", f"{k['churn_rate']:.1f}%",
            f"{k['churn_rate'] - data.Exited.mean()*100:+.1f} pts vs all")
c[2].metric("High-value churn ratio", f"{k['hv_churn_ratio']:.2f}x",
            help="High-value churn rate / overall churn rate in current selection")
c[3].metric("Engagement drop indicator", f"{k['engagement_gap']:.2f}x",
            help="Churn rate of inactive members / churn rate of active members")
c[4].metric("Balance at risk (churned)", f"€{k['revenue_at_risk']/1e6:,.1f}M")

tabs = st.tabs(["Overview", "Geography", "Age & Tenure", "Segments & Demographics",
                "High-value explorer", "Churn risk scoring", "Data"])

def bar(g, x, title, color=None):
    f = px.bar(g, x=x, y="ChurnRate", text=g["ChurnRate"].round(1), title=title,
               color=color, labels={"ChurnRate": "Churn rate (%)"})
    f.update_traces(textposition="outside")
    return f

# ---- Overview
with tabs[0]:
    a, b = st.columns(2)
    pie = df.Exited.map({0: "Retained", 1: "Churned"}).value_counts().reset_index()
    pie.columns = ["Status", "Customers"]
    a.plotly_chart(px.pie(pie, names="Status", values="Customers", hole=.5,
                          title="Retained vs churned"), width="stretch")
    seg = st.selectbox("Segment-wise churn by", ["AgeGroup", "CreditBand", "TenureGroup",
                       "BalanceSegment", "Geography", "Gender", "NumOfProducts"])
    order = {"AgeGroup": U.AGE_ORDER, "CreditBand": U.CREDIT_ORDER,
             "TenureGroup": U.TENURE_ORDER, "BalanceSegment": U.BALANCE_ORDER}.get(seg)
    g = U.churn_by(df, seg, order)
    b.plotly_chart(bar(g, seg, f"Churn rate by {seg}"), width="stretch")
    st.subheader("Churn contribution by segment size")
    st.dataframe(g.round(2), width="stretch", hide_index=True)
    st.subheader("Churned vs retained profile")
    prof = df.groupby("Exited")[["Age", "CreditScore", "Balance", "EstimatedSalary",
                                 "Tenure", "NumOfProducts", "IsActiveMember"]].mean().round(2)
    prof.index = prof.index.map({0: "Retained", 1: "Churned"})
    st.dataframe(prof, width="stretch")

# ---- Geography
with tabs[1]:
    g = U.churn_by(df, "Geography")
    g["GeoRiskIndex"] = g.Geography.map(k["geo_risk_index"]).round(2)
    a, b = st.columns(2)
    a.plotly_chart(bar(g, "Geography", "Churn rate by country"), width="stretch")
    b.plotly_chart(px.bar(g, x="Geography", y="Churned", title="Churned customers by country",
                          text="Churned"), width="stretch")
    st.caption("Geographic Risk Index = country churn rate / overall churn rate (>1 = above-average risk).")
    st.dataframe(g.round(2), width="stretch", hide_index=True)
    st.subheader("Geography × age interaction")
    ga = df.groupby(["Geography", "AgeGroup"])["Exited"].mean().mul(100).reset_index()
    ga["AgeGroup"] = pd.Categorical(ga.AgeGroup, U.AGE_ORDER, ordered=True)
    st.plotly_chart(px.density_heatmap(ga.pivot(index="Geography", columns="AgeGroup",
                    values="Exited").reset_index().melt("Geography"), x="AgeGroup", y="Geography",
                    z="value", text_auto=".1f", color_continuous_scale="Reds",
                    title="Churn rate (%) by country and age group"), width="stretch")

# ---- Age & tenure
with tabs[2]:
    a, b = st.columns(2)
    a.plotly_chart(bar(U.churn_by(df, "AgeGroup", U.AGE_ORDER), "AgeGroup",
                       "Churn rate by age group"), width="stretch")
    b.plotly_chart(bar(U.churn_by(df, "TenureGroup", U.TENURE_ORDER), "TenureGroup",
                       "Churn rate by tenure group"), width="stretch")
    t = U.churn_by(df, "Tenure")
    st.plotly_chart(px.line(t, x="Tenure", y="ChurnRate", markers=True,
                    title="Churn rate by years with bank"), width="stretch")
    st.plotly_chart(px.histogram(df, x="Age", color=df.Exited.map({0: "Retained", 1: "Churned"}),
                    barmode="overlay", opacity=.6, nbins=40, title="Age distribution"),
                    width="stretch")

# ---- Segments & demographics
with tabs[3]:
    a, b = st.columns(2)
    a.plotly_chart(bar(U.churn_by(df, "Gender"), "Gender", "Churn rate by gender"),
                   width="stretch")
    b.plotly_chart(bar(U.churn_by(df, "IsActiveMember"), "IsActiveMember",
                       "Churn rate: active (1) vs inactive (0)"), width="stretch")
    a, b = st.columns(2)
    a.plotly_chart(bar(U.churn_by(df, "CreditBand", U.CREDIT_ORDER), "CreditBand",
                       "Churn rate by credit band"), width="stretch")
    b.plotly_chart(bar(U.churn_by(df, "BalanceSegment", U.BALANCE_ORDER), "BalanceSegment",
                       "Churn rate by balance segment"), width="stretch")
    st.subheader("Financial stability vs churn")
    st.plotly_chart(px.box(df, x=df.Exited.map({0: "Retained", 1: "Churned"}), y="Balance",
                    title="Balance: retained vs churned"), width="stretch")

# ---- High-value explorer
with tabs[4]:
    hv = df[df.HighValue]
    st.write(f"High-value customers: balance ≥ **€{hv_threshold:,.0f}** "
             f"({len(hv):,} customers in current selection)")
    if hv.empty:
        st.info("No high-value customers in the current selection.")
    else:
        m = st.columns(3)
        m[0].metric("High-value churn rate", f"{k['hv_churn_rate']:.1f}%")
        m[1].metric("High-value churners", f"{int(hv.Exited.sum()):,}")
        m[2].metric("Balance lost", f"€{hv.loc[hv.Exited==1,'Balance'].sum()/1e6:,.1f}M",
                    f"{hv.loc[hv.Exited==1,'Balance'].sum()/max(df.loc[df.Exited==1,'Balance'].sum(),1)*100:.0f}% of all churned balance")
        st.plotly_chart(px.scatter(hv.sample(min(len(hv), 3000), random_state=1),
                        x="EstimatedSalary", y="Balance",
                        color=hv.loc[hv.index.isin(hv.sample(min(len(hv), 3000), random_state=1).index), "Exited"].map({0: "Retained", 1: "Churned"}),
                        opacity=.6, title="Salary vs balance (high-value customers)"),
                        width="stretch")
        st.dataframe(hv[hv.Exited == 1].sort_values("Balance", ascending=False)
                     .head(50)[["CustomerId", "Geography", "Age", "Balance", "EstimatedSalary",
                                "NumOfProducts", "IsActiveMember"]], width="stretch",
                     hide_index=True)


# ---- Churn risk scoring (customer-level prediction)
@st.cache_resource(show_spinner="Training churn model...")
def get_models(frame: pd.DataFrame):
    res, Xtr, Xte, ytr, yte = M.evaluate(frame)
    final = M.fit_final(frame)
    return res, final


with tabs[5]:
    st.subheader("Customer-level churn prediction")
    st.caption("Gradient-boosting model trained on the loaded dataset. Scores are the estimated probability "
               "that a customer churns. Use them to prioritise outreach, not to restrict service.")
    res, final = get_models(raw[U.REQUIRED])
    gb, lg = res["Gradient boosting"], res["Logistic regression"]
    m = st.columns(4)
    m[0].metric("Test AUC (gradient boosting)", f"{gb['auc']:.3f}", help="Probability a random churner is ranked above a random non-churner (0.5 = chance)")
    m[1].metric("Test AUC (logistic baseline)", f"{lg['auc']:.3f}")
    m[2].metric("Average precision", f"{gb['ap']:.3f}", help="No-skill baseline = churn rate (about 0.20)")
    m[3].metric("Brier score", f"{gb['brier']:.3f}", help="Lower is better; a constant guess scores about 0.16")

    Xall, _ = M.make_xy(df)
    scored = df.copy()
    scored["ChurnProbability"] = final.predict_proba(Xall)[:, 1]
    scored["RiskTier"] = M.risk_tier(scored["ChurnProbability"]).astype(str)
    st.caption("Scores below are for the customers matching the sidebar filters. They come from a model fitted on all customers, "
               "so use the test metrics above (not the scores themselves) to judge accuracy.")

    a, b = st.columns([1, 1.4])
    tier = scored.RiskTier.value_counts().reindex(["High", "Medium", "Low"]).fillna(0).reset_index()
    tier.columns = ["Risk tier", "Customers"]
    a.plotly_chart(px.bar(tier, x="Risk tier", y="Customers", text="Customers", color="Risk tier",
                          color_discrete_map={"High": "#C0392B", "Medium": "#E0A030", "Low": "#1F3A5F"},
                          title="Customers by risk tier (current filters)"), width="stretch")
    b.plotly_chart(px.histogram(scored, x="ChurnProbability", nbins=40, title="Distribution of predicted churn probability",
                                labels={"ChurnProbability": "Predicted churn probability"}), width="stretch")

    st.subheader("Highest-risk customers")
    tiers_sel = st.multiselect("Risk tiers to show", ["High", "Medium", "Low"], default=["High"])
    top_n = st.slider("Rows", 10, 200, 50, 10)
    show = (scored[scored.RiskTier.isin(tiers_sel)].sort_values("ChurnProbability", ascending=False)
            .head(top_n)[["CustomerId", "ChurnProbability", "RiskTier", "Geography", "Gender", "Age",
                          "NumOfProducts", "IsActiveMember", "Balance"]])
    st.dataframe(show.style.format({"ChurnProbability": "{:.1%}", "Balance": "{:,.0f}"}), width="stretch", hide_index=True)
    st.download_button("Download all scored customers (filtered)", scored.to_csv(index=False), "scored_customers.csv")

    st.subheader("What-if scorer")
    st.caption("Describe one customer and see the model's estimated churn probability.")
    c1, c2, c3, c4 = st.columns(4)
    w_geo = c1.selectbox("Country", sorted(data.Geography.unique()))
    w_gen = c2.selectbox("Gender", sorted(data.Gender.unique()))
    w_age = c3.slider("Age", 18, 90, 45)
    w_ten = c4.slider("Tenure (years)", 0, 10, 5)
    c1, c2, c3, c4 = st.columns(4)
    w_bal = c1.number_input("Balance (€)", 0, 300000, 100000, 5000)
    w_sal = c2.number_input("Estimated salary (€)", 0, 250000, 100000, 5000)
    w_prod = c3.selectbox("Number of products", [1, 2, 3, 4], index=0)
    w_cs = c4.slider("Credit score", 300, 900, 650)
    c1, c2, _, _ = st.columns(4)
    w_card = int(c1.checkbox("Has credit card", True))
    w_act = int(c2.checkbox("Active member", True))
    one = pd.DataFrame([dict(CreditScore=w_cs, Geography=w_geo, Gender=w_gen, Age=w_age, Tenure=w_ten, Balance=w_bal,
                             NumOfProducts=w_prod, HasCrCard=w_card, IsActiveMember=w_act, EstimatedSalary=w_sal)])
    p1 = float(final.predict_proba(M.make_xy(one.assign(Exited=0))[0])[0, 1])
    t1 = str(M.risk_tier(pd.Series([p1])).iloc[0])
    r1, r2 = st.columns(2)
    r1.metric("Predicted churn probability", f"{p1:.1%}")
    r2.metric("Risk tier", t1)

# ---- Data
with tabs[6]:
    st.write(f"{len(raw):,} rows × {raw.shape[1]} columns | missing values: {int(raw.isna().sum().sum())}"
             f" | duplicate rows: {int(raw.duplicated().sum())}")
    st.dataframe(df.head(200), width="stretch")
    st.download_button("Download filtered data", df.to_csv(index=False), "filtered_churn.csv")
