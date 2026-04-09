import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import RunReportRequest
from google.oauth2 import service_account

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="SaaS Analytics Dashboard", layout="wide")

PROPERTY_ID = "531231412"
KEY_FILE_LOCATION = "credentials.json"

# -----------------------------
# GA CLIENT
# -----------------------------
@st.cache_resource
def get_ga_client():
    try:
        credentials = service_account.Credentials.from_service_account_file(
            KEY_FILE_LOCATION
        )
        return BetaAnalyticsDataClient(credentials=credentials)
    except:
        return None

client = get_ga_client()

# -----------------------------
# STATUS
# -----------------------------
if client is not None:
    st.success("🟢 Google Analytics Connected Successfull")
else:
    st.error("🔴 Google Analytics Connection Failed! — Using Sample Data")

# -----------------------------
# SAMPLE DATA (ENHANCED)
# -----------------------------
def generate_sample_data(n_users=300):
    np.random.seed(42)

    countries = ["USA", "India", "UK", "Germany", "Canada", "Australia"]
    devices = ["desktop", "mobile", "tablet"]

    base_date = datetime.today() - timedelta(days=30)

    rows = []

    for i in range(n_users):
        user_id = f"user_{i+1}"

        country = np.random.choice(countries)
        device = np.random.choice(devices)

        signup_date = base_date + timedelta(days=np.random.randint(0, 30))

        sessions = np.random.poisson(5)
        event_count = sessions * np.random.randint(3, 10)
        revenue = event_count * np.random.uniform(2, 25)

        rows.append({
            "user_id": user_id,
            "country": country,
            "device": device,
            "activeUsers": np.random.randint(1, 5),
            "sessions": sessions,
            "eventCount": event_count,
            "revenue": revenue,
            "signup_date": signup_date
        })

    df = pd.DataFrame(rows)
    df["date"] = df["signup_date"].dt.date
    return df

# -----------------------------
# FETCH GA DATA
# -----------------------------
@st.cache_data
def fetch_ga_data():
    if client is None:
        st.warning("⚠️ No GA client. Using sample dataset.")
        return generate_sample_data()

    try:
        request = RunReportRequest(
            property=f"properties/{PROPERTY_ID}",
            dimensions=[
                {"name": "country"},
                {"name": "deviceCategory"}
            ],
            metrics=[
                {"name": "activeUsers"},
                {"name": "sessions"},
                {"name": "eventCount"}
            ],
            date_ranges=[{"start_date": "30daysAgo", "end_date": "today"}]
        )

        response = client.run_report(request)

        if not response.rows:
            st.warning("⚠️ No GA data found. Using sample dataset.")
            return generate_sample_data()

        data = []

        for row in response.rows:
            metrics = [m.value for m in row.metric_values]

            data.append({
                "country": row.dimension_values[0].value,
                "device": row.dimension_values[1].value,
                "activeUsers": int(metrics[0]) if metrics[0] else 0,
                "sessions": int(metrics[1]) if metrics[1] else 0,
                "eventCount": int(metrics[2]) if metrics[2] else 0
            })

        df = pd.DataFrame(data)

        if df.empty:
            return generate_sample_data()

        df["revenue"] = df["eventCount"] * np.random.uniform(5, 50, len(df))
        return df

    except Exception as e:
        st.error("❌ GA API failed Intentionally Because Of Wrong Credentials — using sample dataset Scroll down to view the project !!! ")
        st.exception(e)
        return generate_sample_data()

# -----------------------------
# LOAD DATA
# -----------------------------
df = fetch_ga_data()

# SAFE FIX
if "activeUsers" not in df.columns:
    df["activeUsers"] = 0

if "user_id" not in df.columns:
    df["user_id"] = [f"user_{i}" for i in range(len(df))]

# -----------------------------
# FILTERS
# -----------------------------
st.sidebar.header("🔍 Filters")

country_filter = st.sidebar.multiselect(
    "Country",
    df["country"].unique(),
    df["country"].unique()
)

device_filter = st.sidebar.multiselect(
    "Device",
    df["device"].unique(),
    df["device"].unique()
)

filtered_df = df[
    (df["country"].isin(country_filter)) &
    (df["device"].isin(device_filter))
]

# -----------------------------
# TABS
# -----------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview",
    "📈 Trends",
    "🔄 Funnel",
    "🤖 ML",
    "📉 Advanced ML",
    "ℹ️ About"
])

# -----------------------------
# OVERVIEW
# -----------------------------
with tab1:
    st.subheader("Overview Metrics")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Users", int(filtered_df["activeUsers"].sum()))
    col2.metric("Sessions", int(filtered_df["sessions"].sum()))
    col3.metric("Events", int(filtered_df["eventCount"].sum()))
    col4.metric("Revenue", f"${filtered_df['revenue'].sum():.2f}")

# -----------------------------
# TRENDS
# -----------------------------
with tab2:
    st.subheader("Trends")

    if "date" in df.columns:
        trend = df.groupby("date").sum(numeric_only=True).reset_index()

        fig1 = px.line(trend, x="date", y="sessions")
        fig2 = px.line(trend, x="date", y="revenue")

        st.plotly_chart(fig1, use_container_width=True)
        st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# FUNNEL
# -----------------------------
with tab3:
    st.subheader("Funnel")

    funnel_data = pd.DataFrame({
        "Stage": ["Users", "Sessions", "Events"],
        "Value": [
            filtered_df["activeUsers"].sum(),
            filtered_df["sessions"].sum(),
            filtered_df["eventCount"].sum()
        ]
    })

    fig = px.funnel(funnel_data, x="Value", y="Stage")
    st.plotly_chart(fig)

# -----------------------------
# ML (Anomaly)
# -----------------------------
with tab4:
    st.subheader("Anomaly Detection")

    ml_df = filtered_df[["revenue", "sessions", "eventCount"]]

    model = IsolationForest(contamination=0.05)
    filtered_df["anomaly"] = model.fit_predict(ml_df)

    filtered_df["label"] = filtered_df["anomaly"].apply(
        lambda x: "Anomaly" if x == -1 else "Normal"
    )

    fig = px.scatter(filtered_df, x="sessions", y="revenue", color="label")
    st.plotly_chart(fig)

# -----------------------------
# ADVANCED ML
# -----------------------------
with tab5:
    st.subheader("Advanced ML")

    # Cohort
    st.markdown("### Cohort Analysis")

    df["signup_date"] = pd.to_datetime(df["signup_date"], errors="coerce")
    df["cohort"] = df.groupby("user_id")["signup_date"].transform("min")

    cohort = df.groupby(["cohort"])["user_id"].nunique().reset_index()
    st.dataframe(cohort)

    # Churn
    st.markdown("### Churn Prediction")

    df_ml = df.copy()
    df_ml["churn"] = (df_ml["sessions"] < df_ml["sessions"].median()).astype(int)

    X = df_ml[["sessions", "eventCount", "revenue"]]
    y = df_ml["churn"]

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    churn_model = RandomForestClassifier()
    churn_model.fit(X_train, y_train)

    df_ml["churn_prob"] = churn_model.predict_proba(X)[:, 1]

    fig = px.histogram(df_ml, x="churn_prob")
    st.plotly_chart(fig)

    # Forecast
    st.markdown("### Revenue Forecast")

    ts = df.groupby("date")["revenue"].sum().reset_index()
    ts["date"] = pd.to_datetime(ts["date"])
    ts["day"] = (ts["date"] - ts["date"].min()).dt.days

    model = LinearRegression()
    model.fit(ts[["day"]], ts["revenue"])
    ts["forecast"] = model.predict(ts[["day"]])

    fig = px.line(ts, x="date", y=["revenue", "forecast"])
    st.plotly_chart(fig)

# -----------------------------
# ABOUT
# -----------------------------
with tab6:
    st.title("About This Project")

    st.markdown("""
This SaaS Analytics Dashboard integrates:

- Google Analytics API
- Synthetic SaaS dataset fallback
- Machine Learning models

### ML Models Used:

- Isolation Forest → anomaly detection  
- Random Forest → churn prediction  
- Linear Regression → revenue forecasting  

### What is being analyzed?

- User behavior  
- Revenue trends  
- Engagement metrics  
- Retention & churn  

### Business Value:

- Identify anomalies early  
- Predict churn risk  
- Forecast revenue  
- Understand cohorts  
- Improve decision making  

This is a production-style analytics system combining:
data engineering + analytics + machine learning.
""")
