import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Behavior Analytics & Churn Prediction Dashboard")

@st.cache_data
def load_data():
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    df["TotalCharges"].fillna(
        df["TotalCharges"].median(),
        inplace=True
    )

    return df

df = load_data()

# Feature Engineering
df["AvgMonthlySpend"] = (
    df["TotalCharges"] /
    (df["tenure"] + 1)
)

df["LongTermCustomer"] = (
    df["tenure"] >= 24
).astype(int)

df["ContractRisk"] = (
    df["Contract"] == "Month-to-month"
).astype(int)

service_cols = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies"
]

df["ServiceCount"] = 0

for col in service_cols:
    df["ServiceCount"] += (
        df[col] == "Yes"
    ).astype(int)

df["SecuritySupportScore"] = (
    (df["OnlineSecurity"] == "Yes").astype(int)
    +
    (df["TechSupport"] == "Yes").astype(int)
)

page = st.sidebar.selectbox(
    "Navigation",
    [
        "Dataset Overview",
        "EDA",
        "Customer Segmentation",
        "Churn Prediction",
        "Business Insights"
    ]
)

if page == "Dataset Overview":

    st.header("Dataset Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Total Customers",
            len(df)
        )

    with col2:
        st.metric(
            "Total Features",
            df.shape[1]
        )

    st.subheader("Dataset Sample")
    st.dataframe(df.head())

    st.subheader("Summary Statistics")
    st.dataframe(df.describe())

elif page == "EDA":

    st.header("Exploratory Data Analysis")

    st.subheader("Churn Distribution")

    fig, ax = plt.subplots()

    sns.countplot(
        x="Churn",
        data=df,
        ax=ax
    )

    st.pyplot(fig)

    st.subheader("Monthly Charges vs Churn")

    fig, ax = plt.subplots(figsize=(8,5))

    sns.boxplot(
        x="Churn",
        y="MonthlyCharges",
        data=df,
        ax=ax
    )

    st.pyplot(fig)

    st.subheader("Tenure vs Churn")

    fig, ax = plt.subplots(figsize=(8,5))

    sns.boxplot(
        x="Churn",
        y="tenure",
        data=df,
        ax=ax
    )

    st.pyplot(fig)

    st.subheader("Contract Type vs Churn")

    fig, ax = plt.subplots(figsize=(8,5))

    sns.countplot(
        x="Contract",
        hue="Churn",
        data=df,
        ax=ax
    )

    plt.xticks(rotation=15)

    st.pyplot(fig)

elif page == "Customer Segmentation":

    st.header("Customer Segmentation")

    segment_data = df[
        [
            "TotalCharges",
            "MonthlyCharges",
            "tenure"
        ]
    ]

    scaler = StandardScaler()

    segment_scaled = scaler.fit_transform(
        segment_data
    )

    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    df["Cluster"] = kmeans.fit_predict(
        segment_scaled
    )

    fig, ax = plt.subplots(figsize=(8,6))

    sns.scatterplot(
        data=df,
        x="MonthlyCharges",
        y="TotalCharges",
        hue="Cluster",
        palette="Set1",
        ax=ax
    )

    st.pyplot(fig)

    st.subheader("Cluster Counts")

    st.dataframe(
        df["Cluster"]
        .value_counts()
        .reset_index()
    )

elif page == "Churn Prediction":

    st.header("Churn Prediction")

    temp_df = df.copy()

    temp_df["Churn"] = (
        temp_df["Churn"] == "Yes"
    ).astype(int)

    X = pd.get_dummies(
        temp_df.drop(
            columns=["customerID", "Churn"]
        ),
        drop_first=True
    )

    y = temp_df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        max_depth=10
    )

    model.fit(X_train, y_train)

    accuracy = model.score(
        X_test,
        y_test
    )

    st.success(
        f"Model Accuracy: {accuracy:.2%}"
    )

    churn_prob = model.predict_proba(X)[:,1]

    temp_df["ChurnProbability"] = churn_prob

    st.subheader("Top 20 Highest Risk Customers")

    high_risk = temp_df.sort_values(
        "ChurnProbability",
        ascending=False
    )

    st.dataframe(
        high_risk[
            [
                "customerID",
                "tenure",
                "MonthlyCharges",
                "ChurnProbability"
            ]
        ].head(20)
    )

elif page == "Business Insights":

    st.header("Business Insights")

    churn_rate = (
        (df["Churn"] == "Yes")
        .mean()
        * 100
    )

    st.metric(
        "Overall Churn Rate",
        f"{churn_rate:.2f}%"
    )

    avg_monthly = round(
        df["MonthlyCharges"].mean(),
        2
    )

    st.metric(
        "Average Monthly Charges",
        avg_monthly
    )

    revenue_risk = (
        df[
            df["Contract"] == "Month-to-month"
        ]["MonthlyCharges"]
        .sum()
    )

    st.metric(
        "Revenue at Risk",
        f"${revenue_risk:,.2f}"
    )

    st.subheader("Key Findings")

    st.markdown("""
    - Month-to-month customers show the highest churn.
    - Low tenure customers are more likely to leave.
    - High monthly charges correlate with churn.
    - Customers without tech support churn more often.
    - Long-term customers are significantly more loyal.
    """)

    contract_data = (
        df["Contract"]
        .value_counts()
    )

    st.subheader("Contract Distribution")

    st.bar_chart(contract_data)