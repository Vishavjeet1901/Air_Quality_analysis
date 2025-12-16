import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


# App Configuration
st.set_page_config(
    page_title="Air Quality Analysis",
    layout="wide"
)

st.title("Air Quality Analysis & AQI Prediction")


# Load Data and Model
@st.cache_data
def load_data():
    return pd.read_csv("data.csv")

@st.cache_resource
def load_model():
    model = joblib.load("models/aqi_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    features = joblib.load("models/features.pkl")
    return model, scaler, features


df = load_data()
model, scaler, features = load_model()

# Convert Date
df['Date'] = pd.to_datetime(
    df['Date'],
    format="mixed",
    dayfirst=True,
    errors="coerce"
)



# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Data Overview", "Exploratory Data Analysis", "Modelling & Prediction"]
)


# Data Overview
if page == "Data Overview":
    st.header("Data Overview")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Shape")
    st.write(f"Rows: {df.shape[0]}")
    st.write(f"Columns: {df.shape[1]}")

    st.subheader("Column Information")
    st.write(df.dtypes)

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    st.subheader("Summary Statistics")
    st.write(df.describe())


# Page 2: Exploratory Data Analysis
elif page == "Exploratory Data Analysis":
    st.header("Exploratory Data Analysis")

    st.subheader("AQI Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['AQI'], bins=40, kde=True, ax=ax)
    st.pyplot(fig)

    st.subheader("AQI Categories")
    fig, ax = plt.subplots()
    df['AQI_Bucket'].value_counts().plot(kind='bar', ax=ax)
    st.pyplot(fig)

    st.subheader("PM2.5 vs AQI")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='PM2.5', y='AQI', alpha=0.5, ax=ax)
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, cmap="coolwarm", linewidths=0.5, ax=ax)
    st.pyplot(fig)


# Modelling & Prediction
elif page == "Modelling & Prediction":
    st.header("AQI Prediction")

    st.write("Enter pollutant values to predict AQI:")

    user_input = {}

    col1, col2, col3 = st.columns(3)

    for i, feature in enumerate(features):
        if i % 3 == 0:
            user_input[feature] = col1.number_input(feature, min_value=0.0)
        elif i % 3 == 1:
            user_input[feature] = col2.number_input(feature, min_value=0.0)
        else:
            user_input[feature] = col3.number_input(feature, min_value=0.0)

    if st.button("Predict AQI"):
        input_df = pd.DataFrame([user_input])
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)

        st.success(f"Predicted AQI: {int(prediction[0])}")
