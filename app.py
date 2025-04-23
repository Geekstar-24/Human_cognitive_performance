
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from mpl_toolkits.mplot3d import Axes3D
import warnings
import io
import joblib
import os
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

warnings.filterwarnings("ignore")
st.set_page_config(layout="wide")
st.title("🧠 Human Cognitive Performance Analysis Dashboard")

# Load data directly from a path
@st.cache_data
def load_data():
    df = pd.read_csv("./human_cognitive_performance.csv")
    return df

df = load_data()

# Show pandas inspection tools
st.header("📑 Data Overview")

if st.checkbox("🔍 Show df.head()"):
    st.subheader("Top 5 Rows")
    st.dataframe(df.head())

if st.checkbox("🔍 Show df.tail()"):
    st.subheader("Bottom 5 Rows")
    st.dataframe(df.tail())

if st.checkbox("🔢 Show df.nunique()"):
    st.subheader("Unique Value Counts")
    st.dataframe(df.nunique().to_frame(name="Unique Values"))

if st.checkbox("🧾 Show df.info()"):
    st.subheader("DataFrame Info")
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

if st.checkbox("🧮 Show df.describe()"):
    st.subheader("Statistical Summary")
    st.dataframe(df.describe())

if st.checkbox("🚫 Show Null Values"):
    st.subheader("Missing Values per Column")
    st.dataframe(df.isnull().sum().to_frame(name="Null Count"))

if st.checkbox("🗂️ Show Columns"):
    st.subheader("Column Names")
    st.write(df.columns.tolist())

# Label Encoding
le = LabelEncoder()
if 'Gender' in df.columns:
    df['Gender'] = le.fit_transform(df['Gender'])
if 'Diet_Type' in df.columns:
    df['Diet_Type'] = le.fit_transform(df['Diet_Type'])

st.markdown("---")
st.header("📊 Visualizations")

if st.checkbox("1️⃣ Correlation Heatmap"):
    st.subheader("Correlation Matrix")
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='Spectral', fmt=".2f", ax=ax)
    st.pyplot(fig)

if st.checkbox("2️⃣ KDE Plot: Sleep Duration by Gender"):
    st.subheader("KDE: Sleep Duration Distribution by Gender")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.kdeplot(data=df, x="Sleep_Duration", hue="Gender", fill=True, common_norm=False, alpha=0.6)
    plt.title("Sleep Duration Distribution by Gender", fontsize=14)
    st.pyplot(fig)

if st.checkbox("3️⃣ Boxplot: Diet Type vs Cognitive Score"):
    st.subheader("Cognitive Score by Diet Type")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='Diet_Type', y='Cognitive_Score', data=df, palette='Set3', ax=ax)
    st.pyplot(fig)

if st.checkbox("4️⃣ Scatterplot: Caffeine vs Reaction Time"):
    st.subheader("Reaction Time vs Caffeine Intake")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='Caffeine_Intake', y='Reaction_Time', data=df, hue='Gender', palette='cool', ax=ax)
    st.pyplot(fig)

if st.checkbox("5️⃣ Regression: Age vs Cognitive Score"):
    st.subheader("Cognitive Score by Age")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.regplot(x='Age', y='Cognitive_Score', data=df, color='indigo', scatter_kws={'alpha': 0.4}, ax=ax)
    st.pyplot(fig)

if st.checkbox("6️⃣ Violin Plot: Exercise vs Cognitive Score"):
    st.subheader("Cognitive Score by Exercise Frequency")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(x='Exercise_Frequency', y='Cognitive_Score', data=df, palette='Pastel1', ax=ax)
    st.pyplot(fig)

if st.checkbox("7️⃣ Pairplot of Core Features"):
    st.subheader("Pairwise Feature Distributions")
    core_features = ['Age', 'Sleep_Duration', 'Stress_Level', 'Cognitive_Score']
    fig = sns.pairplot(df[core_features], corner=True, diag_kind='kde', palette='coolwarm')
    st.pyplot(fig)

if st.checkbox("8️⃣ Boxen Plot: Stress Level vs Cognitive Score"):
    st.subheader("Cognitive Score by Stress Level")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxenplot(x='Stress_Level', y='Cognitive_Score', data=df, palette='coolwarm', ax=ax)
    st.pyplot(fig)

if st.checkbox("9️⃣ Histogram: Cognitive Score Distribution"):
    st.subheader("Distribution of Cognitive Score")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['Cognitive_Score'], kde=True, bins=20, color='skyblue', ax=ax)
    st.pyplot(fig)

if st.checkbox("🔟 Count Plot: Exercise Frequency"):
    st.subheader("Exercise Frequency Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x='Exercise_Frequency', data=df, palette='spring', ax=ax)
    st.pyplot(fig)

if st.checkbox("🔢 3D Scatter: Memory vs Reaction vs Cognitive Score"):
    st.subheader("3D Plot: Memory vs Reaction vs Cognitive Score")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df['Memory_Test_Score'], df['Reaction_Time'], df['Cognitive_Score'],
               c=df['Cognitive_Score'], cmap='viridis')
    ax.set_xlabel('Memory Test Score')
    ax.set_ylabel('Reaction Time')
    ax.set_zlabel('Cognitive Score')
    st.pyplot(fig)

st.markdown("---")
st.header("📊 Model Performance Summary")

# Static metrics table
metrics_data = {
    "Model": [
        "Linear Regression", "Linear Regression (Optimized)",
        "Ridge Regression", "Ridge Regression (Optimized)",
        "Decision Tree", "Random Forest", "Gradient Boosting",
        "XGBoost", "XGBoost (Optimized)",
        "CatBoost", "CatBoost (Optimized)"
    ],
    "MSE": [
        6.4666014346837635, 6.4666014346837635,
        6.466601470087294, 6.466601791772979,
        13.990707856250001, 6.956420698152501,
        7.004200221769708, 3.671468840645579,
        3.8302324272519277, 1.7438930658859735,
        4.151986407390175
    ],
    "R2 Score": [
        0.9876744492723173, 0.9876744492723173,
        0.987674449204837, 0.9876744485916937,
        0.9733332599604028, 0.9867408379711942,
        0.9866497686594332, 0.9930020620727035,
        0.9926994535603048, 0.9966760836175946,
        0.9920861801052937
    ]
}
metrics_df = pd.DataFrame(metrics_data)
st.dataframe(metrics_df)

# 🔮 Prediction Input Section
st.header("🔮 Predict Cognitive Score Using Saved Models")

with st.form("prediction_form"):
    st.subheader("📥 Enter Feature Values")

    age = st.number_input("Age", min_value=0, max_value=100, value=25)
    gender = st.selectbox("Gender", ["Male", "Female"])
    exercise = st.selectbox("Exercise Frequency", ["Rarely", "Sometimes", "Often", "Daily"])
    sleep = st.number_input("Sleep Duration (hours)", min_value=0.0, max_value=24.0, value=7.0)
    stress = st.slider("Stress Level (1-10)", 1, 10, 5)
    screen = st.number_input("Daily Screen Time (hrs)", min_value=0.0, max_value=24.0, value=6.0)
    caffeine = st.number_input("Caffeine Intake (mg)", min_value=0.0, max_value=1000.0, value=100.0)
    reaction = st.number_input("Reaction Time (ms)", min_value=0.0, max_value=1000.0, value=250.0)
    memory = st.number_input("Memory Test Score", min_value=0.0, max_value=100.0, value=75.0)
    ai_score = st.number_input("AI Predicted Score", min_value=0.0, max_value=100.0, value=50.0)

    submit = st.form_submit_button("🔍 Predict")

if submit:
    st.subheader("📈 Prediction Results")

    gender_encoded = 1 if gender == "Male" else 0
    exercise_map = {"Rarely": 0, "Sometimes": 1, "Often": 2, "Daily": 3}
    exercise_encoded = exercise_map[exercise]

    input_df = pd.DataFrame([[age, gender_encoded, exercise_encoded, sleep, stress, screen,
                              caffeine, reaction, memory, ai_score]],
                            columns=['Age','Gender','Exercise_Frequency','Sleep_Duration','Stress_Level',
                                     'Daily_Screen_Time','Caffeine_Intake','Reaction_Time',
                                     'Memory_Test_Score','AI_Predicted_Score'])

    model_dir = "saved_models"
    if not os.path.exists(model_dir):
        st.error("⚠️ No saved models found in `saved_models/`")
    else:
        model_files = [f for f in os.listdir(model_dir) if f.endswith(".joblib")]
        if not model_files:
            st.warning("❌ No .joblib models in saved_models/")
        else:
            for file in sorted(model_files):
                model_path = os.path.join(model_dir, file)
                try:
                    model_obj = joblib.load(model_path)
                    model = model_obj["model"] if isinstance(model_obj, dict) and "model" in model_obj else model_obj
                    prediction = model.predict(input_df)[0]
                    model_name = file.replace(".joblib", "").replace("_", " ").title()
                    st.markdown(f"**{model_name}** : Predicted Cognitive Score : `{prediction:.2f}`")
                except Exception as e:
                    st.error(f"❌ Error using {file}: {e}")