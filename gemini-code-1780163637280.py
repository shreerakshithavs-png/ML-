import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ============================================================
# CONFIG & THEME SETUP
# ============================================================
st.set_page_config(page_title="Advanced Honey Purity Dashboard", page_icon="🍯", layout="wide")

MODEL_PATH = "honey_rf_model.pkl"
SCALER_PATH = "scaler.pkl"
FEATURES_PATH = "features.pkl"
CLASSES_PATH = "pollen_classes.pkl"

# ============================================================
# AUTONOMOUS PIPELINE / MODEL TRAINING
# ============================================================
def run_training_pipeline():
    """Cleans data, fixes leakage, trains model, and saves artifacts."""
    if not os.path.exists("honey_purity_dataset.csv"):
        return False
        
    df = pd.read_csv("honey_purity_dataset.csv")
    df.drop_duplicates(inplace=True)
    
    # Handle missing values safely
    numeric_columns = df.select_dtypes(include=np.number).columns
    for col in numeric_columns:
        df[col] = df[col].fillna(df[col].mean())
        
    # Encode categorical data
    encoder = LabelEncoder()
    if 'Pollen_analysis' in df.columns:
        df['Pollen_analysis'] = encoder.fit_transform(df['Pollen_analysis'].astype(str))
        joblib.dump(encoder.classes_, CLASSES_PATH)
        
    # Create target label
    if 'Purity' in df.columns:
        df['Purity_Label'] = df['Purity'].apply(lambda x: 1 if x >= 0.75 else 0)
        
    # Splitting Features & Targets properly (Fixing Data Leakage)
    # Note: 'Purity' must be dropped from features so the model doesn't 'cheat'
    X = df.drop(columns=['Purity', 'Purity_Label'], errors='ignore')
    y = df['Purity_Label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Feature Scaling after the split to avoid leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train primary production model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    
    # Save the pipeline artifacts
    joblib.dump(rf_model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(X.columns.tolist(), FEATURES_PATH)
    
    return df, X_test_scaled, y_test, rf_model, scaler, X.columns.tolist()

# Run/Load the pipeline quietly behind the scenes
pipeline_success = True
if not os.path.exists(MODEL_PATH):
    with st.spinner("Initializing Pipeline & Training Models..."):
        res = run_training_pipeline()
        if res is False:
            pipeline_success = False

# Load data for visualization metrics if dataset exists
@st.cache_data
def load_raw_data():
    if os.path.exists("honey_purity_dataset.csv"):
        return pd.read_csv("honey_purity_dataset.csv")
    return None

df_raw = load_raw_data()

# ============================================================
# STREAMLIT UI SIDEBAR & NAVIGATION
# ============================================================
st.sidebar.title("🍯 Honey Analytics Hub")
page = st.sidebar.radio("Navigate Project", ["Real-time Predictor", "Model Analytics", "Dataset Exploratory (EDA)"])

if not pipeline_success or df_raw is None:
    st.error("⚠️ `honey_purity_dataset.csv` not found! Place your dataset in the root directory to run analytics.")
    st.stop()

# Load necessary production artifacts
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_names = joblib.load(FEATURES_PATH)
pollen_classes = joblib.load(CLASSES_PATH)

# ============================================================
# PAGE 1: REAL-TIME PREDICTION INDEX
# ============================================================
if page == "Real-time Predictor":
    st.title("🔬 Production Purity Diagnostic Tool")
    st.write("Input a honey sample's chemical profile parameters to evaluate purity instantly.")
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        cs = st.slider("Color Score (CS)", 1.0, 10.0, 5.0, step=0.1)
        density = st.slider("Density (g/cm³)", 1.1, 1.9, 1.4, step=0.01)
        wc = st.slider("Water Content (%)", 12.0, 25.0, 18.0, step=0.1)
        ph = st.slider("pH Level", 3.0, 8.0, 4.5, step=0.1)
        ec = st.slider("Electrical Conductivity (EC)", 0.5, 1.5, 0.8, step=0.01)

    with col2:
        fructose = st.slider("Fructose Level", 20.0, 50.0, 35.0, step=0.1)
        glucose = st.slider("Glucose Level", 20.0, 50.0, 30.0, step=0.1)
        viscosity = st.number_input("Viscosity (mPa·s)", min_value=1000.0, max_value=15000.0, value=4500.0)
        price = st.number_input("Price Index ($)", min_value=10.0, max_value=2000.0, value=500.0)
        pollen_type = st.selectbox("Pollen Analysis Source", options=pollen_classes)

    pollen_encoded = np.where(pollen_classes == pollen_type)[0][0]

    # Map exact order of keys trained during fit mapping
    input_data = {
        'CS': cs, 'Density': density, 'WC': wc, 'pH': ph, 'EC': ec,
        'F': fructose, 'G': glucose, 'Pollen_analysis': pollen_encoded,
        'Viscosity': viscosity, 'Price': price
    }
    
    # Process inputs exactly like training data structure
    input_df = pd.DataFrame([input_data])[feature_names]
    
    st.markdown("###")
    if st.button("Run Diagnostics Matrix", type="primary", use_container_width=True):
        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0]
        probabilities = model.predict_proba(scaled_input)[0]
        
        st.markdown("---")
        if prediction == 1:
            st.success(f"### ✅ Result: The Honey Sample is PURE! (Confidence: {probabilities[1]*100:.2f}%)")
        else:
            st.error(f"### ❌ Result: The Honey Sample is IMPURE / ADULTERATED! (Confidence: {probabilities[0]*100:.2f}%)")

# ============================================================
# PAGE 2: MODEL PERFORMANCE ANALYTICS
# ============================================================
elif page == "Model Analytics":
    st.title("📊 Pipeline & Algorithm Analytics Comparison")
    
    # Run dynamic performance comparison safely isolated from data leakage
    df_clean = df_raw.copy().drop_duplicates()
    if 'Pollen_analysis' in df_clean.columns:
        df_clean['Pollen_analysis'] = LabelEncoder().fit_transform(df_clean['Pollen_analysis'].astype(str))
    if 'Purity' in df_clean.columns:
        df_clean['Purity_Label'] = df_clean['Purity'].apply(lambda x: 1 if x >= 0.75 else 0)
        
    X_unscaled = df_clean.drop(columns=['Purity', 'Purity_Label'], errors='ignore')
    y_target = df_clean['Purity_Label']
    
    X_tr, X_te, y_tr, y_te = train_test_split(X_unscaled, y_target, test_size=0.2, random_state=42, stratify=y_target)
    
    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr)
    X_te_sc = sc.transform(X_te)
    
    # Models benchmarking
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "KNN Classifier": KNeighborsClassifier(n_neighbors=5)
    }
    
    accuracies = {}
    reports = {}
    cms = {}
    
    for name, clf in models.items():
        clf.fit(X_tr_sc, y_tr)
        preds = clf.predict(X_te_sc)
        accuracies[name] = accuracy_score(y_te, preds)
        reports[name] = classification_report(y_te, preds, output_dict=True)
        cms[name] = confusion_matrix(y_te, preds)
        
    # UI Layout columns
    m_col1, m_col2 = st.columns([1, 1])
    
    with m_col1:
        st.subheader("Model Benchmark Accuracy Comparison")
        fig, ax = plt.subplots(figsize=(6, 4.2))
        sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), ax=ax, palette="Blues_r")
        ax.set_ylabel("Accuracy Score")
        ax.set_ylim(0, 1.05)
        for i, v in enumerate(accuracies.values()):
            ax.text(i, v + 0.02, f"{v*100:.2f}%", ha='center', fontweight='bold')
        st.pyplot(fig)

    with m_col2:
        st.subheader("Confusion Matrix (Production RF Model)")
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        sns.heatmap(cms["Random Forest"], annot=True, fmt='d', cmap='Oranges', cbar=False,
                    xticklabels=['Impure', 'Pure'], yticklabels=['Impure', 'Pure'], ax=ax_cm)
        ax_cm.set_xlabel("Predicted Label")
        ax_cm.set_ylabel("True Label")
        st.pyplot(fig_cm)
        
    st.markdown("---")
    st.subheader("Feature Importances Breakdown")
    rf_prod = models["Random Forest"]
    importance_df = pd.DataFrame({
        'Feature': X_unscaled.columns,
        'Importance': rf_prod.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    fig_imp, ax_imp = plt.subplots(figsize=(10, 4))
    sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis', ax=ax_imp)
    st.pyplot(fig_imp)

# ============================================================
# PAGE 3: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================
elif page == "Dataset Exploratory (EDA)":
    st.title("📈 Exploratory Insights & Data Distributions")
    
    st.subheader("Statistical Glimpse")
    st.dataframe(df_raw.describe(), use_container_width=True)
    
    st.markdown("---")
    eda_col1, eda_col2 = st.columns(2)
    
    with eda_col1:
        st.subheader("Target Balanced Distribution")
        if 'Purity' in df_raw.columns:
            labels_purity = df_raw['Purity'].apply(lambda x: 'Pure (>=0.75)' if x >= 0.75 else 'Impure (<0.75)').value_counts()
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.pie(labels_purity, labels=labels_purity.index, autopct='%1.1f%%', colors=['#ffcc5c', '#ff6f69'], startangle=90)
            st.pyplot(fig)
            
    with eda_col2:
        st.subheader("Correlation Heatmap Matrix")
        numeric_df = df_raw.select_dtypes(include=np.number)
        fig, ax = plt.subplots(figsize=(7, 5.5))
        sns.heatmap(numeric_df.corr(), annot=False, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
        
    st.markdown("---")
    st.subheader("Feature Variable Distribution Inspector")
    selected_col = st.selectbox("Choose feature parameter to visualize distribution trend:", options=df_raw.select_dtypes(include=np.number).columns)
    
    fig_dist, ax_dist = plt.subplots(figsize=(10, 4))
    sns.histplot(df_raw[selected_col], kde=True, color='orange', ax=ax_dist)
    st.pyplot(fig_dist)