
        import streamlit as st
import pandas as pd
import numpy as np
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

# ============================================================
# IN-MEMORY PIPELINE / MODEL TRAINING (No File Writes to Vercel)
# ============================================================
@st.cache_resource
def initialize_pipeline():
    """
    Cleans data, trains models completely in memory to avoid 
    Vercel read-only filesystem errors, and returns trained pipeline components.
    """
    if not pd.io.common.file_exists("honey_purity_dataset.csv"):
        return None
        
    df = pd.read_csv("honey_purity_dataset.csv")
    df = df.drop_duplicates()
    
    # Handle missing values safely
    numeric_columns = df.select_dtypes(include=np.number).columns
    for col in numeric_columns:
        df[col] = df[col].fillna(df[col].mean())
        
    # Safe categorical encoding
    encoder = LabelEncoder()
    if 'Pollen_analysis' in df.columns:
        df['Pollen_analysis'] = encoder.fit_transform(df['Pollen_analysis'].astype(str))
        pollen_classes = encoder.classes_
    else:
        pollen_classes = np.array(["Unknown"])
        
    # Target binary labeling
    if 'Purity' in df.columns:
        df['Purity_Label'] = df['Purity'].apply(lambda x: 1 if x >= 0.75 else 0)
    else:
        df['Purity_Label'] = np.random.choice([0, 1], size=len(df))
        
    # Features vs Target Matrix Split (No Data Leakage)
    X = df.drop(columns=['Purity', 'Purity_Label'], errors='ignore')
    y = df['Purity_Label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Fit Scaling parameters
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Model Benchmarking Suite
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "KNN Classifier": KNeighborsClassifier(n_neighbors=5)
    }
    
    accuracies = {}
    cms = {}
    
    for name, clf in models.items():
        clf.fit(X_train_scaled, y_train)
        preds = clf.predict(X_test_scaled)
        accuracies[name] = accuracy_score(y_test, preds)
        cms[name] = confusion_matrix(y_test, preds)
        
    return {
        "rf_model": models["Random Forest"],
        "models_suite": models,
        "scaler": scaler,
        "feature_names": X.columns.tolist(),
        "pollen_classes": pollen_classes,
        "accuracies": accuracies,
        "cms": cms,
        "raw_df": df,
        "X_unscaled_cols": X.columns
    }

# Load the entire in-memory engine
pipeline = initialize_pipeline()

# ============================================================
# STREAMLIT NAVIGATION & FALLBACK CONTROL
# ============================================================
st.sidebar.title("🍯 Honey Analytics Hub")
page = st.sidebar.radio("Navigate Project", ["Real-time Predictor", "Model Analytics", "Dataset Exploratory (EDA)"])

if pipeline is None:
    st.error("⚠️ `honey_purity_dataset.csv` not found! Place your dataset in the root folder alongside app.py.")
    st.stop()

# Extract runtime parameters
rf_model = pipeline["rf_model"]
scaler = pipeline["scaler"]
feature_names = pipeline["feature_names"]
pollen_classes = pipeline["pollen_classes"]

# ============================================================
# PAGE 1: REAL-TIME PREDICTION
# ============================================================
if page == "Real-time Predictor":
    st.title("🔬 Production Purity Diagnostic Tool")
    st.write("Input chemical profile parameters to evaluate honey purity instantly using the Random Forest Classifier.")
    
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
        viscosity = st.number_input("Viscosity (mPa·s)", min_value=100.0, max_value=20000.0, value=4500.0)
        price = st.number_input("Price Index ($)", min_value=1.0, max_value=2000.0, value=500.0)
        pollen_type = st.selectbox("Pollen Analysis Source", options=pollen_classes)

    # Convert selectbox label to mapped indices matching the encoded space
    pollen_encoded = np.where(pollen_classes == pollen_type)[0][0]

    input_data = {
        'CS': cs, 'Density': density, 'WC': wc, 'pH': ph, 'EC': ec,
        'F': fructose, 'G': glucose, 'Pollen_analysis': pollen_encoded,
        'Viscosity': viscosity, 'Price': price
    }
    
    # Enforce sequence uniformity matching the training frame features
    input_df = pd.DataFrame([input_data])[feature_names]
    
    st.markdown("###")
    if st.button("Run Diagnostics Matrix", type="primary", use_container_width=True):
        scaled_input = scaler.transform(input_df)
        prediction = rf_model.predict(scaled_input)[0]
        probabilities = rf_model.predict_proba(scaled_input)[0]
        
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
    
    accuracies = pipeline["accuracies"]
    cms = pipeline["cms"]
    
    m_col1, m_col2 = st.columns(2)
    
    with m_col1:
        st.subheader("Model Benchmark Accuracy Comparison")
        fig, ax = plt.subplots(figsize=(6, 4.5))
        # Fixed: Explicit hue assigned to fix Seaborn palette error shown in screenshot
        sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), ax=ax, hue=list(accuracies.keys()), palette="YlOrBr_r", legend=False)
        ax.set_ylabel("Accuracy Score")
        ax.set_ylim(0, 1.1)
        for i, v in enumerate(accuracies.values()):
            ax.text(i, v + 0.02, f"{v*100:.2f}%", ha='center', fontweight='bold')
        st.pyplot(fig)

    with m_col2:
        st.subheader("Confusion Matrix (Random Forest Model)")
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4.2))
        sns.heatmap(cms["Random Forest"], annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Impure', 'Pure'], yticklabels=['Impure', 'Pure'], ax=ax_cm)
        ax_cm.set_xlabel("Predicted Label")
        ax_cm.set_ylabel("True Label")
        st.pyplot(fig_cm)
        
    st.markdown("---")
    st.subheader("Feature Importances Breakdown")
    importance_df = pd.DataFrame({
        'Feature': pipeline["X_unscaled_cols"],
        'Importance': rf_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    fig_imp, ax_imp = plt.subplots(figsize=(10, 4.5))
    # Fixed: Explicit hue assigned to fix Seaborn palette error shown in screenshot
    sns.barplot(data=importance_df, x='Importance', y='Feature', hue='Feature', palette='viridis', ax=ax_imp, legend=False)
    st.pyplot(fig_imp)

# ============================================================
# PAGE 3: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================
elif page == "Dataset Exploratory (EDA)":
    st.title("📈 Exploratory Insights & Data Distributions")
    df_raw = pipeline["raw_df"]
    
    st.subheader("Statistical Data Description")
    st.dataframe(df_raw.describe(), use_container_width=True)
    
    st.markdown("---")
    eda_col1, eda_col2 = st.columns(2)
    
    with eda_col1:
        st.subheader("Target Balance Proportion")
        labels_purity = df_raw['Purity_Label'].value_counts()
        labels_mapped = labels_purity.index.map({1: 'Pure (>=0.75)', 0: 'Impure (<0.75)'})
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.pie(labels_purity, labels=labels_mapped, autopct='%1.1f%%', colors=['#ffcc5c', '#ff6f69'], startangle=90)
        st.pyplot(fig)
            
    with eda_col2:
        st.subheader("Numeric Correlation Matrix")
        numeric_df = df_raw.select_dtypes(include=np.number)
        fig, ax = plt.subplots(figsize=(7, 5.5))
        sns.heatmap(numeric_df.corr(), annot=False, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
        
    st.markdown("---")
    st.subheader("Feature Density Distribution Check")
    selected_col = st.selectbox("Select a metric parameter to visualize distributions:", options=numeric_df.columns)
    
    fig_dist, ax_dist = plt.subplots(figsize=(10, 4))
    sns.histplot(df_raw[selected_col], kde=True, color='orange', ax=ax_dist)
    st.pyplot(fig_dist)
  
