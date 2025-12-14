
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA

# Load breast cancer dataset
data = load_breast_cancer()
features = data.feature_names
X = pd.DataFrame(data.data, columns=features)
y = pd.Series(data.target)

# Feature Descriptions (short)
feature_descriptions = {
    "radius": "Mean distance from center to points on the perimeter",
    "texture": "Standard deviation of gray-scale values",
    "perimeter": "Perimeter of the nucleus",
    "area": "Area of the nucleus",
    "smoothness": "Local variation in radius lengths",
    "compactness": "(Perimeter² / Area) - 1.0",
    "concavity": "Severity of concave portions of the contour",
    "concave points": "Number of concave portions of the contour",
    "symmetry": "Symmetry of the cell nuclei",
    "fractal_dimension": "Roughness or complexity of the contour"
}

# -------- SIDEBAR --------
st.sidebar.title("⚙️ Settings")
model_option = st.sidebar.selectbox("Choose a Model", ["Random Forest", "K-Nearest Neighbors"])

# Load model and scaler
model_file = "breast_cancer_model.pkl" if model_option == "Random Forest" else "breast_cancer_knn.pkl"
model = joblib.load(model_file)
scaler = joblib.load("scaler.pkl")

# Sliders for feature input (all in sidebar)
st.sidebar.markdown("### 🧮 Input Feature Sliders")
user_input = []
for feature in features:
    val = st.sidebar.slider(
        label=feature,
        min_value=float(X[feature].min()),
        max_value=float(X[feature].max()),
        value=float(X[feature].mean()),
        step=0.01
    )
    user_input.append(val)

# -------- MAIN PAGE --------
st.title("🔬 Breast Cancer Prediction Dashboard")
st.markdown("Use this app to predict whether a tumor is **malignant** (cancerous) or **benign** (non-cancerous) using a trained machine learning model.")

with st.expander("ℹ️ How to Use"):
    st.markdown("""
    - Use the sidebar to choose a model and input feature values.
    - The app will predict the tumor type and show confidence.
    - Visualizations below show how your input compares to real cases.
    """)

# Graphs and Analysis
st.subheader("📈 Feature Distribution")
selected_feature = st.selectbox("Select a feature to visualize", features)

fig1, ax1 = plt.subplots(figsize=(8, 3))
sns.histplot(X[selected_feature], kde=True, color='skyblue', ax=ax1)
ax1.axvline(user_input[features.tolist().index(selected_feature)], color='red', linestyle='--', label="Your Input")
ax1.set_title(f"Distribution of {selected_feature}")
ax1.legend()
st.pyplot(fig1)

st.subheader("🌐 PCA Projection (2D)")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
input_array = np.array([user_input])
input_pca = pca.transform(input_array)

fig2, ax2 = plt.subplots()
ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', alpha=0.5, label="Dataset")
ax2.scatter(input_pca[:, 0], input_pca[:, 1], color='black', marker='X', s=200, label="Your Input")
ax2.set_xlabel("PCA 1")
ax2.set_ylabel("PCA 2")
ax2.set_title("2D PCA - Tumor Dataset")
ax2.legend()
st.pyplot(fig2)

# 📘 Tumor Type Info
st.markdown("### 📘 Understanding Tumor Types")
st.info("""
🔴 **Malignant** tumors are cancerous and can grow/spread quickly.
🟢 **Benign** tumors are non-cancerous and usually not life-threatening.
""")

# 📗 Feature Prefix Descriptions
st.markdown("### 📗 Feature Explanation (Prefix-based)")
desc_df = pd.DataFrame.from_dict(feature_descriptions, orient='index', columns=['Description']).reset_index()
desc_df.columns = ['Feature Prefix', 'Description']
st.dataframe(desc_df, use_container_width=True)

# Prediction
st.markdown("---")
st.subheader("🔍 Prediction Result")

input_scaled = scaler.transform([user_input])
prediction = model.predict(input_scaled)[0]
prob = model.predict_proba(input_scaled)[0]

if prediction == 1:
    st.success(f"🟢 The tumor is likely **Benign** with `{prob[1]*100:.2f}%` confidence.")
else:
    st.error(f"🔴 The tumor is likely **Malignant** with `{prob[0]*100:.2f}%` confidence.")

with st.expander("📊 Show Raw Prediction Probabilities"):
    st.write(f"Malignant: `{prob[0]:.4f}`")
    st.write(f"Benign: `{prob[1]:.4f}`")

# Footer
st.markdown("---")
st.caption("Developed by Raviranjan Singh • 2025")
