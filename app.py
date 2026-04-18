import streamlit as st
import pandas as pd
import os

from modules.cleaning import clean_data
from modules.profiling import generate_profile
from modules.visualization import show_visuals
from modules.model import run_model, compare_models
from modules.utillis import data_quality_score


# -------------------------------
# 🎨 Page Config
# -------------------------------
st.set_page_config(
    page_title="AI Data Analytics",
    layout="wide",
    page_icon="🚀"
)

# -------------------------------
# 🌌 Custom CSS (Futuristic UI)
# -------------------------------
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: white;
    }
    h1, h2, h3 {
        color: #00f5d4;
    }
    .stButton>button {
        background-color: #00f5d4;
        color: black;
        border-radius: 10px;
        font-weight: bold;
    }
    .stDownloadButton>button {
        background-color: #ff006e;
        color: white;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# 🧭 Sidebar Navigation
# -------------------------------
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to", [
    "Home",
    "Upload & Clean",
    "Visualization",
    "Machine Learning"
])

# -------------------------------
# 🏠 HOME PAGE
# -------------------------------
if menu == "Home":
    st.title("🚀 AI-Powered Data Analytics System")

    st.markdown("""
    ### 🔥 Features:
    - Automated Data Cleaning  
    - Data Profiling  
    - Interactive Visualizations  
    - Machine Learning Models  
    - Model Comparison  

    ---
    ### 💡 How to Use:
    1. Upload your dataset  
    2. Clean the data  
    3. Explore insights  
    4. Train ML models  
    """)

# -------------------------------
# 📂 Upload & Clean
# -------------------------------
elif menu == "Upload & Clean":

    st.header("📂 Upload & Clean Data")

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        st.subheader("📌 Raw Data Preview")
        st.dataframe(df)

        if st.button("🧹 Clean Data"):
            cleaned_df = clean_data(df)

            st.session_state["cleaned_df"] = cleaned_df
            st.session_state["file_name"] = uploaded_file.name

            st.success("Data Cleaned Successfully!")

            # KPI Cards
            col1, col2, col3 = st.columns(3)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(f"""
                <div style="background-color:#1f2937;padding:20px;border-radius:10px;text-align:center">
                    <h4 style="color:white;">Rows</h4>
                    <h2 style="color:#00f5d4;">{cleaned_df.shape[0]}</h2>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div style="background-color:#1f2937;padding:20px;border-radius:10px;text-align:center">
                    <h4 style="color:white;">Columns</h4>
                    <h2 style="color:#fca311;">{cleaned_df.shape[1]}</h2>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                score = data_quality_score(cleaned_df)
                st.markdown(f"""
                <div style="background-color:#1f2937;padding:20px;border-radius:10px;text-align:center">
                    <h4 style="color:white;">Quality Score</h4>
                    <h2 style="color:#ff006e;">{score}%</h2>
                </div>
                """, unsafe_allow_html=True)
            st.subheader("📊 Cleaned Data")
            st.dataframe(cleaned_df)

            # Download
            file_name = os.path.splitext(uploaded_file.name)[0]
            cleaned_file_name = f"{file_name}_cleaned.csv"

            st.download_button(
                "📥 Download Cleaned Data",
                cleaned_df.to_csv(index=False),
                cleaned_file_name
            )

# -------------------------------
# 📊 Visualization Page
# -------------------------------
elif menu == "Visualization":

    st.header("📊 Data Visualization")

    if "cleaned_df" in st.session_state:
        show_visuals(st.session_state["cleaned_df"])
    else:
        st.warning("⚠️ Please clean data first.")

# -------------------------------
# 🤖 Machine Learning Page
# -------------------------------
elif menu == "Machine Learning":

    st.header("🤖 Machine Learning")

    if "cleaned_df" not in st.session_state:
        st.warning("⚠️ Please clean data first.")
    else:
        df = st.session_state["cleaned_df"]

        problem_type = st.selectbox(
            "Select Problem Type",
            ["regression", "classification", "clustering"]
        )

        if problem_type == "regression":
            models = ["Linear Regression", "Decision Tree", "Random Forest"]
        elif problem_type == "classification":
            models = ["Logistic Regression", "Decision Tree", "Random Forest"]
        else:
            models = ["KMeans"]

        model_name = st.selectbox("Select Model", models)

        target = None
        if problem_type != "clustering":
            target = st.selectbox("Select Target Column", df.columns)

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🚀 Run Model"):
                result = run_model(df, target, model_name, problem_type)
                st.success(f"Result: {result}")

        with col2:
            if problem_type != "clustering":
                if st.button("📊 Compare Models"):
                    results = compare_models(df, target, problem_type)
                    st.dataframe(pd.DataFrame(results.items(), columns=["Model", "Score"]))