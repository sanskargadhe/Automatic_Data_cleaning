import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def show_visuals(df):
    st.subheader("📊 Data Visualization Dashboard")

    numeric_cols = df.select_dtypes(include=['int64','float64']).columns
    cat_cols = df.select_dtypes(include=['object']).columns

    # -------------------------------
    # 📈 Correlation Heatmap
    # -------------------------------
    if len(numeric_cols) > 1:
        st.markdown("### 📈 Correlation Heatmap")
        plt.figure(figsize=(10,6))
        sns.heatmap(df.corr(numeric_only=True), annot=True)
        st.pyplot(plt)

    # -------------------------------
    # 📊 Histogram + KDE
    # -------------------------------
    if len(numeric_cols) > 0:
        st.markdown("### 📊 Distribution Plots")
        for col in numeric_cols:
            plt.figure()
            sns.histplot(df[col], kde=True)
            plt.title(f"Distribution of {col}")
            st.pyplot(plt)

    # -------------------------------
    # 📦 Box Plot (Outlier Detection)
    # -------------------------------
    if len(numeric_cols) > 0:
        st.markdown("### 📦 Box Plots (Outliers)")
        for col in numeric_cols:
            plt.figure()
            sns.boxplot(x=df[col])
            plt.title(f"Boxplot of {col}")
            st.pyplot(plt)

    # -------------------------------
    # 📊 Count Plot (Categorical)
    # -------------------------------
    if len(cat_cols) > 0:
        st.markdown("### 🏷️ Categorical Count Plots")
        for col in cat_cols:
            if df[col].nunique() < 20:
                plt.figure()
                sns.countplot(x=df[col])
                plt.xticks(rotation=45)
                plt.title(f"Count Plot of {col}")
                st.pyplot(plt)

    # -------------------------------
    # 🔗 Pair Plot (Relationships)
    # -------------------------------
    if len(numeric_cols) > 1:
        st.markdown("### 🔗 Pair Plot (Feature Relationships)")
        pairplot_fig = sns.pairplot(df[numeric_cols])
        st.pyplot(pairplot_fig)

    # -------------------------------
    # 📉 Line Plot (if time-based)
    # -------------------------------
    date_cols = df.select_dtypes(include=['datetime64']).columns
    if len(date_cols) > 0 and len(numeric_cols) > 0:
        st.markdown("### 📉 Time Series Plot")
        for date_col in date_cols:
            for num_col in numeric_cols:
                plt.figure()
                sns.lineplot(x=df[date_col], y=df[num_col])
                plt.xticks(rotation=45)
                plt.title(f"{num_col} over {date_col}")
                st.pyplot(plt)

    # -------------------------------
    # 🔥 Missing Values Heatmap
    # -------------------------------
    st.markdown("### 🔥 Missing Values Heatmap")
    plt.figure(figsize=(10,4))
    sns.heatmap(df.isnull(), cbar=False)
    st.pyplot(plt)