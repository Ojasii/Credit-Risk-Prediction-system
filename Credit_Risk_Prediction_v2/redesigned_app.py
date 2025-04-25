import streamlit as st
import pandas as pd
from data_preprocessing import load_and_preprocess_data
from feature_engineering import engineer_features
from model_interpretation import plot_feature_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# Inject custom CSS
st.markdown(
    '''
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .block-container {
        padding-top: 2rem;
    }
    h1 {
        color: #1f77b4;
        font-weight: 800;
    }
    </style>
    ''' ,
    unsafe_allow_html=True
)

# Sidebar
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/190/190411.png", width=80)
st.sidebar.title("\U0001F50D Credit Risk UI")
st.sidebar.markdown("Upload your dataset and get insights instantly!")

# Main Title
st.title("💳 Credit Risk Predictor")
st.markdown("Use machine learning to classify applicants as **Good** or **Bad Credit Risk**.")

# Upload file
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    try:
        with st.spinner("🔄 Processing data..."):
            raw_df = pd.read_csv(uploaded_file)
            df = load_and_preprocess_data(raw_df)
            df = engineer_features(df)

        st.success("✅ Data processed successfully!")

        st.markdown("📂 Preview Cleaned Data")
        st.write(df.head())

        # Splitting data
        X = df.drop(columns=['Risk'])
        y = df['Risk']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Models to compare
        models = {
            "Random Forest": RandomForestClassifier(random_state=42),
            "SVM": SVC(probability=True, random_state=42),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
        }

        results = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            results[name] = {
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred),
                "Recall": recall_score(y_test, y_pred),
                "F1 Score": f1_score(y_test, y_pred),
                "Model": model
            }

        # Display model comparison
        st.markdown("### 🧠 Model Comparison Results")
        for model_name, metrics in results.items():
            st.markdown(f"#### 📌 {model_name}")
            st.write({k: round(v, 4) for k, v in metrics.items() if k != "Model"})

        # Determine best model by F1 Score, break ties by preferred order
        preferred_order = ["Random Forest", "XGBoost", "SVM", "Logistic Regression"]
        best_model_name = sorted(
            results.items(),
            key=lambda x: (-x[1]["F1 Score"], preferred_order.index(x[0]))
        )[0][0]

        st.markdown(f"### ✅ Best Performing Model: **{best_model_name}**")
        st.markdown("#### 📊 Classification Report")
        best_model = results[best_model_name]["Model"]
        best_pred = best_model.predict(X_test)
        best_report = classification_report(y_test, best_pred, output_dict=True)
        st.json(best_report)

        # Feature Importance if RF is best
        if best_model_name == "Random Forest":
            st.markdown("### 🔍 Feature Importance")
            plot_feature_importance(best_model, X.columns)
            st.image("feature_importance.png")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
else:
    st.info("📁 Upload a CSV file to begin.")

# Footer
st.markdown("---")
st.markdown("Made by Ojasi Lavanya | Powered by Streamlit")
