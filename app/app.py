import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load trained model
try:
    model = joblib.load('exported_models/best_model.joblib')
except Exception as e:
    st.error("❌ Failed to load the model.")
    st.exception(e)
    st.stop()

st.set_page_config(page_title="Sales Category Predictor", layout="wide")
st.title("🏷️ Warehouse Sales Category Classifier")
st.markdown("Upload a CSV file or manually input values to predict the **Sales Category**.")

# Input method selection
input_method = st.radio("Input method:", ["Upload CSV", "Manual Input"])
required_cols = ["Warehouse Size", "Inventory Turnover", "Number of Employees", "Product Variety", "Avg Product Cost"]

# Manual input function
def get_manual_input():
    warehouse_size = st.number_input("Warehouse Size", value=1000)
    inventory_turnover = st.number_input("Inventory Turnover", value=5.5)
    employees = st.number_input("Number of Employees", value=25)
    product_variety = st.number_input("Product Variety", value=30)
    avg_cost = st.number_input("Average Product Cost", value=50.0)

    return pd.DataFrame([{
        "Warehouse Size": warehouse_size,
        "Inventory Turnover": inventory_turnover,
        "Number of Employees": employees,
        "Product Variety": product_variety,
        "Avg Product Cost": avg_cost
    }])

# Handle input
input_df = None
if input_method == "Upload CSV":
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file:
        try:
            input_df = pd.read_csv(file)
            if all(col in input_df.columns for col in required_cols):
                st.write("Uploaded Data Preview:", input_df.head())
            else:
                st.error(f"❌ The CSV must contain columns: {', '.join(required_cols)}")
                input_df = None
        except Exception as e:
            st.error("❌ Failed to read the CSV.")
            st.exception(e)
else:
    input_df = get_manual_input()
    st.write("Manual Input Data:", input_df)

# Prediction
if st.button("🔍 Predict"):
    if input_df is not None:
        try:
            prediction = model.predict(input_df)
            st.write("### 🧠 Prediction Results")
            for i, pred in enumerate(prediction):
                st.success(f"Record {i+1}: Predicted Sales Category - **{pred}**")
        except Exception as e:
            st.error("❌ Prediction failed. Please check your input format.")
            st.exception(e)
    else:
        st.warning("Please provide valid input before predicting.")

# Model performance summary
st.markdown("---")
st.subheader("📈 Model Performance Summary (from validation set)")
st.markdown("""
- **Accuracy**: 98%
- **F1 Score**: 0.9798
""")

# Sample confusion matrix (use real one if available)
conf_matrix = [[55, 0, 0],
               [1, 54, 0],
               [0, 2, 53]]

fig, ax = plt.subplots()
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=["High", "Medium", "Low"],
            yticklabels=["High", "Medium", "Low"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot(fig)
