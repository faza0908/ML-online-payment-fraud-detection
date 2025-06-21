import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
import kagglehub
import os

# Load the saved LightGBM model
try:
    model = joblib.load('best_lgbm_model.pkl')
except FileNotFoundError:
    st.error("Model file not found. Please make sure 'best_lgbm_model.pkl' is in the same directory.")
    st.stop()

# Define the dataset URL (This is no longer used since we are downloading via KaggleHub)
# dataset_url = "YOUR_DATASET_URL" # Replace with the actual URL of your dataset


# Load the dataset from the downloaded path
@st.cache_data # Cache the data to avoid reloading on every interaction
def load_data(path):
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        st.error(f"Error loading dataset from path: {e}")
        st.stop()

# Download latest version of the dataset using kagglehub and load it
try:
    path = kagglehub.dataset_download("jainilcoder/online-payment-fraud-detection")
    print("Path to dataset files:", path)
    # Define the path to the CSV file within the downloaded dataset
    # You might need to adjust this path based on the dataset structure
    dataset_path = os.path.join(path, "online-payment-fraud-detection", "kaggle_data.csv") # **UPDATE THIS PATH**

    data = load_data(dataset_path)

except Exception as e:
    st.error(f"Error downloading dataset from KaggleHub or loading data: {e}")
    st.stop()


# You can now use the 'data' DataFrame in your Streamlit app
# For example, you might want to display some information about the data
# st.write("Dataset loaded successfully. First 5 rows:")
# st.write(data.head())


# Create a Streamlit application title
st.title("Online Payment Fraud Detection")

# Add a brief description
st.write("Enter the transaction details below to predict if it is a fraudulent transaction.")

# Create input fields for each feature
amount = st.number_input("Amount", value=0.0)
oldbalanceOrg = st.number_input("Old Balance Originator", value=0.0)
newbalanceOrig = st.number_input("New Balance Originator", value=0.0)
oldbalanceDest = st.number_input("Old Balance Destination", value=0.0)
newbalanceDest = st.number_input("New Balance Destination", value=0.0)

type_CASH_OUT = st.checkbox("Transaction Type: CASH_OUT")
type_DEBIT = st.checkbox("Transaction Type: DEBIT")
type_PAYMENT = st.checkbox("Transaction Type: PAYMENT")
type_TRANSFER = st.checkbox("Transaction Type: TRANSFER")

# Calculate transaction_change feature
transaction_change = oldbalanceOrg - newbalanceOrig

# Create a button to trigger the prediction
if st.button("Predict"):
    # Collect user input into a DataFrame
    input_data = pd.DataFrame({
        'amount': [amount],
        'oldbalanceOrg': [oldbalanceOrg],
        'newbalanceOrig': [newbalanceOrig],
        'oldbalanceDest': [oldbalanceDest],
        'newbalanceDest': [newbalanceDest],
        'type_CASH_OUT': [int(type_CASH_OUT)], # Convert boolean to int
        'type_DEBIT': [int(type_DEBIT)], # Convert boolean to int
        'type_PAYMENT': [int(type_PAYMENT)], # Convert boolean to int
        'type_TRANSFER': [int(type_TRANSFER)], # Convert boolean to int
        'transaction_change': [transaction_change]
    })

    # Recreate and fit a scaler on a small sample of representative data or load a saved one
    # For simplicity in this example, we will create a dummy scaler and fit it
    # on the input data. In a real application, you would load a pre-fitted scaler.
    # Let's fit a new scaler on the input data for demonstration purposes.
    # WARNING: This is not the correct way to use StandardScaler in a real application.
    # You should fit the scaler on your training data and save/load it.
    scaler = StandardScaler()
    numerical_cols = ["amount", "oldbalanceOrg", "newbalanceOrig", "transaction_change"]
    try:
        input_data[numerical_cols] = scaler.fit_transform(input_data[numerical_cols])
    except ValueError as e:
        st.error(f"Error during scaling: {e}. This might happen with constant input values.")
        st.stop()


    # Make prediction
    prediction = model.predict(input_data)

    # Display prediction result
    if prediction[0] == 1:
        st.error("Prediction: Fraudulent Transaction")
    else:
        st.success("Prediction: Non-Fraudulent Transaction")
