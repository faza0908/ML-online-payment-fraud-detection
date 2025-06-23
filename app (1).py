import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the saved model
model = joblib.load('random_forest_model.pkl')

# Assuming you have saved the scaler and label encoder
# You need to save these in the previous steps if you haven't already
# For demonstration purposes, I'll create dummy ones. In a real scenario,
# you would load the actual fitted objects.
# For the 'type' column, we need the classes the LabelEncoder was fitted on
type_classes = ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN'] # Replace with actual classes
le = LabelEncoder()
le.fit(type_classes) # Fit with the actual classes

# For the StandardScaler, we need the mean and variance it was fitted on
# You would load the actual fitted scaler here
# For demonstration, create a dummy scaler and fit it on sample data
# Replace this with loading your saved scaler
sample_data = pd.DataFrame({
    'type': type_classes,
    'amount': [0.0, 1000.0, 5000.0, 100.0, 2000.0],
    'oldbalanceOrg': [1000.0, 2000.0, 3000.0, 4000.0, 5000.0],
    'newbalanceOrig': [1000.0, 1000.0, 1000.0, 1000.0, 1000.0],
    'oldbalanceDest': [0.0, 0.0, 1000.0, 0.0, 500.0],
    'newbalanceDest': [0.0, 0.0, 1000.0, 0.0, 1500.0]
})
sample_data['type'] = le.transform(sample_data['type'])
scaler = StandardScaler()
scaler.fit(sample_data)


st.title('Online Payment Fraud Detection')

st.header('Enter Transaction Details')

# Input fields
transaction_type = st.selectbox('Transaction Type', type_classes)
amount = st.number_input('Amount', min_value=0.0, format='%f')
oldbalanceOrg = st.number_input('Old Balance Sender', min_value=0.0, format='%f')
newbalanceOrig = st.number_input('New Balance Sender', min_value=0.0, format='%f')
oldbalanceDest = st.number_input('Old Balance Recipient', min_value=0.0, format='%f')
newbalanceDest = st.number_input('New Balance Recipient', min_value=0.0, format='%f')

# Prediction button
if st.button('Predict'):
    # Create a DataFrame from input
    input_data = pd.DataFrame([[transaction_type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest]],
                              columns=['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest'])

    # Preprocess the input
    input_data['type'] = le.transform(input_data['type'])
    input_data[input_data.columns] = scaler.transform(input_data[input_data.columns])

    # Make prediction
    prediction = model.predict(input_data)

    # Display result
    if prediction[0] == 1:
        st.error('Fraudulent Transaction')
    else:
        st.success('Non-Fraudulent Transaction')
