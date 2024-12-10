import streamlit as st
from utils.preprocess import preprocess_input
from utils.predict import predict_group

# Streamlit app
st.title("Customer Group Prediction")

# Collect user input
st.sidebar.header("Client Features")
input_data = {
    'tenure': st.sidebar.number_input("Tenure (months)", min_value=0, max_value=60, value=24),
    'age': st.sidebar.number_input("Age (years)", min_value=18, max_value=100, value=35),
    'address': st.sidebar.number_input("Address Duration (years)", min_value=0, max_value=50, value=10),
    'income': st.sidebar.number_input("Income ($)", min_value=10000, max_value=1000000, value=75000),
    'employ': st.sidebar.number_input("Employment Duration (years)", min_value=0, max_value=50, value=15)
}

# Predict button
if st.button("Predict Group"):
    # Preprocess input
    preprocessed_input = preprocess_input(input_data)
    
    # Predict group
    prediction_result = predict_group(preprocessed_input)
    
    # Display results
    st.subheader("Prediction Results")
    st.write(f"**Predicted Group**: {prediction_result['predicted_group']}")
    st.write("**Probabilities**:")
    for group, prob in prediction_result['probabilities'].items():
        st.write(f"{group}: {prob:.2%}")
    st.write(f"**Confidence**: {prediction_result['confidence']:.2%}")
