import pandas as pd
import numpy as np
import joblib

# Load scaler
scaler = joblib.load('models/scaler.pkl')

def preprocess_input(input_data):
    """
    Preprocess user input for model prediction.

    Args:
        input_data (dict): Dictionary containing client features.

    Returns:
        np.array: Preprocessed input ready for model prediction.
    """
    # Convert input dictionary to DataFrame
    client_df = pd.DataFrame([input_data])
    
    # Add derived feature: income_employ_ratio
    client_df['income_employ_ratio'] = client_df['income'] / (client_df['employ'] + 1)
    
    # Feature list based on training data
    feature_columns = ['tenure', 'age', 'address', 'income', 'employ', 'income_employ_ratio']
    
    # Scale numerical features
    client_df[feature_columns] = scaler.transform(client_df[feature_columns])
    
    return client_df[feature_columns].values
