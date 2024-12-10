import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model('models/trained_model.h5')

def predict_group(preprocessed_input):
    """
    Predict customer group based on preprocessed input.

    Args:
        preprocessed_input (np.array): Preprocessed input features.

    Returns:
        dict: Predicted group, probabilities, and confidence.
    """
    # Predict probabilities
    prediction_probs = model.predict(preprocessed_input)[0]
    predicted_class = np.argmax(prediction_probs)
    
    # Adjust back to original labels if necessary
    predicted_group = predicted_class + 1  # Assuming labels were shifted during training
    
    return {
        'predicted_group': predicted_group,
        'probabilities': {f'Group {i + 1}': prob for i, prob in enumerate(prediction_probs)},
        'confidence': float(prediction_probs[predicted_class])
    }
