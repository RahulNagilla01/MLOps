# T3: Predict and print probabilities
import sys
import joblib
import numpy as np
from classifier import is_valid_input
import warnings
warnings.filterwarnings('ignore') # Ignore feature names warning for single predictions

# Take a comma-separated list of 30 features from the command line
input_str = sys.argv[1]
features = [float(x) for x in input_str.split(",")]

if not is_valid_input(features):
    print("Invalid input: Measurements are outside realistic biological ranges.")
else:
    model = joblib.load("best_classifier.joblib")
    X_new = np.array(features).reshape(1, -1)

    prediction = model.predict(X_new)[0]
    probabilities = model.predict_proba(X_new)[0]

    # 0 = Malignant, 1 = Benign in sklearn's breast cancer dataset
    class_name = "Malignant" if prediction == 0 else "Benign"
    confidence = probabilities[prediction] * 100

    print(f"Prediction: {class_name}")
    print(f"Confidence: {confidence:.2f}%")
