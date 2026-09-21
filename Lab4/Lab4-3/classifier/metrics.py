# T4: Evaluate saved model without retraining
import joblib
from sklearn.metrics import confusion_matrix, classification_report

def evaluate_saved_model():
    try:
        model = joblib.load("best_classifier.joblib")
        X_test, y_test = joblib.load("test_data.joblib")
    except FileNotFoundError:
        print("Model or test data not found. Run train.py first.")
        return

    preds = model.predict(X_test)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))
    print("\nClassification Report:")
    print(classification_report(y_test, preds))
