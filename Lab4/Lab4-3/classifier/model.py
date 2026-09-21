# T1: Cross-validation and automatic saving of the best model
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
import joblib
import numpy as np

def train_and_save_best_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=10000),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "RandomForest": RandomForestClassifier(random_state=42)
    }

    best_score = 0
    best_name = ""
    best_model = None

    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        mean_score = np.mean(scores)
        print(f"{name} CV Accuracy: {mean_score:.4f}")

        if mean_score > best_score:
            best_score = mean_score
            best_name = name
            best_model = model

    print(f"--> Best model selected: {best_name} (CV Score: {best_score:.4f})")

    # Train the winning model on the full training set
    best_model.fit(X_train, y_train)

    # Save the model and the test data (for metrics.py later)
    joblib.dump(best_model, "best_classifier.joblib")
    joblib.dump((X_test, y_test), "test_data.joblib")

    return best_model
