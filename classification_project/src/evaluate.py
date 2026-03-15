
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from src.config import MODEL_PATH, METRICS_PATH

def evaluate_model():
    model, vectorizer, X_test, y_test = joblib.load(MODEL_PATH)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    with open(METRICS_PATH, "w") as f:
        f.write(f"Accuracy: {accuracy}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm))
        f.write("\n\nClassification Report:\n")
        f.write(report)

    print(f"Model Accuracy: {accuracy}")
