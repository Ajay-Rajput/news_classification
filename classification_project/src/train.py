# import joblib
# from sklearn.linear_model import LogisticRegression
# from src.feature_engineering import extract_features
# from src.config import MODEL_PATH, RANDOM_STATE

# def train_model():
#     X, y, vectorizer = extract_features()

#     model = LogisticRegression(max_iter=200, random_state=RANDOM_STATE)
#     model.fit(X, y)

#     joblib.dump((model, vectorizer), MODEL_PATH)

#     print("Model training completed and saved.")

import joblib
from sklearn.svm import LinearSVC
from src.feature_engineering import extract_features
from src.config import MODEL_PATH, RANDOM_STATE

def train_model():
    X_train, X_test, y_train, y_test, vectorizer = extract_features()

    model = LinearSVC(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    joblib.dump((model, vectorizer, X_test, y_test), MODEL_PATH)

    print("Model training completed and saved.")
