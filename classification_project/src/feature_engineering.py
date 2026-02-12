
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from src.config import PROCESSED_DATA_PATH, TEST_SIZE, RANDOM_STATE, MAX_FEATURES

def extract_features():
    df = pd.read_csv(PROCESSED_DATA_PATH)

    X = df["text"]
    y = df["label"]

    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)

    X_vectorized = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vectorized, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    return X_train, X_test, y_train, y_test, vectorizer




