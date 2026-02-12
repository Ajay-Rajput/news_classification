import pandas as pd
import re
from .config import RAW_DATA_PATH, PROCESSED_DATA_PATH


def load_data(path=RAW_DATA_PATH):
    return pd.read_csv(path)


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_and_save(in_path=RAW_DATA_PATH, out_path=PROCESSED_DATA_PATH):
    df = load_data(in_path)
    # Combine Title and Description into text field
    df['text'] = (df['Title'] + ' ' + df['Description']).fillna("").apply(clean_text)
    # Keep only label and text columns
    df = df[['Class Index', 'text']].rename(columns={'Class Index': 'label'})
    df.to_csv(out_path, index=False)
    return df


def preprocess_data():
    preprocess_and_save()
    print("Data preprocessing completed.")
