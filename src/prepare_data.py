import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier

# Set the data directory path
DATA_DIR = Path('.', 'data', 'raw')


def load_data():
    # Load training values and labels, and test values from CSV files using building_id as index
    train_values = pd.read_csv(
        DATA_DIR / 'train_values.csv', index_col='building_id')
    train_labels = pd.read_csv(
        DATA_DIR / 'train_labels.csv', index_col='building_id')
    test_values = pd.read_csv(
        DATA_DIR / 'test_values.csv', index_col='building_id')
    return train_values, train_labels, test_values


def preprocess_data(values, selected_features):
    # Filter the data for selected features and apply one-hot encoding
    values_subset = values[selected_features]
    values_subset = pd.get_dummies(values_subset)
    return values_subset


def create_pipeline():
    # Create a machine learning pipeline with StandardScaler and RandomForestClassifier
    pipe = make_pipeline(
        StandardScaler(), RandomForestClassifier(n_jobs=10, class_weight='balanced', random_state=2018))
    return pipe
