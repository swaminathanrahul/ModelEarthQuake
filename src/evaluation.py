import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from prepare_data import DATA_DIR
from prepare_data import load_data
from prediction import generate_prediction


def evaluate_model(selected_features, param_grid):
    train_values, train_labels, test_values = load_data()

    sub_train_values, sub_test_values, sub_train_labels, sub_test_labels = train_test_split(
        train_values, train_labels, test_size=0.5, random_state=42)

    test_preds = generate_prediction(
        sub_train_values, sub_test_values, sub_train_labels, selected_features, param_grid)

    test_f1_score = f1_score(sub_test_labels, test_preds, average='micro')

    print("Test F1 Score:", test_f1_score)
