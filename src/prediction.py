from src.model import train_model
from src.prepare_data import preprocess_data
from src.profiling import create_profile


def generate_prediction(X_train, X_test, y_train, selected_features, param_grid):
    X_train_subset = preprocess_data(X_train, selected_features)
    X_test_subset = preprocess_data(X_test, selected_features)

    gs = train_model(X_train_subset, y_train, param_grid)

    create_profile(X_train_subset, y_train)

    return gs, gs.predict(X_test_subset)
