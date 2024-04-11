# Import necessary libraries and functions
from sklearn.model_selection import GridSearchCV
from prepare_data import create_pipeline


def train_model(X, y, param_grid):
    pipe = create_pipeline()
    gs = GridSearchCV(pipe, param_grid, cv=5)

    gs.fit(X, y.values.ravel())
    return gs
