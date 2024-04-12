import sys

from src.features import get_features
from src.evaluation import evaluate_model
from src.submission import generate_submission

param_grid = {
    'randomforestclassifier__n_estimators': [50, 100],
    'randomforestclassifier__min_samples_leaf': [1, 5]
}

if __name__ == "__main__":
    categories = sys.argv[1:]
    selected_features = get_features(*categories)
    predictions, score = evaluate_model(selected_features, param_grid)
    print('F1 score is : ', score)
    generate_submission(predictions)
