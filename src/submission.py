import pandas as pd
from src.prepare_data import DATA_DIR


def generate_submission(predictions):
    submission_format = pd.read_csv(
        DATA_DIR / 'submission_format.csv', index_col='building_id')
    required_length = len(submission_format)

    # Adjust predictions to match required length
    if len(predictions) > required_length:
        predictions = predictions[:required_length]  # Truncate if too long
    elif len(predictions) < required_length:
        # Extend with default_value if too short
        predictions += [default_value] * (required_length - len(predictions))

    result = pd.DataFrame(
        data=predictions, columns=submission_format.columns, index=submission_format.index)

    result.to_csv('submission.csv', index=True)
