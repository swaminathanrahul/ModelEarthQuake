import pandas as pd
from src.prepare_data import DATA_DIR


def generate_submission(predictions, *categories):
    submission_format = pd.read_csv(
        DATA_DIR / 'submission_format.csv', index_col='building_id')

    required_length = len(submission_format)

    if len(predictions) > required_length:
        predictions = predictions[:required_length]  # Truncate if too long

    categories_str = '_'.join(categories).replace(
        ' ', '_')  # Replace spaces with underscores if any

    filename = f'dist/submission-{categories_str}.csv'

    result = pd.DataFrame(
        data=predictions, columns=submission_format.columns, index=submission_format.index)

    result.to_csv(filename, index=True)
