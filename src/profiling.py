import pandas as pd
from pandas_profiling import ProfileReport


def create_profile(train_labels, train_values):
    # Convert dictionary to DataFrame
    df = pd.concat(train_labels, train_values, axis=1)

    # Generate the profile report
    profile = ProfileReport(
        df, title="Pandas Profiling Report", explorative=True)

    # Save the report to a file
    profile.to_file("dist/data_profile.html")
