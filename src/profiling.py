import pandas as pd
from pandas_profiling import ProfileReport

# Load datasets
train_values = pd.read_csv('dataset/raw/Train_Values.csv')

# Convert dictionary to DataFrame
df = pd.DataFrame(train_values)

# Generate the profile report
profile = ProfileReport(df, title="Pandas Profiling Report", explorative=True)

# Save the report to a file
profile.to_file("assets/data_profile.html")
