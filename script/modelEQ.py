#%matplotlib inline

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

# for preprocessing the data
from sklearn.preprocessing import StandardScaler
# the model
from sklearn.ensemble import RandomForestClassifier
# for combining the preprocess with model training
from sklearn.pipeline import make_pipeline
# for optimizing the hyperparameters of the pipeline
from sklearn.model_selection import GridSearchCV



DATA_DIR = Path('.', 'dataset', 'raw')

# Load datasets

train_values = pd.read_csv(DATA_DIR / 'train_values.csv', index_col='building_id')
train_labels = pd.read_csv(DATA_DIR / 'train_labels.csv', index_col='building_id')

# Convert dictionary to DataFrame
df = pd.DataFrame(train_values)

selected_features = ['foundation_type', 
                     'area_percentage', 
                     'height_percentage',
                     'count_floors_pre_eq',
                     'land_surface_condition',
                     'has_superstructure_cement_mortar_stone']

train_values_subset = train_values[selected_features]

sns.pairplot(train_values_subset.join(train_labels), 
             hue='damage_grade')


#preprocess the data 
train_values_subset = pd.get_dummies(train_values_subset)

pipe = make_pipeline(StandardScaler(), 
                     RandomForestClassifier(random_state=2018))

param_grid = {'randomforestclassifier__n_estimators': [50, 100],
              'randomforestclassifier__min_samples_leaf': [1, 5]}
gs = GridSearchCV(pipe, param_grid, cv=5)

gs.fit(train_values_subset, train_labels.values.ravel())

gs.best_params_


from sklearn.metrics import f1_score

in_sample_preds = gs.predict(train_values_subset)
f1_score(train_labels, in_sample_preds, average='micro')


test_values = pd.read_csv(DATA_DIR / 'test_values.csv', index_col='building_id')

test_values_subset = test_values[selected_features]
test_values_subset = pd.get_dummies(test_values_subset)

predictions = gs.predict(test_values_subset)

submission_format = pd.read_csv(DATA_DIR / 'submission_format.csv', index_col='building_id')

my_submission = pd.DataFrame(data=predictions,
                             columns=submission_format.columns,
                             index=submission_format.index)

my_submission.head()






predictions = gs.predict(train_values_subset)

# Importing all necessary libraries
from sklearn.metrics import accuracy_score
# Calculating the accuracy of classifier
print(f"Accuracy of the classifier is: {accuracy_score(train_labels.values.ravel(), predictions)}")


# Importing all necessary libraries
from sklearn.metrics import precision_score

# Calculating the precision score of classifier
print(f"Precision Score of the classifier is: {precision_score(train_labels.values.ravel(), predictions, average='micro')}")

# Importing all necessary libraries
from sklearn.metrics import recall_score

# Calculating the recall score of classifier
print(f"Recall Score of the classifier is: {recall_score(train_labels.values.ravel(), predictions,average='micro')}")

# Importing all necessary libraries
from sklearn.metrics import f1_score

# Calculating the F1 score of classifier
print(f"F1 Score of the classifier is: {f1_score(train_labels.values.ravel(), predictions, average='micro')}")