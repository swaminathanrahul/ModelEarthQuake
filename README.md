# Building Earth Quake Damage Prediction Model

## Overview

This project develops a predictive model aimed at assessing building damage levels. Utilizing a RandomForestClassifier, the model is part of a pipeline that incorporates extensive data preprocessing and hyperparameter tuning through GridSearchCV. The goal is to accurately predict the extent of damage a building may sustain based on various structural and material characteristics collected pre-disaster. This predictive capability is vital for improving preparedness and response strategies in disaster-prone areas.

## Repository Structure

- `src/`: Directory containing Python scripts for various stages of the machine learning pipeline, including data handling, feature extraction, model training, and evaluation.
- `data/raw/`: Intended location for storing raw datasets needed for model training and testing.
- `dist/`: Destination folder for outputs such as prediction files, formatted as CSV for easy submission or further analysis.
- `README.md`: The document you are currently reading that provides an overview and instructions for the project.

## Setup

### Dependencies

The model requires the following Python packages:

- `pandas`: For data manipulation and ingestion.
- `scikit-learn`: For creating and utilizing the machine learning pipeline and model.

You can install these dependencies via pip with the following command:

```bash
pip install pandas scikit-learn
```

Data
Ensure that your training and testing data are placed within the data/raw/ directory under the following filenames:

train_values.csv - Feature set for training.
train_labels.csv - Corresponding labels for the training dataset.
test_values.csv - Feature set for model evaluation or real-world testing.
submission_format.csv - A template outlining the format required for submitting predictions.

Running the Code
To initiate the model training and prediction pipeline, navigate to the project directory and run the following command in your terminal:

```bash
python main.py [feature_category_names]
```

you can find the complete structure here for selection. It can be extended as needed.

```
feature_set = {
    'Basic_Structural': [
        'geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id',
        'count_floors_pre_eq', 'age', 'area_percentage', 'height_percentage'
    ],
    'Construction_Materials': [
        'foundation_type', 'roof_type', 'ground_floor_type', 'other_floor_type'
    ],
    'Superstructure_Materials': [
        'has_superstructure_adobe_mud', 'has_superstructure_mud_mortar_stone',
        'has_superstructure_stone_flag', 'has_superstructure_cement_mortar_stone',
        'has_superstructure_mud_mortar_brick', 'has_superstructure_cement_mortar_brick',
        'has_superstructure_timber', 'has_superstructure_bamboo',
        'has_superstructure_rc_non_engineered', 'has_superstructure_rc_engineered',
        'has_superstructure_other'
    ],
    'Usage_and_Legal': [
        'legal_ownership_status', 'count_families', 'has_secondary_use',
        'has_secondary_use_agriculture', 'has_secondary_use_hotel', 'has_secondary_use_rental',
        'has_secondary_use_institution', 'has_secondary_use_school', 'has_secondary_use_industry',
        'has_secondary_use_health_post', 'has_secondary_use_gov_office', 'has_secondary_use_use_police',
        'has_secondary_use_other'
    ]
}
```

Features

The features utilized in this model are organized into sets based on their characteristics, such as structural attributes or materials used in construction. These sets are configurable and can be specified when executing the main script. This modular feature selection allows for flexible experimentation with different combinations of predictors to optimize model performance.

Models and Training

The evaluate_model script encapsulates the model training process. Here, the RandomForestClassifier is embedded within a pipeline that includes preprocessing steps such as scaling and encoding, ensuring the input features are appropriately formatted for model training. The use of GridSearchCV for hyperparameter tuning allows the selection of the best model parameters based on cross-validation performance, optimizing the classifier for accuracy.

Prediction and Submission File Generation

After training, the model uses unseen test data to make predictions. These predictions are then formatted according to a specified submission structure and output to the dist/ directory. The naming convention for output files includes the feature categories used, making it easier to track experiments across different model configurations.
