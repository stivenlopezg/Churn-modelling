import numpy as np

input_filename = 'Churn_Modelling.csv'
target = 'Exited'

# AWS

region_name = 'us-east-1'
bucket = 'banking-data'
key = 'data'

# Features

feature_columns_names = ['RowNumber', 'CustomerId', 'Surname', 'CreditScore',
                         'Geography', 'Gender', 'Age', 'Tenure', 'Balance',
                         'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']

label_column = 'Exited'

numerical_features = ['CreditScore', 'Age', 'Tenure', 'Balance',
                      'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
categorical_features = ['Geography', 'Gender']

cols_to_modeling = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure',
                    'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']

# Features - Dtypes

feature_columns_dtypes = {
    'RowNumber': np.int64,
    'CustomerId': np.int64,
    'Surname': 'category',
    'CreditScore': np.int64,
    'Geography': 'category',
    'Gender': 'category',
    'Age': np.int64,
    'Tenure': np.int64,
    'Balance': np.float64,
    'NumOfProducts': np.int64,
    'HasCrCard': np.int64,
    'IsActiveMember': np.int64,
    'EstimatedSalary': np.float64
}

label_column_dtype = {
    'Exited': np.int64
}

# Categories for categorical_features

gender_categories = ['Female', 'Male']
geography_categories = ['France', 'Spain', 'Germany']
card_and_member_categories = ['True', 'False']
