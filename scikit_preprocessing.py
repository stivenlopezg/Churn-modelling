from __future__ import print_function
import os
from io import StringIO

import argparse
import json
import numpy as np
import pandas as pd

from config import feature_columns_names, label_column, \
                   feature_columns_dtypes, label_column_dtype, to_boolean, \
                   numerical_features, categorical_features, cols_to_modeling
from custom_pipeline import ColumnSelector, ConvertDtypes, \
                            GetDummies, BooleanTransformation
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.externals import joblib

from sagemaker_containers.beta.framework import encoders, worker

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    args = parser.parse_args()

    # Read the data
    input_files = [os.path.join(args.train, file) for file in os.listdir(args.train)]
    if len(input_files) == 0:
        raise ValueError(f'There are no files in {args.train}. \n'
                         f'This usually indicates that the channel ({"train"}) was incorrectly specified, \n'
                         f'the data specification in S3 was incorrectly specified or the role specified\n'
                         f'does not have permission to acces the data.')
    raw_data = [pd.read_csv(file, names=feature_columns_names + [label_column],
                            sep=';', dtype=feature_columns_dtypes.update(label_column_dtype)) for file in input_files]
    data = pd.concat(raw_data)
    # Build Pipeline
    general_transformations = Pipeline([('boolean', BooleanTransformation(columns=to_boolean)),
                                        ('dtypes', ConvertDtypes(numerical=numerical_features,
                                                                 categorical=categorical_features)),
                                        ('selector', ColumnSelector(columns=cols_to_modeling))])

    numerical_selector = Pipeline([('numerical_selector', ColumnSelector(columns=numerical_features))])

    categorical_transformations = Pipeline([('categorical_selector', ColumnSelector(columns=categorical_features)),
                                            ('ohe', GetDummies(columns=categorical_features))])

    preprocessor = Pipeline([('general', general_transformations),
                             ('features', FeatureUnion([
                                 ('numerical', numerical_selector),
                                 ('categorical', categorical_transformations)
                             ]))])
    preprocessor.fit(data)
    joblib.dump(preprocessor, filename=os.path.join(args.model_dir, 'model.joblib'))
    print("The model has been saved!")


def model_fn(model_dir):
    preprocessor_job = joblib.load(os.path.join(model_dir, 'model.joblib'))
    return preprocessor_job


def input_fn(input_data, content_type):
    if content_type == 'text/csv':
        df = pd.read_csv(StringIO(input_data), sep=';', header=None)
        if len(df.columns) == len(feature_columns_names) + 1:
            df.columns = feature_columns_names + [label_column]
        elif len(df.columns) == len(feature_columns_names):
            df.columns = feature_columns_names
        return df
    else:
        raise ValueError(f'{content_type} not supported by script')


def predict_fn(input_data, model):
    features = model.transform(input_data)
    if label_column in input_data:
        return np.insert(features, 0, input_data[label_column], axis=1)
    else:
        return features


def output_fn(prediction, accept):
    if accept == 'application/json':
        instances = []
        for row in prediction.tolist():
            instances.append({'features': row})

        json_output = {'instances': instances}
        return worker.Response(json.dumps(json_output), accept=accept, mimetype=accept)
    elif accept == 'text/csv':
        return worker.Response(encoders.encode(prediction, accept), accept=accept, mimetype=accept)
    else:
        raise RuntimeError(f'{accept} accept type is not supported by this script.')
