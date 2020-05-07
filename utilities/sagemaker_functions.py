import os
import xgboost
import sagemaker
import pandas as pd
from sagemaker import s3_input
from sagemaker.amazon.amazon_estimator import get_image_uri


def create_model(image: str, hyperparameters: dict, instance_type: str, output_path: str,
                 region_name: str, role: str, s3_train: str, s3_validation: str, job_name: str):
    if image == 'xgboost':
        container = get_image_uri(region_name, image, '0.90-2')
    else:
        container = get_image_uri(region_name, image)
    model = sagemaker.estimator.Estimator(container,
                                          role=role,
                                          train_instance_count=1,
                                          train_instance_type=instance_type,
                                          train_use_spot_instances=True,
                                          output_path=output_path)
    model.set_hyperparameters(**hyperparameters)
    data_channel = {'train': s3_input(s3_train, content_type='text/csv'),
                    'validation': s3_input(s3_validation, content_type='text/csv')}
    model.fit(data_channel, job_name=job_name)
    return model


def transform(model, data: str, output_path: str, instance_type: str):
    transformer = model.transformer(instance_count=1,
                                    instance_type=instance_type,
                                    max_payload=100,
                                    output_path=output_path,
                                    assemble_with='Line',
                                    accept='text/csv')
    transformer.transform(data, content_type='text/csv')
    transformer.wait()


def download_model(model_path: str, local_dir: str):
    return os.system(command=f'aws s3 cp {model_path} {local_dir}')


def decompress_model(local_dir: str):
    return os.system(command=f'tar xvf {local_dir}')


def prediction_df(model, file_path: str, score: float):
    names = [f'f{i}' for i in range(0, 12)]
    data = pd.read_csv(file_path, sep=',', names=names)
    data['prediction'] = model.predict(xgboost.DMatrix(data=data))
    data['prediction'] = data['prediction'].apply(lambda x: 1 if x >= score else 0)
    return data
