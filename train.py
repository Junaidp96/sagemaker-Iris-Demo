import pandas as pd
import sagemaker
from sagemaker import get_execution_role
from sagemaker.estimator import Estimator
from sagemaker import image_uris
import boto3
from train import train_model, upload_csv_to_s3
def upload_csv_to_s3(bucket_name):
    # Load the dataset
    df = pd.read_csv('iris_train.csv', header=None)
    print("CSV loaded successfully.")

    # Save to a temporary location if needed
    csv_path = '/tmp/iris_train.csv'
    df.to_csv(csv_path, index=False, header=False)

    # Upload to S3
    s3 = boto3.client('s3')
    s3.upload_file(csv_path, 'iris/iris_train.csv', Bucket=bucket_name)
    print(f"Uploaded to S3: s3://ablonebucket/iris/iris_train.csv")

def train_model(bucket_name):
    sagemaker_session = sagemaker.Session()
    role = get_execution_role()

    image_uri = image_uris.retrieve('linear-learner', 'us-east-2')

    # Create the estimator
    estimator = Estimator(
        image_uri=image_uri,
        role=role,
        instance_count=1,
        instance_type='ml.m5.large',
        output_path=f's3://ablonebucket/output',
        sagemaker_session=sagemaker_session,
    )

    # Set hyperparameters
    estimator.set_hyperparameters(feature_dim=4, predictor_type='multiclass_classifier', num_classes=3)

    # Define input data (using the CSV directly)
    train_data_input = f's3://ablonebucket/iris/iris_train.csv'
    estimator.fit({'train': train_data_input})

    return estimator

if __name__ == "__main__":
    bucket_name = 'ablonebucket'  # Your actual S3 bucket name
    upload_csv_to_s3(bucket_name)  # Upload CSV data
    train_model(bucket_name)  # Train the model




