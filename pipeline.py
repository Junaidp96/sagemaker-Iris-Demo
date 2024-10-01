import boto3
from sagemaker import get_execution_role
from sagemaker.predictor import Predictor
from train import train_model
from eval import evaluate_model
from train import convert_to_recordio, train_model
from train import train_model, upload_csv_to_s3


if __name__ == "__main__":
    bucket_name = 'ablonebucket'  # Your actual S3 bucket name
    convert_to_recordio(bucket_name)  # Convert and upload data
    train_model(bucket_name)  # Train the model


def run_pipeline():
    bucket_name = 'ablonebucket'  # Change this to your S3 bucket name
    # Preprocessing
    import preprocessing
    preprocessing.load_and_preprocess_data()


    
    # Train Model
    estimator = train_model(bucket_name)

    # Deploy Model
    predictor = estimator.deploy(instance_type='ml.m5.large')

    # Evaluate Model
    evaluate_model(predictor)

    # Clean Up
    predictor.delete_endpoint()

if __name__ == "__main__":
    run_pipeline()
