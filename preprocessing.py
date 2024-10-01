import pandas as pd
import boto3

def load_and_preprocess_data():
    # Load the dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    data = pd.read_csv(url, header=None, names=column_names)
    
    # Convert species to numeric
    data['species'] = data['species'].astype('category').cat.codes
    
    # Save to CSV for SageMaker
    train_data = pd.concat([data.iloc[:, :-1], data.iloc[:, -1]], axis=1)
    train_data.to_csv('iris_train.csv', index=False, header=False)

    # Upload to S3
    s3 = boto3.client('s3')
    bucket_name = 'ablonebucket'  # Your actual S3 bucket name
    s3.upload_file(Filename='iris_train.csv', Bucket=bucket_name, Key='iris/iris_train.csv')  # Corrected line

if __name__ == "__main__":
    load_and_preprocess_data()
