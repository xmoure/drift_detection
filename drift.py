import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
from glob import glob
import pickle
from pathlib import Path
import yaml
import argparse
from evidently.metrics import EmbeddingsDriftMetric, Comment
from evidently.report import Report
from evidently.metrics.data_drift.embedding_drift_methods import model, ratio, mmd
import pickle
import pandas as pd
from evidently import ColumnMapping
from sklearn.preprocessing import StandardScaler, MinMaxScaler, normalize
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pymongo
from bson.objectid import ObjectId
from pymongo import MongoClient

def load_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, [128, 128])
    return img / 255.0

def create_dataset(image_paths, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

# Function to preprocess a list of image paths
def preprocess_image_paths(image_paths):
    images = [load_image(path) for path in image_paths]
    return images


def detect_drift(embeddings_train, embeddings_test, reference_split_name, current_split_name, file_name, full_path, ml_flow_experiment):

    client = MlflowClient()
    mlflow.set_experiment(ml_flow_experiment)

    # Z-Score Normalization
    scaler_z = StandardScaler()
    embeddings_train_z = scaler_z.fit_transform(embeddings_train)
    embeddings_test_z = scaler_z.transform(embeddings_test)

    # Min-Max Normalization
    scaler_minmax = MinMaxScaler()
    embeddings_train_minmax = scaler_minmax.fit_transform(embeddings_train)
    embeddings_test_minmax = scaler_minmax.transform(embeddings_test)

    # L2 Normalization
    embeddings_train_l2 = normalize(embeddings_train, norm='l2')
    embeddings_test_l2 = normalize(embeddings_test, norm='l2')

    # Convert to DataFrames
    embeddings_train_z_df = pd.DataFrame(embeddings_train_z, columns=[f"z_{i}" for i in range(embeddings_train_z.shape[1])])
    embeddings_test_z_df = pd.DataFrame(embeddings_test_z, columns=[f"z_{i}" for i in range(embeddings_test_z.shape[1])])

    embeddings_train_minmax_df = pd.DataFrame(embeddings_train_minmax, columns=[f"minmax_{i}" for i in range(embeddings_train_minmax.shape[1])])
    embeddings_test_minmax_df = pd.DataFrame(embeddings_test_minmax, columns=[f"minmax_{i}" for i in range(embeddings_test_minmax.shape[1])])

    embeddings_train_l2_df = pd.DataFrame(embeddings_train_l2, columns=[f"l2_{i}" for i in range(embeddings_train_l2.shape[1])])
    embeddings_test_l2_df = pd.DataFrame(embeddings_test_l2, columns=[f"l2_{i}" for i in range(embeddings_test_l2.shape[1])])

    # Combine all the normalized embeddings into a single DataFrame
    embeddings_train_combined_df = pd.concat([embeddings_train_z_df, embeddings_train_minmax_df, embeddings_train_l2_df], axis=1)
    embeddings_test_combined_df = pd.concat([embeddings_test_z_df, embeddings_test_minmax_df, embeddings_test_l2_df], axis=1)


    # Define the column mapping based on the different normalization methods
    column_mapping = ColumnMapping(
        embeddings={
            'embeddings_z': [f"z_{i}" for i in range(embeddings_train_z_df.shape[1])],
            'embeddings_minmax': [f"minmax_{i}" for i in range(embeddings_train_minmax_df.shape[1])],
            'embeddings_l2': [f"l2_{i}" for i in range(embeddings_train_l2_df.shape[1])]
        }
    )

    embeddings_drift_metric_mmd = EmbeddingsDriftMetric(
        'embeddings_z',
        drift_method = mmd(
                                threshold = 0.015,
                                bootstrap = None,
                                quantile_probability = 0.95,
                                pca_components = None,
                            )
    )

    embeddings_drift_metric_ratio = EmbeddingsDriftMetric(
        'embeddings_minmax',
        drift_method = ratio(
                                component_stattest = 'wasserstein',
                                component_stattest_threshold = 0.1,
                                threshold = 0.2,
                                pca_components = None,
                            )
    ) 

    embeddings_drift_metric_model = EmbeddingsDriftMetric(
        'embeddings_l2',
        drift_method = model(
                                threshold = 0.55,
                                bootstrap = None,
                                quantile_probability = 0.95,
                                pca_components = None,
                            )
    ) 

    model_description = f""" #  Drift detection between {reference_split_name} and {current_split_name}. Normalized"""

    report = Report(metrics=[
        Comment(model_description),
        embeddings_drift_metric_mmd,
        embeddings_drift_metric_ratio,
        embeddings_drift_metric_model
        #embeddings_drift_metric_distance
    ])

    # Run the report to calculate the drift
    report.run(reference_data=embeddings_train_combined_df, current_data=embeddings_test_combined_df, column_mapping=column_mapping)

    # Save the report as HTML
    path = f"{full_path}{file_name}"
    report.save_html(path)

    report_json = report.as_dict()

    mmd_result = report_json['metrics'][1]['result']['drift_detected']
    ratio_result = report_json['metrics'][2]['result']['drift_detected']
    model_result = report_json['metrics'][3]['result']['drift_detected']
    
    run_name_mlf= f"Drift detection {reference_split_name} vs {current_split_name}"

    # Start an MLflow run
    with mlflow.start_run(run_name=run_name_mlf):

        mlflow.log_param("reference_split_name", reference_split_name)
        mlflow.log_param("current_split_name", current_split_name)

        # Log whether drift was detected (True/False)
        mlflow.log_param("mmd_drift_detected", mmd_result)
        mlflow.log_param("ratio_drift_detected", ratio_result)
        mlflow.log_param("model_drift_detected", model_result)

        # Log the drift scores if available
        mlflow.log_metric("mmd_drift_score", report_json['metrics'][1]['result']['drift_score'])
        mlflow.log_metric("ratio_drift_score", report_json['metrics'][2]['result']['drift_score'])
        mlflow.log_metric("model_drift_score", report_json['metrics'][3]['result']['drift_score'])

        # Log the HTML report as an artifact
        artifact_path = f"{full_path}{file_name}"
        mlflow.log_artifact(artifact_path)

    print("Drift detection results logged in MLflow.")


def update_split_status(split_id):
    # Initialize MongoDB connection
    mongo_connection_string = os.getenv("MONGO_DB_CONNECTION_STRING")
    mongo_db = os.getenv("MONGO_DB")
    mongo_collection_splits = os.getenv("MONGO_COLLECTION_SPLITS")
    
    client = MongoClient(mongo_connection_string)
    db = client[mongo_db]
    splits_collection = db[mongo_collection_splits]

    result = splits_collection.update_one(
        {"_id": ObjectId(split_id)},
        {"$set": {"used_for_drift_detection": True}}
    )

    if result.modified_count > 0:
        print(f"Document with ID {split_id} updated successfully.")
    else:
        print(f"Document with ID {split_id} could not be updated.")


if __name__ == "__main__":

    params_path = Path("params.yaml")

    with open(params_path, "r", encoding="utf-8") as params_file:
        try:
            params = yaml.safe_load(params_file)
            params = params["extract_embeddings"]
        except yaml.YAMLError as exc:
            print(exc)

    print("params", params)
    
    model_location = params['MODEL_FOLDER']

    STORE_EMBEDDINGS_PATH = params['STORE_EMBEDDINGS_PATH']

    model_path = os.path.expanduser(model_location)

    print("model_path", model_path)

    encoder = load_model(model_path) if os.path.exists(model_path) else None
    if not encoder:
        print("Model not found.")
        exit(1)
    
    parser = argparse.ArgumentParser(description="Get embeddings for a specific split path")
    parser.add_argument("--split_path", required=True, help="Path to the split for which to get embeddings")
    parser.add_argument("--split_id", required=True, help="MongoDB _id of the split document to update after processing")
    args = parser.parse_args()

    split_path = args.split_path
    split_id = args.split_id

    print("split_paty", split_path)
    print("split_id", split_id)

    split_name = os.path.basename(split_path)
    print(split_name)

    print("### Loading the data and generating embeddings")
    image_dataset = create_dataset(split_path)
    new_embeddings = encoder.predict(image_dataset)

    embeddings_path = f"{STORE_EMBEDDINGS_PATH}/{split_name}_embeddings.pkl"
    embeddings_path  = os.path.expanduser(embeddings_path)

    embeddings_dir = os.path.dirname(embeddings_path)
    os.makedirs(embeddings_dir, exist_ok=True)
    # Save embeddings
    try:
        with open(embeddings_path, 'wb') as f:
            pickle.dump(new_embeddings, f)
        print(f"Embeddings saved successfully to {embeddings_path}")
    except Exception as e:
        print("Error saving embeddings:", e)
        exit(1)

    reference_emb_path = os.path.expanduser(params['REFERENCE_EMBEDDINGS'])
    drift_report_path = os.path.expanduser(params['REPORT_PATH'])
    ml_flow_experiment = params['ML_FLOW_EXPERIMENT']

    # LOAD REFERENCE EMBEDDINGS
    with open(reference_emb_path, "rb") as f:
        embeddings_train = pickle.load(f)

    drift_report_name = f"drift_report_referenceSplit_vs_{split_name}.html"
    
    detect_drift(embeddings_train, new_embeddings, "REFERENCE SPLIT", split_name, drift_report_name, drift_report_path, ml_flow_experiment)

    update_split_status(split_id)


