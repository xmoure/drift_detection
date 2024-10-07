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
from bson.objectid import ObjectId
from pymongo import MongoClient
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from dotenv import load_dotenv

load_dotenv()

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


def load_and_downsample_image_paths(data_folder):

    occupied = glob(os.path.join(data_folder, "[!empty]*/*.png"))
    empty = glob(os.path.join(data_folder, "empty/*.png"))

    print(f"Occupied samples: {len(occupied)}")
    print(f"Empty samples: {len(empty)}")

    # Downsample the empty class to match the number of occupied samples
    if len(empty) > len(occupied):
        rng = np.random.default_rng(seed=42)
        rng.shuffle(empty)
        empty = empty[:len(occupied)]

    # Combine paths
    paths = np.array(occupied + empty)

    return paths

def send_email(new_split, mmd_drift_detected, ratio_drift_detected, model_drift_detected):
    api_key = os.getenv("SENDGRID_API_KEY")
    to_email = os.getenv("TO_EMAIL")
    from_email = os.getenv("FROM_EMAIL")

    subject = "Drift Detection Alert"

    if mmd_drift_detected or ratio_drift_detected or model_drift_detected:
        subject = "Drift Detected"
    else:
        subject= "No Drift Detected"

    body_text = f"Hello, this is an automated notification regarding drift detection in your model. Drift detection was done between reference split and  {new_split}."

    drift_details_text = f"""
    Drift Detection Results:
    - MMD Drift: {'Detected' if mmd_drift_detected else 'Not Detected'}
    - Ratio Drift: {'Detected' if ratio_drift_detected else 'Not Detected'}
    - Model Drift: {'Detected' if model_drift_detected else 'Not Detected'}
    """

    drift_details_html = f"""
    <h2>Drift Detection Results:</h2>
    <ul>
        <li><strong>MMD Drift:</strong> {'Detected' if mmd_drift_detected else 'Not Detected'}</li>
        <li><strong>Ratio Drift:</strong> {'Detected' if ratio_drift_detected else 'Not Detected'}</li>
        <li><strong>Model Drift:</strong> {'Detected' if model_drift_detected else 'Not Detected'}</li>
    </ul>
    """

    # Combine general body text with drift details for plain text and HTML
    body_text_combined = f"{body_text}\n\n{drift_details_text}"
    body_html_combined = f"""
    <html>
    <body>
        <p>{body_text}</p>
        {drift_details_html}
    </body>
    </html>
    """

    message = Mail(
        from_email=from_email,
        to_emails=to_email,
        subject=subject,
        plain_text_content=body_text_combined,
        html_content=body_html_combined
    )

    try:
        sg = SendGridAPIClient(api_key)
        response = sg.send(message)
        print(f"Email sent successfully: {response.status_code}")
    except Exception as e:
        print(f"Failed to send email: {e}")


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

    send_email(current_split_name, mmd_result, ratio_result, model_result)

def get_mongo_splits_connection():
    # MongoDB connection details
    mongo_connection_string = os.getenv("MONGO_DB_CONNECTION_STRING")
    mongo_db= os.getenv("MONGO_DB")
    print(f"MONGO_DB: {mongo_db} (type: {type(mongo_db)})")
    mongo_collection_splits= os.getenv("MONGO_COLLECTION_SPLITS")
    mongo_client = MongoClient(mongo_connection_string)
    db = mongo_client[mongo_db]
    splits_collection = db[mongo_collection_splits]
    return splits_collection


def update_split_status(split_id):
    
    splits_collection = get_mongo_splits_connection()

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
    
    model_location = params['MODEL_FOLDER']

    STORE_EMBEDDINGS_PATH = params['STORE_EMBEDDINGS_PATH']

    model_path = os.path.expanduser(model_location)

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

    split_name = os.path.basename(split_path)

    print("### Loading the data and generating embeddings")
    new_image_paths = load_and_downsample_image_paths(split_path)
    image_dataset = create_dataset(new_image_paths)
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

    print("RUNNING DRIFT DETECTOR")
    
    detect_drift(embeddings_train, new_embeddings, "REFERENCE SPLIT", split_name, drift_report_name, drift_report_path, ml_flow_experiment)

    print("UPDATING SPLIT COLLECTION")
    update_split_status(split_id)


