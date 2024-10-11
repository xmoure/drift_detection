import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from evidently.report import Report
from evidently.metric_preset import ClassificationPreset
import os
import cv2
import numpy as np
import mlflow
from scipy.stats import wasserstein_distance
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Attachment, FileContent, FileName, FileType, Disposition
from dotenv import load_dotenv
from pathlib import Path
import yaml
import argparse
import base64
from mlflow.tracking import MlflowClient


load_dotenv()


categories = ["bb", "bk", "bn", "bp", "bq", "br", "wb", "wk", "wn", "wp", "wq", "wr", "empty"]

label_mapping = {category: 1 for category in categories if category != "empty"}
label_mapping["empty"] = 0

# Function to load images and their labels
def load_images_and_labels(base_path, categories):
    images = []
    labels = []
    image_paths = []

    for category in categories:
        category_path = os.path.join(base_path, category)
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (128, 128))
            images.append(img)
            labels.append(label_mapping[category])
            image_paths.append(img_path)


    return np.array(images), np.array(labels), np.array(image_paths)


def get_model_predictions(model, images, batch_size=32, threshold = 0.5):
    num_images = len(images)
    predictions = []

    # Process in batches
    for start in range(0, num_images, batch_size):
        end = min(start + batch_size, num_images)
        batch_images = images[start:end]
        batch_predictions = model.predict(batch_images)
        print("predictions", batch_predictions)

        # Convert probabilities to binary labels based on threshold
        batch_labels = (batch_predictions >= threshold).astype(int)

        # Flatten the array and add to the list of predictions
        predictions.extend(batch_labels.flatten())
        print(f"Processed {end}/{num_images} images ({(end / num_images) * 100:.2f}% complete)")

    return np.array(predictions)


def get_model_scores(model, images, batch_size=32):
    num_images = len(images)
    scores = []
    for start in range(0, num_images, batch_size):
        end = min(start + batch_size, num_images)
        batch_images = images[start:end]
        batch_scores = model.predict(batch_images)
        scores.extend(batch_scores.flatten())

        print(f"Processed {end}/{num_images} images ({(end / num_images) * 100:.2f}% complete)")

    return np.array(scores)


def get_metrics(ground_truth, predictions):
    accuracy = accuracy_score(ground_truth, predictions)
    precision = precision_score(ground_truth, predictions)
    recall = recall_score(ground_truth, predictions)
    f1 = f1_score(ground_truth, predictions)
    return accuracy, precision, recall, f1


# Function to calculate batch-wise error rates
def batch_error_rates(model, images, labels, batch_size=32):
    batch_errors = []
    for start in range(0, len(images), batch_size):
        end = min(start + batch_size, len(images))
        batch_images = images[start:end]
        batch_labels = labels[start:end]

        # Get predictions for the batch
        batch_preds = model.predict(batch_images)
        batch_preds = (batch_preds >= 0.5).astype(int).flatten()

        # Calculate error rate for this batch
        incorrect = np.sum(batch_preds != batch_labels)
        error_rate = incorrect / len(batch_labels)
        batch_errors.append(error_rate)

    return batch_errors



def send_email(new_split, drift_detected_with_ref_split, drift_detected_previous_split, accuracy_new, attachment_path=None):
    api_key = os.getenv("SENDGRID_API_KEY")
    to_email = os.getenv("TO_EMAIL")
    from_email = os.getenv("FROM_EMAIL")

    subject = "Concept Drift Detection Alert"

    if drift_detected_with_ref_split or  drift_detected_previous_split:
        subject = "Concept Drift Detected"
    else:
        subject= "No Concept Drift Detected"

    body_text = f"Hello, this is an automated notification regarding concept drift detection in your model. Drift detection was done between reference split and  {new_split}."

    drift_details_text = f"""
    Accuracy has dropped to {accuracy_new}
    """

    # Combine general body text with drift details for plain text and HTML
    body_text_combined = f"{body_text}\n\n{drift_details_text}"

    message = Mail(
        from_email=from_email,
        to_emails=to_email,
        subject=subject,
        plain_text_content=body_text_combined
    )

    # Add attachment if provided
    if attachment_path:
        with open(attachment_path, "rb") as f:
            file_data = f.read()
            encoded_file = base64.b64encode(file_data).decode()

        # Create the attachment
        attachment = Attachment(
            FileContent(encoded_file),
            FileName(os.path.basename(attachment_path)),
            FileType("application/octet-stream"),
            Disposition("attachment")
        )

        # Add the attachment to the email
        message.attachment = attachment

    try:
        sg = SendGridAPIClient(api_key)
        response = sg.send(message)
        print(f"Email sent successfully: {response.status_code}")
    except Exception as e:
        print(f"Failed to send email: {e}")



def detect_drift(split_name, split_path, reference_path, accuracy_drop_threshold, f1_drop_threshold, model, report_path, ml_flow_experiment ):
    client = MlflowClient()
    mlflow.set_experiment(ml_flow_experiment)
    # --- Start MLflow Run ---
    with mlflow.start_run() as run:
        mlflow.set_tag("split_name", split_name)
        # Check if reference data is already logged
        reference_run = mlflow.search_runs(order_by=["start_time DESC"], max_results=1, filter_string="tags.mlflow.runName = 'reference_run' AND status = 'FINISHED'")

        if reference_run.empty:
            # Process and log reference split for the first time
            print("Processing and logging reference split...")
            images_ref, labels_ref, img_paths_ref = load_images_and_labels(reference_path, categories)
            predictions_ref = get_model_predictions(model, images_ref)
            batch_errors_ref = batch_error_rates(model, images_ref, labels_ref)
            accuracy_ref, precision_ref, recall_ref, f1_ref = get_metrics(labels_ref, predictions_ref)

            # Log reference metrics and batch errors as an artifact
            mlflow.log_metrics({
                "accuracy_ref": accuracy_ref,
                "precision_ref": precision_ref,
                "recall_ref": recall_ref,
                "f1_ref": f1_ref
            })
            np.savetxt("reference_batch_errors.csv", batch_errors_ref, delimiter=",")
            np.savetxt("reference_labels.csv", labels_ref, delimiter=",")
            np.savetxt("reference_predictions.csv", predictions_ref, delimiter=",")

            mlflow.log_artifact("reference_labels.csv")
            mlflow.log_artifact("reference_predictions.csv")
            mlflow.log_artifact("reference_batch_errors.csv")
            mlflow.set_tag("mlflow.runName", "reference_run")
        else:
            mlflow.set_tag("mlflow.runName", f"{split_name}_run")
            # Load reference error rates from MLflow
            ref_run_id = reference_run.iloc[0]["run_id"]
            ref_errors_path = mlflow.artifacts.download_artifacts(run_id=ref_run_id, artifact_path="reference_batch_errors.csv")
            batch_errors_ref = np.loadtxt(ref_errors_path, delimiter=",")

            # Check if reference predictions and labels exist, and load them
            try:
                ref_predictions_path = mlflow.artifacts.download_artifacts(run_id=ref_run_id, artifact_path="reference_predictions.csv")
                predictions_ref = np.loadtxt(ref_predictions_path, delimiter=",")

                ref_labels_path = mlflow.artifacts.download_artifacts(run_id=ref_run_id, artifact_path="reference_labels.csv")
                labels_ref = np.loadtxt(ref_labels_path, delimiter=",")
            except OSError:
                print("Reference predictions or labels not found in MLflow artifacts. Skipping Evidently report generation.")
                predictions_ref, labels_ref = None, None  # Set to None if loading fails

            # Process the new split
            images_new, labels_new , img_paths_new= load_images_and_labels(split_path, categories)
            predictions_new = get_model_predictions(model, images_new)
            batch_errors_new = batch_error_rates(model, images_new, labels_new)
            accuracy_new, precision_new, recall_new, f1_new = get_metrics(labels_new, predictions_new)

            # Log new split metrics and error rates to MLflow
            mlflow.log_metrics({
                f"{split_name}_accuracy": accuracy_new,
                f"{split_name}_precision": precision_new,
                f"{split_name}_recall": recall_new,
                f"{split_name}_f1": f1_new
            })
            error_file = f"previous_batch_errors.csv"
            np.savetxt(error_file, batch_errors_new, delimiter=",")
            mlflow.log_artifact(error_file)

            # Check for Incremental Drift with Previous Errors
            current_run_id = mlflow.active_run().info.run_id

            previous_run = mlflow.search_runs(
                order_by=["start_time DESC"],
                max_results=1,
                filter_string=f"tags.mlflow.runName != 'reference_run' AND status = 'FINISHED' AND run_id != '{current_run_id}'"
            )
            if not previous_run.empty:
                previous_run_id = previous_run.iloc[0]["run_id"]
                try:
                    # Try downloading the previous batch errors artifact
                    previous_errors_path = mlflow.artifacts.download_artifacts(run_id=previous_run_id,  artifact_path="previous_batch_errors.csv")
                    batch_errors_previous = np.loadtxt(previous_errors_path, delimiter=",")
                    print("Current Batch Error Rates:", batch_errors_new)
                    print("Previous Batch Error Rates:", batch_errors_previous)

                    # Run drift checks
                    wd_prev = wasserstein_distance(batch_errors_previous, batch_errors_new)
                    # Detect drift based on threshold
                    drift_detected_prev = wd_prev > ws_threshold_prev

                    # Log drift detection results
                    mlflow.log_metric(f"{split_name}_wasserstein_distance_prev", wd_prev)
                    mlflow.log_metric(f"{split_name}_drift_detected_prev", int(drift_detected_prev))

                except OSError:
                    print("No previous batch errors file found in MLflow artifacts. Skipping comparison for this run.")
            else:
                print("No previous run found. Skipping comparison for this run.")

            # Save new batch errors in MLflow as an artifact for the next run
            np.savetxt("previous_batch_errors.csv", batch_errors_new, delimiter=",")
            mlflow.log_artifact("previous_batch_errors.csv")

            # --- Long-Term Drift Detection with Reference Split ---
            wd_ref = wasserstein_distance(batch_errors_ref, batch_errors_new)
            # Detect drift based on threshold
            drift_detected_ref = wd_ref > ws_threshold

            mlflow.log_metric(f"{split_name}_wasserstein_distance_ref", wd_ref)
            mlflow.log_metric(f"{split_name}_drift_detected_ref", int(drift_detected_ref))

            #--- Generate and Log Evidently Report ---
            # Generate and Log Evidently Report if Reference Predictions and Labels are Available
            if predictions_ref is not None and labels_ref is not None:
                reference_data = pd.DataFrame({'prediction': predictions_ref, 'target': labels_ref, 'dataset': 'reference'})
                new_data = pd.DataFrame({'prediction': predictions_new, 'target': labels_new, 'dataset': 'new'})
                classification_report = Report(metrics=[ClassificationPreset()])
                classification_report.run(reference_data=reference_data, current_data=new_data)

                # Save and log the Evidently report
                report_file = f"{split_name}_classification_report.html"
                path = f"{report_path}{report_file}"
                classification_report.save_html(path)
                mlflow.log_artifact(path)

                results = classification_report.as_dict()
                current_metrics = results["metrics"][0]["result"]["current"]
                reference_metrics = results["metrics"][0]["result"]["reference"]
                accuracy_current = current_metrics["accuracy"]
                f1_current = current_metrics["f1"]
                recall_current = current_metrics["recall"]
                accuracy_reference = reference_metrics["accuracy"]
                f1_reference = reference_metrics["f1"]

                accuracy_significant_drop = (accuracy_reference - accuracy_current) > (accuracy_reference * accuracy_drop_threshold)
                f1_significant_drop = (f1_reference - f1_current) > (f1_reference * f1_drop_threshold)
                print("Significant Accuracy Drop:", accuracy_significant_drop)
                print("Significant F1 Score Drop:", f1_significant_drop)

                mlflow.log_metric("accuracy_significant_drop", int(accuracy_significant_drop))
                mlflow.log_metric("f1_significant_drop", int(f1_significant_drop))
                mlflow.log_metric(f"{split_name}_accuracy", accuracy_current)
                mlflow.log_metric(f"{split_name}_f1", f1_current)
                mlflow.log_metric(f"{split_name}_recall", recall_current)

            else:
                print("Skipping Evidently report as reference predictions and labels are unavailable.")

            # Log long-term drift results
            mlflow.log_metric(f"{split_name}_wasserstein_distance_ref", wd_ref)

        print("Run completed and all data logged to MLflow!")

        send_email(split_name, drift_detected_ref, drift_detected_prev, accuracy_new, path)



if __name__ == "__main__":
    ws_threshold = 0.12
    ws_threshold_prev = 0.08

    accuracy_drop_threshold = 0.1 # for a 10% decrease
    f1_drop_threshold = 0.1 # for a 10% decrease

    params_path = Path("params.yaml")

    with open(params_path, "r", encoding="utf-8") as params_file:
        try:
            params = yaml.safe_load(params_file)
            params = params["extract_embeddings"]
        except yaml.YAMLError as exc:
            print(exc)

    parser = argparse.ArgumentParser(description="Get embeddings for a specific split path")
    parser.add_argument("--split_path", required=True, help="Path to the split for which to get embeddings")
    parser.add_argument("--split_id", required=True, help="MongoDB _id of the split document to update after processing")
    args = parser.parse_args()

    split_path = args.split_path
    split_id = args.split_id

    split_name = os.path.basename(split_path)

    model_location = params['PRODUCTION_MODEL_FOLDER']
    model_path = os.path.expanduser(model_location)

    model = load_model(model_path) if os.path.exists(model_path) else None
    if not model:
        print("Model not found.")
        exit(1)

    reference_dataset_location = params['REFERENCE_DATASET_PATH']
    reference_dataset_path = os.path.expanduser(reference_dataset_location)

    ml_flow_experiment = params['ML_FLOW_CONCEPT_DRIFT_EXPERIMENT']

    drift_report_path = os.path.expanduser(params['REPORT_PATH'])

    drift_report_name = f"concept_drift_report_referencesplit_vs_{split_name}.html"

    print("RUNNING CONCEPT DRIFT DETECTOR")

    detect_drift(split_name, split_path, reference_dataset_path, accuracy_drop_threshold, f1_drop_threshold, model, drift_report_path, ml_flow_experiment)


