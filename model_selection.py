import mlflow
import mlflow.sklearn
import joblib
import os
from mlflow.tracking import MlflowClient

# Constants
EXPERIMENT_NAME = "Default"
EXPORT_DIR = "exported_models"
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"

# Configure MLflow tracking URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def fetch_best_run(experiment_name, metric_name="f1_score"):
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=[f"metrics.{metric_name} DESC"],
        max_results=1
    )

    if not runs:
        raise ValueError("No completed runs found in experiment.")

    return runs[0]

def download_and_load_model(run_id):
    model_uri = f"runs:/{run_id}/model"
    return mlflow.sklearn.load_model(model_uri)

def main():
    os.makedirs(EXPORT_DIR, exist_ok=True)

    print("🔍 Fetching best run based on F1 Score...")
    best_run = fetch_best_run(EXPERIMENT_NAME, metric_name="f1_score")

    print("\n✅ Best Run Found:")
    print(f"Run ID   : {best_run.info.run_id}")
    print(f"F1 Score : {best_run.data.metrics.get('f1_score', 'N/A')}")
    print(f"Accuracy : {best_run.data.metrics.get('accuracy', 'N/A')}")
    print(f"Params   : {best_run.data.params}")

    print("\n📦 Loading best model from MLflow...")
    best_model = download_and_load_model(best_run.info.run_id)

    export_path = os.path.join(EXPORT_DIR, f"best_model_{best_run.info.run_id}.pkl")
    joblib.dump(best_model, export_path)

    print(f"\n✅ Best model exported to: {export_path}")
    print("🏁 Model selection and export completed.")

if __name__ == "__main__":
    main()
