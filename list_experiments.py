from mlflow.tracking import MlflowClient

client = MlflowClient()
experiment_ids = [exp.experiment_id for exp in client.search_experiments()]

for exp_id in experiment_ids:
    exp = client.get_experiment(exp_id)
    print(f"Name: {exp.name}, ID: {exp.experiment_id}")
