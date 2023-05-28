import joblib
import mlflow
import argparse
from pprint import pprint
from train_model import read_params
from mlflow.tracking import MlflowClient


def log_production_model(config_path):
    config = read_params(config_path)
    mlflow_config = config["mlflow_config"]
    model_name = mlflow_config["registered_model_name"]
    model_dir = config["model_dir"]
    remote_server_uri = mlflow_config["remote_server_uri"]


    mlflow.set_tracking_uri(remote_server_uri)
    runs = mlflow.search_runs(experiment_ids=['1'])
    print('runs',runs["metrics.mse"])
    #max_accuracy = max(runs["metrics.accuracy"])
    #max_accuracy_run_id = list(runs[runs["metrics.accuracy"] == max_accuracy]["run_id"])[0]
    #max_r2 = max(runs["metrics.r2"])
    max_mse = runs["metrics.mse"].max() #max(runs["metrics.mse"])
    print('max_mse',max_mse)
    print('runs["metrics.mse"]',runs["metrics.mse"])
    print('max_mse_run_id',list(runs[runs["metrics.mse"] == max_mse]["run_id"])[0])
    #max_r2_run_id = list(runs[runs["metrics.r2"] == max_r2]["run_id"])[0]
    max_mse_run_id = list(runs[runs["metrics.mse"] == max_mse]["run_id"])[0]

   # mlflow.pyfunc.get_model_dependencies(model_uri)

    client = MlflowClient()
    logged_model=''
    for mv in client.search_model_versions(f"name='{model_name}'"):
        mv = dict(mv)
        print('mv',mv)
        print('logged_model',logged_model)
        if mv["run_id"] == max_mse_run_id:
            current_version = mv["version"]
            logged_model = mv["source"]
            pprint(mv, indent=4)
            client.transition_model_version_stage(
                name=model_name,
                version=current_version,
                stage="Production"
            )
        else:
            current_version = mv["version"]
            logged_model = mv["source"]
            pprint(mv, indent=4)
            client.transition_model_version_stage(
                name=model_name,
                version=current_version,
                stage="Staging"
            )
    last_logged_model = logged_model
    print('***logged_model****', logged_model)
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    joblib.dump(loaded_model, model_dir)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    data = log_production_model(config_path=parsed_args.config)