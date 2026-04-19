import os
import mlflow

def promote_model():
    dagshub_token = os.getenv("CAPSTONE_TEST")
    if not dagshub_token:
        raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    mlflow.set_tracking_uri(
        "https://dagshub.com/Hello-Mitra/E2E-Text-Summarization.mlflow"
    )

    client     = mlflow.MlflowClient()
    model_name = "my_model"

    # Get the challenger version
    try:
        challenger = client.get_model_version_by_alias(model_name, "challenger")
    except Exception:
        print("No model with alias 'challenger' found — nothing to promote")
        return

    new_version = challenger.version

    # Reassign champion alias to new version
    client.set_registered_model_alias(
        name=model_name,
        alias="champion",
        version=new_version
    )
    print(f"Model version {new_version} promoted to 'champion' ✅")

if __name__ == "__main__":
    promote_model()