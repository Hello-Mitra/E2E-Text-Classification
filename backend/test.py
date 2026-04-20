import os
import mlflow
from dotenv import load_dotenv

load_dotenv()

dagshub_token = os.getenv("CAPSTONE_TEST")
os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

mlflow.set_tracking_uri(
    "https://dagshub.com/Hello-Mitra/E2E-Text-Summarization.mlflow"
)

client = mlflow.MlflowClient()

# Check champion alias
try:
    v = client.get_model_version_by_alias("my_model", "champion")
    print(f"champion alias → version {v.version}")
except Exception as e:
    print(f"champion alias not found: {e}")

# Check challenger alias
try:
    v = client.get_model_version_by_alias("my_model", "challenger")
    print(f"challenger alias → version {v.version}")
except Exception as e:
    print(f"challenger alias not found: {e}")

# Check all versions
versions = client.search_model_versions("name='my_model'")
for v in versions:
    print(f"version {v.version} | aliases: {v.aliases} | stage: {v.current_stage}")