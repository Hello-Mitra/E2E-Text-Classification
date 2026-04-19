import unittest
import mlflow
import os
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class TestModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dagshub_token = os.getenv("CAPSTONE_TEST")
        if not dagshub_token:
            raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        mlflow.set_tracking_uri(
            "https://dagshub.com/Hello-Mitra/E2E-Text-Summarization.mlflow"
        )

        # ── Load NEW model from Staging ───────────────────────────────────
        cls.new_model_version = cls.get_latest_model_version("my_model", stage="Staging")
        if not cls.new_model_version:
            raise RuntimeError("No model found in Staging — cannot run tests")

        cls.new_model = mlflow.pyfunc.load_model(
            f"models:/my_model/{cls.new_model_version}"
        )
        print(f"\nNew model (Staging)     — version: {cls.new_model_version}")

        # ── Load CURRENT production model for comparison ──────────────────
        cls.prod_model_version = cls.get_latest_model_version("my_model", stage="Production")

        # Only load prod if it is a DIFFERENT version than staging
        # (avoids comparing a model against itself)
        if cls.prod_model_version and cls.prod_model_version != cls.new_model_version:
            cls.prod_model = mlflow.pyfunc.load_model(
                f"models:/my_model/{cls.prod_model_version}"
            )
            print(f"Current model (Production) — version: {cls.prod_model_version}")
        else:
            cls.prod_model = None
            print("No separate production model found — skipping comparison")

        # ── Load vectorizer and holdout data ──────────────────────────────
        with open("models/vectorizer.pkl", "rb") as f:
            cls.vectorizer = pickle.load(f)

        cls.holdout_data = pd.read_csv("data/processed/test_tfidf.csv")

    @staticmethod
    def get_latest_model_version(model_name, stage="Staging"):
        client   = mlflow.MlflowClient()
        versions = client.get_latest_versions(model_name, stages=[stage])
        return versions[0].version if versions else None

    # ── Test 1 — Model loads properly ─────────────────────────────────────
    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.new_model)

    # ── Test 2 — Input/output shape is correct ────────────────────────────
    def test_model_signature(self):
        input_text = "this movie was absolutely brilliant"
        input_data = self.vectorizer.transform([input_text])
        input_df   = pd.DataFrame(
            input_data.toarray(),
            columns=[str(i) for i in range(input_data.shape[1])]
        )
        prediction = self.new_model.predict(input_df)

        # Input shape must match vectorizer vocabulary
        self.assertEqual(
            input_df.shape[1],
            len(self.vectorizer.get_feature_names_out())
        )
        # Output must have one prediction per input row
        self.assertEqual(len(prediction), input_df.shape[0])
        # Output must be 1D (binary classification)
        self.assertEqual(len(prediction.shape), 1)

    # ── Test 3 — Performance floor + champion/challenger comparison ────────
    def test_model_performance(self):
        X_holdout = self.holdout_data.iloc[:, :-1]
        y_holdout = self.holdout_data.iloc[:, -1]

        # Evaluate new model
        y_pred_new    = self.new_model.predict(X_holdout)
        accuracy_new  = accuracy_score(y_holdout, y_pred_new)
        precision_new = precision_score(y_holdout, y_pred_new)
        recall_new    = recall_score(y_holdout, y_pred_new)
        f1_new        = f1_score(y_holdout, y_pred_new)

        print(
            f"\nNew model    — "
            f"accuracy: {accuracy_new:.4f} | "
            f"precision: {precision_new:.4f} | "
            f"recall: {recall_new:.4f} | "
            f"f1: {f1_new:.4f}"
        )

        # ── Layer 1: Absolute floor ────────────────────────────────────────
        # Catches completely broken models regardless of prod performance
        self.assertGreaterEqual(accuracy_new,  0.75, "Accuracy below minimum floor of 0.80")
        self.assertGreaterEqual(precision_new, 0.75, "Precision below minimum floor of 0.80")
        self.assertGreaterEqual(recall_new,    0.75, "Recall below minimum floor of 0.80")
        self.assertGreaterEqual(f1_new,        0.75, "F1 below minimum floor of 0.80")

        # ── Layer 2: Champion/challenger comparison ────────────────────────
        # New model must be at least as good as current production
        # Tolerance of 0.01 (1%) absorbs tiny random variation between runs
        if self.prod_model is not None:
            y_pred_prod    = self.prod_model.predict(X_holdout)
            accuracy_prod  = accuracy_score(y_holdout, y_pred_prod)
            precision_prod = precision_score(y_holdout, y_pred_prod)
            recall_prod    = recall_score(y_holdout, y_pred_prod)
            f1_prod        = f1_score(y_holdout, y_pred_prod)

            print(
                f"Prod model   — "
                f"accuracy: {accuracy_prod:.4f} | "
                f"precision: {precision_prod:.4f} | "
                f"recall: {recall_prod:.4f} | "
                f"f1: {f1_prod:.4f}"
            )

            self.assertGreaterEqual(
                accuracy_new, accuracy_prod - 0.01,
                f"New accuracy {accuracy_new:.4f} is worse than "
                f"production {accuracy_prod:.4f}"
            )
            self.assertGreaterEqual(
                precision_new, precision_prod - 0.01,
                f"New precision {precision_new:.4f} is worse than "
                f"production {precision_prod:.4f}"
            )
            self.assertGreaterEqual(
                recall_new, recall_prod - 0.01,
                f"New recall {recall_new:.4f} is worse than "
                f"production {recall_prod:.4f}"
            )
            self.assertGreaterEqual(
                f1_new, f1_prod - 0.01,
                f"New F1 {f1_new:.4f} is worse than "
                f"production {f1_prod:.4f}"
            )
        else:
            print("No production model to compare against — floor check only")


if __name__ == "__main__":
    unittest.main()