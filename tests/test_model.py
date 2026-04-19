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

        client = mlflow.MlflowClient()

        # ── Load challenger (new model to test) ───────────────────────────
        try:
            challenger = client.get_model_version_by_alias("my_model", "challenger")
            cls.new_model_version = challenger.version
        except Exception:
            raise RuntimeError("No model with alias 'challenger' found — cannot run tests")

        cls.new_model = mlflow.pyfunc.load_model(
            "models:/my_model@challenger"
        )
        print(f"\nChallenger model — version: {cls.new_model_version}")

        # ── Load champion (current production model) ──────────────────────
        try:
            champion = client.get_model_version_by_alias("my_model", "champion")
            cls.prod_model_version = champion.version
        except Exception:
            cls.prod_model_version = None

        if cls.prod_model_version and cls.prod_model_version != cls.new_model_version:
            cls.prod_model = mlflow.pyfunc.load_model(
                "models:/my_model@champion"
            )
            print(f"Champion model   — version: {cls.prod_model_version}")
        else:
            cls.prod_model = None
            print("No separate champion model found — floor check only")

        # ── Load vectorizer and holdout data ──────────────────────────────
        with open("models/vectorizer.pkl", "rb") as f:
            cls.vectorizer = pickle.load(f)

        cls.holdout_data = pd.read_csv("data/processed/test_tfidf.csv")

    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.new_model)

    def test_model_signature(self):
        input_text = "this movie was absolutely brilliant"
        input_data = self.vectorizer.transform([input_text])
        input_df   = pd.DataFrame(
            input_data.toarray(),
            columns=[str(i) for i in range(input_data.shape[1])]
        )
        prediction = self.new_model.predict(input_df)
        self.assertEqual(input_df.shape[1], len(self.vectorizer.get_feature_names_out()))
        self.assertEqual(len(prediction), input_df.shape[0])
        self.assertEqual(len(prediction.shape), 1)

    def test_model_performance(self):
        X_holdout = self.holdout_data.iloc[:, :-1]
        y_holdout = self.holdout_data.iloc[:, -1]

        y_pred_new    = self.new_model.predict(X_holdout)
        accuracy_new  = accuracy_score(y_holdout, y_pred_new)
        precision_new = precision_score(y_holdout, y_pred_new)
        recall_new    = recall_score(y_holdout, y_pred_new)
        f1_new        = f1_score(y_holdout, y_pred_new)

        print(
            f"\nChallenger — "
            f"accuracy: {accuracy_new:.4f} | precision: {precision_new:.4f} | "
            f"recall: {recall_new:.4f} | f1: {f1_new:.4f}"
        )

        # Layer 1 — absolute floor
        self.assertGreaterEqual(accuracy_new,  0.75, "Accuracy below floor")
        self.assertGreaterEqual(precision_new, 0.75, "Precision below floor")
        self.assertGreaterEqual(recall_new,    0.75, "Recall below floor")
        self.assertGreaterEqual(f1_new,        0.75, "F1 below floor")

        # Layer 2 — champion/challenger
        if self.prod_model is not None:
            y_pred_prod    = self.prod_model.predict(X_holdout)
            accuracy_prod  = accuracy_score(y_holdout, y_pred_prod)
            precision_prod = precision_score(y_holdout, y_pred_prod)
            recall_prod    = recall_score(y_holdout, y_pred_prod)
            f1_prod        = f1_score(y_holdout, y_pred_prod)

            print(
                f"Champion    — "
                f"accuracy: {accuracy_prod:.4f} | precision: {precision_prod:.4f} | "
                f"recall: {recall_prod:.4f} | f1: {f1_prod:.4f}"
            )

            self.assertGreaterEqual(accuracy_new,  accuracy_prod  - 0.01, "Accuracy regression vs champion")
            self.assertGreaterEqual(precision_new, precision_prod - 0.01, "Precision regression vs champion")
            self.assertGreaterEqual(recall_new,    recall_prod    - 0.01, "Recall regression vs champion")
            self.assertGreaterEqual(f1_new,        f1_prod        - 0.01, "F1 regression vs champion")
        else:
            print("No champion to compare against — floor check only")


if __name__ == "__main__":
    unittest.main()