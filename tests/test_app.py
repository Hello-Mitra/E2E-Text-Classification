import os
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

os.environ.setdefault("CAPSTONE_TEST", "dummy-token-for-testing")


class TestFastAPIApp(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Patch external dependencies before importing app.
        Model loading now happens in lifespan so we patch
        the lifespan from being called during TestClient init.
        """
        # Patch dagshub and mlflow setup
        patch("dagshub.init").start()
        patch("mlflow.set_tracking_uri").start()

        # Patch get_latest_model_version used in lifespan
        patch(
            "backend.main.get_latest_model_version",
            return_value="4"
        ).start()

        # Patch mlflow.pyfunc.load_model used in lifespan
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([1])
        patch("mlflow.pyfunc.load_model", return_value=mock_model).start()

        # Patch pickle.load for vectorizer
        mock_vectorizer = MagicMock()
        mock_array = np.zeros((1, 100))
        mock_transform_result = MagicMock()
        mock_transform_result.toarray.return_value = mock_array
        mock_transform_result.shape = (1, 100)
        mock_vectorizer.transform.return_value = mock_transform_result
        patch("pickle.load", return_value=mock_vectorizer).start()

        # Patch builtins.open so pickle.load doesn't need real file
        patch("builtins.open", unittest.mock.mock_open()).start()

        # Now safe to import
        from fastapi.testclient import TestClient
        from backend.main import app
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        patch.stopall()

    def test_health_returns_200(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)

    def test_health_returns_ok(self):
        self.assertEqual(
            self.client.get("/health").json()["status"], "ok"
        )

    def test_predict_returns_200(self):
        resp = self.client.post(
            "/predict", json={"text": "This movie was great!"}
        )
        self.assertEqual(resp.status_code, 200)

    def test_predict_returns_sentiment(self):
        resp = self.client.post(
            "/predict", json={"text": "This movie was great!"}
        )
        self.assertIn(resp.json()["sentiment"], ["Positive", "Negative"])

    def test_predict_returns_confidence(self):
        resp = self.client.post(
            "/predict", json={"text": "This movie was great!"}
        )
        self.assertIsInstance(resp.json()["confidence"], float)

    def test_predict_missing_text_returns_422(self):
        self.assertEqual(
            self.client.post("/predict", json={}).status_code, 422
        )

    def test_metrics_returns_200(self):
        self.assertEqual(
            self.client.get("/metrics").status_code, 200
        )


if __name__ == "__main__":
    unittest.main()