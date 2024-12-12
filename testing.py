import unittest
from main import detect_drift, train_on_subject, central_training_loop
from model_trainer import ModelTrainer
from data_handler import DataHandler
import pandas as pd

# Constants
LABEL_COLUMN = 'label'
FEATURE_COLUMNS = ['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']
RANDOM_STATE = 42
TRAIN_SAMPLE_SIZE = 500
TEST_SAMPLE_SIZE = 2000

THRESHOLDS = {
    'accuracy': 0.9,
    'f1_score': 0.85
}

DRIFT_THRESHOLD = {
    "accuracy": 0.05,  # Maximum allowed drop in accuracy
    "f1_score": 0.05,  # Maximum allowed drop in F1-score
    "precision": 0.05,  # Maximum allowed drop in precision
    "recall": 0.05,  # Maximum allowed drop in recall
}

# Unit testing
class TestDriftDetection(unittest.TestCase):
    def test_detect_drift(self):
        previous_metrics = {"accuracy": 0.9, "f1_score": 0.85}
        new_metrics_no_drift = {"accuracy": 0.91, "f1_score": 0.86}
        new_metrics_with_drift = {"accuracy": 0.8, "f1_score": 0.7}

        self.assertFalse(detect_drift(previous_metrics, new_metrics_no_drift))
        self.assertTrue(detect_drift(previous_metrics, new_metrics_with_drift))


class TestTrainOnSubject(unittest.TestCase):
    def test_train_on_subject(self):
        data_handler = DataHandler(LABEL_COLUMN, FEATURE_COLUMNS, RANDOM_STATE)
        mock_data = pd.DataFrame({"feature1": [0.1, 0.2], "label": [1, 0]})
        cumulative_data = pd.DataFrame()
        trainer = ModelTrainer(LABEL_COLUMN, FEATURE_COLUMNS, RANDOM_STATE)

        model, updated_data = train_on_subject(data_handler, cumulative_data, trainer, data_handler.train_subjects[0])

        self.assertGreater(len(updated_data), len(cumulative_data))
        self.assertIsNotNone(model)

# Integration testing
class TestTrainingPipeline(unittest.TestCase):
    def test_central_training_loop(self):
        data_handler = DataHandler(LABEL_COLUMN, FEATURE_COLUMNS, RANDOM_STATE)
        thresholds = {"accuracy": 0.9, "f1_score": 0.85}

        # Run the training loop
        central_training_loop(data_handler, thresholds, max_iterations=5)

        # Assertions
        self.assertGreater(len(data_handler.train_used_indices), 0)
        self.assertTrue(data_handler.logs["metrics_logged"])


if __name__ == "__main__":
    unittest.main()