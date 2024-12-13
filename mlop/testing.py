import unittest
from main import detect_drift, train_on_subject, central_training_loop
from model_trainer import ModelTrainer
from data_handler import DataHandler
import pandas as pd
import mlflow

# Constants
LABEL_COLUMN = 'label'
FEATURE_COLUMNS = ['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']
RANDOM_STATE = 42
TRAIN_SAMPLE_SIZE = 500
TEST_SAMPLE_SIZE = 2000

THRESHOLDS = {
    'accuracy': 0.8,
    'f1_score': 0.8
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
        new_metrics_on_threshold = {"accuracy": 0.861, "f1_score": 0.821}
        new_metrics_drift_one_metric = {"accuracy": 0.849, "f1_score": 0.85}
        new_metrics_identical = previous_metrics

        self.assertFalse(detect_drift(previous_metrics, new_metrics_no_drift))
        self.assertFalse(detect_drift(previous_metrics, new_metrics_identical))
        self.assertFalse(detect_drift(previous_metrics, new_metrics_on_threshold))
        self.assertTrue(detect_drift(previous_metrics, new_metrics_with_drift))
        self.assertTrue(detect_drift(previous_metrics, new_metrics_drift_one_metric))


class TestTrainOnSubject(unittest.TestCase):
    def test_train_on_subject(self):
        data_handler = DataHandler(LABEL_COLUMN, FEATURE_COLUMNS, RANDOM_STATE)
        trainer = ModelTrainer(LABEL_COLUMN, FEATURE_COLUMNS, RANDOM_STATE)
        mock_data = pd.DataFrame({"feature1": [0.1, 0.2], "label": [1, 0]})
        cumulative_data = pd.DataFrame()

        mlflow.set_experiment("Test Experiment")
        # Test to ensure new samples are collected properly
        model, updated_data = train_on_subject(data_handler, cumulative_data, trainer, data_handler.train_subjects[0])
        self.assertGreater(len(updated_data), len(cumulative_data))
        # and that a model is generated
        self.assertIsNotNone(model)


class TestDataHandler(unittest.TestCase):
    def test_create_stratified_sample(self):
        data_handler = DataHandler(LABEL_COLUMN, FEATURE_COLUMNS, RANDOM_STATE)
        mock_data = pd.DataFrame({
            'label': [1, 1, 2, 2, 3, 3],
            'feature1': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            'subject': ['S1', 'S1', 'S1', 'S1', 'S2', 'S2']
        })
        sample = data_handler.create_stratified_sample(mock_data, data_handler.test_used_indices, sample_size=4, subject='S1')

        # Check if correct sample size is created
        self.assertEqual(len(sample), 4)
        # and that it has been taken from the specified subject
        self.assertTrue(sample['subject'].eq('S1').all())
        # that the used indices were logged
        self.assertSetEqual(set(sample.index), data_handler.test_used_indices)

        # Testing behaviour when no subject specified
        used_indices = set()
        sample = data_handler.create_stratified_sample(mock_data, used_indices, sample_size=3, subject=None)
        self.assertEqual(len(sample), 3)

        # Ensure used indices are excluded
        data_handler.test_used_indices = set([0, 1])
        sample = data_handler.create_stratified_sample(mock_data, data_handler.test_used_indices, sample_size=3, subject=None)
        # Ensure none of the used indices are in the new sample
        self.assertTrue(set(sample.index).isdisjoint(data_handler.test_used_indices))
        # Ensure used indices are updated to include the new sample
        self.assertSetEqual(data_handler.test_used_indices, {0, 1}.union(set(sample.index)))
        


# Integration testing
class TestTrainingPipeline(unittest.TestCase):
    def test_central_training_loop(self):
        data_handler = DataHandler(LABEL_COLUMN, FEATURE_COLUMNS, RANDOM_STATE)
        thresholds = {"accuracy": 0.8, "f1_score": 0.8}

        # Run the training loop
        central_training_loop(data_handler, thresholds, max_iterations=2)

        # Check to see sampling was done
        self.assertGreater(len(data_handler.train_used_indices), 0)


if __name__ == "__main__":
    unittest.main()