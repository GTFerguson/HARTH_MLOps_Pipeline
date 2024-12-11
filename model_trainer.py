from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

class ModelTrainer:

    tolerance_thresholds = {
        'accuracy': 0.9,
        'f1_score': 0.85
    }


    def __init__ (self, label_column : str, feature_columns : list[str], random_state : int):
        self.label_column = label_column
        self.feature_columns = feature_columns
        self.random_state = random_state
        mlflow.set_tracking_uri('http://127.0.0.1:5000')
        mlflow.autolog()


    def train_model (self, training_data : pd.DataFrame) -> RandomForestClassifier:
        """
        Train a Random Forest model and logs the training process with MLFlow.

        Args:
            training_data (pd.DataFrame): The data used for training.

        Returns:
            RandomForestClassifier: The trained Random Forest model.
        """
        # Separate the feature and label columns
        x_train = training_data[self.feature_columns]
        y_train = training_data[self.label_column]

        # Initialise and train our model
        model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        
        # Log model parameters
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", model.n_estimators)
        mlflow.log_param("random_state", self.random_state)
        # Train the model...
        model.fit(x_train, y_train)
        # ...then log it.
        # Log the model with an example input
        input_example = np.array([training_data.iloc[0].tolist()])
        mlflow.sklearn.log_model(model, "random_forest_model", input_example=input_example)

        return model
    

    def test_model (self, model : RandomForestClassifier, test_data : pd.DataFrame):
        x_test = test_data[self.feature_columns]
        y_test = test_data[self.label_column]

        y_pred = model.predict(x_test)

        # Evaluate model
        accuracy    = accuracy_score(y_test, y_pred)
        precision   = precision_score(y_test, y_pred, average='weighted')
        recall      = recall_score(y_test, y_pred, average='weighted')
        f1          = f1_score(y_test, y_pred, average='weighted')

        print(f"\nAccuracy: {accuracy}")
        print(f"Recall: {recall}")
        print(f"Precision: {precision}")
        print(f"F1 Score: {f1}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))

        # Log metrics to MLFlow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
