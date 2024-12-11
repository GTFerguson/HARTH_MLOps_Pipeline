from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

class ModelTrainer:

    tolerance_thresholds = {
        'accuracy': 0.9,
        'f1_score': 0.85
    }


    def __init__ (self, label_column : str, feature_columns : list[str], random_state : int):
        self.label_column = label_column
        self.feature_columns = feature_columns
        self.random_state = random_state


    def train_model (self, training_data : pd.DataFrame) -> RandomForestClassifier:
        # Separate the feature and label columns
        x_train = training_data[self.feature_columns]
        y_train = training_data[self.label_column]

        # Initialise and train our model
        model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        model.fit(x_train, y_train)

        return model
    

    def test_model (self, model : RandomForestClassifier, test_data : pd.DataFrame):
        x_test = test_data[self.feature_columns]
        y_test = test_data[self.label_column]

        y_pred = model.predict(x_test)

        # Evaluate model
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nAccuracy: {accuracy}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))