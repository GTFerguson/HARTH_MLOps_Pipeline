import pandas as pd
import mlflow
from mlflow import MlflowClient
import data_handler as dh
import model_trainer as mt

# Constants
LABEL_COLUMN = 'label'
FEATURE_COLUMNS = ['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']
RANDOM_STATE = 42
TRAIN_SAMPLE_SIZE = 2000
TEST_SAMPLE_SIZE = 4000

THRESHOLDS = {
    'accuracy': 0.85,
    'recall': 0.85,
    'precision': 0.85,
    'f1_score': 0.85
}


DRIFT_THRESHOLD = {
    "accuracy": 0.02,  # Maximum allowed drop in accuracy
    "f1_score": 0.02,  # Maximum allowed drop in F1-score
    "precision": 0.02,  # Maximum allowed drop in precision
    "recall": 0.02,  # Maximum allowed drop in recall
}



def cross_subject_train_and_test (data_handler : dh.DataHandler):
    '''
    This function performs a test attempting to train on one subjects data
    and test on another.

    Args:
        data_handler (dh.DataHandler): Data handling object that contains all relevant data. 
    '''

    # Pick our subjects
    test_subject = data_handler.test_subjects[0]
    train_subject = data_handler.train_subjects[0]

    mlflow.set_experiment("Cross-Subject Tests")

    iteration = 1
    parent_run_name = f"iteration_{iteration}_trained_on_{train_subject}"
    with mlflow.start_run(run_name=parent_run_name):
        mlflow.set_tag("iteration", iteration)
        mlflow.set_tag("train_subjects", train_subject)

        print(f"Creating stratified sample of training subject '{train_subject}'.")
        train_sample = data_handler.create_stratified_sample(
            data=data_handler.train_data,
            used_indices=data_handler.train_used_indices,
            sample_size=TRAIN_SAMPLE_SIZE,
            subject=train_subject
        )

        trainer = mt.ModelTrainer(LABEL_COLUMN, FEATURE_COLUMNS, RANDOM_STATE)
        model = trainer.train_model(train_sample)

        test_iteration = 1
        same_subject_run_name = f"test_{test_iteration}_subject_{train_subject}"

        # Log Same-Subject Test as Nested Run
        with mlflow.start_run(run_name=same_subject_run_name, nested=True):
            mlflow.set_tag("test_type", "same_subject")
            mlflow.set_tag("test_subjects", train_subject)

            # Get a sample from the same subject as we trained on
            same_subject_sample = data_handler.create_stratified_sample(
                data=data_handler.train_data,
                used_indices=data_handler.train_used_indices,
                sample_size=TEST_SAMPLE_SIZE,
                subject=train_subject
            )
            print("\nSame subject test")
            trainer.test_model(model, same_subject_sample)
        
        test_iteration += 1

        cross_subject_run_name = f"test_{test_iteration}_subject_{test_subject}"
        # Log Cross-Subject Test as Nested Run
        with mlflow.start_run(run_name=cross_subject_run_name, nested=True):
            mlflow.set_tag("test_type", "cross_subject")
            mlflow.set_tag("test_subjects", test_subject)

            # Sample taken from different subject than the one trained on
            print(f"Creating stratified sample of testing subject '{test_subject}'.")
            test_sample = data_handler.create_stratified_sample(
                data=data_handler.test_data, 
                used_indices=data_handler.test_used_indices,
                sample_size=TEST_SAMPLE_SIZE,
                subject=test_subject
            )
            print("\nCross-subject test")
            trainer.test_model(model, test_sample)


def test_model (
    data_handler: dh.DataHandler, 
    trainer: mt.ModelTrainer,
    model,
    test_sample: pd.DataFrame,
    thresholds: dict) -> tuple:
    """
    Perform a test on a given model using the provided sample and evaluating it's performance.

    Args:
        data_handler (dh.DataHandler): Data handling object.
        trainer (mt.ModelTrainer): Model trainer object.
        model: Trained model.
        test_sample (pd.DataFrame): Predefined test sample.
        thresholds (dict): Performance thresholds.

    Returns:
        bool: True if performance meets thresholds, False otherwise.
    """

    metrics = trainer.test_model(model, test_sample)

    # Success state
    if all(metrics[metric] >= thresholds[metric] for metric in thresholds):
        return True, metrics

    # Fail state
    return False, metrics


def test_cross_subject(
    data_handler: dh.DataHandler, 
    trainer: mt.ModelTrainer, model, 
    thresholds: dict) -> tuple[bool, dict]:
    """
    Perform cross-subject testing and evaluate performance.

    Args:
        data_handler (dh.DataHandler): Data handling object.
        trainer (mt.ModelTrainer): Model trainer object.
        model: Trained model.
        thresholds (dict): Performance thresholds.

    Returns:
        int: The number of tests failed.
        dict: Evaluation metrics for each test performed.
    """

    print("Performing cross-subject testing...")
    all_metrics = {}
    cross_subject_failures = 0

    for test_subject in data_handler.test_subjects:
        with mlflow.start_run(run_name=f"cross_subject_test_{test_subject}", nested=True):
            mlflow.set_tag("test_type", "cross-subject")
            mlflow.set_tag("test_subject", test_subject)
            print(f"Testing on {test_subject}...")

            test_sample = data_handler.create_stratified_sample(
                data=data_handler.test_data,
                used_indices=data_handler.test_used_indices,
                sample_size=TEST_SAMPLE_SIZE,
                subject=test_subject,
            )

            metrics = trainer.test_model(model, test_sample)
            all_metrics[test_subject] = metrics

            if not all(metrics[metric] >= thresholds[metric] for metric in thresholds):
                print(f"Cross-subject test failed for {test_subject}.")
                cross_subject_failures += 1

    # Fail state
    print(f"Cross-subject failures: {cross_subject_failures}")
    return cross_subject_failures, all_metrics


def train_on_subject (
    data_handler: dh.DataHandler, 
    cumulative_data: pd.DataFrame, 
    trainer : mt.ModelTrainer, 
    train_subject: str, 
):
    """
    Train and evaluate a model on a single training subject.

    Args:
        data_handler (dh.DataHandler): Data handling object.
        cumulative_data (pd.DataFrame): Previously sampled training data.
        trainer (mt.ModelTrainer): Model Training object.
        train_subject (str): Subject to train the model on.

    Returns:
        Model: A ML model trained on a sample from the set training subject.
        pd.DataFrame: Updated cumulative training data.
    """

    # Get a new sample from training subject
    train_sample = data_handler.create_stratified_sample(
        data=data_handler.train_data,
        used_indices=data_handler.train_used_indices,
        sample_size=TRAIN_SAMPLE_SIZE,
        subject=train_subject
    )

    # Combine new sample into cumulative data
    updated_cumulative_data = pd.concat([cumulative_data, train_sample])

    model = trainer.train_model(updated_cumulative_data)
    return model, updated_cumulative_data

    
def detect_drift (previous_metrics: dict, new_metrics: dict):
    '''
    Helper method to check if there is significant performance drift between two given metrics.

    Args:
        previous_metrics (dict): The previous iterations metrics.
        new_metrics (dict): The newest iterations metrics.

    Return:
        Bool: True if significant drift was detected.   
    '''
    drift_detected = any(
        (previous_metrics[metric] - new_metrics[metric]) > DRIFT_THRESHOLD[metric]
        for metric in THRESHOLDS
    )
    return drift_detected

    
def same_subject_test (
    data_handler: dh.DataHandler,
    trainer: mt.ModelTrainer,
    model,
    iteration: int,
    training_set_size: int,
    test_sample: dict,
    thresholds: dict
    ):

    mlflow.set_tag("test_type", "same-subject")
    mlflow.set_tag("iteration", iteration)
    mlflow.set_tag("training_set_size", training_set_size)

    successful, metrics = test_model(
        data_handler, trainer, model, test_sample, thresholds
    )
    if successful:    # Success state
        mlflow.log_metric("same_subject_pass", 1)
        print("Same-subject test passed.")
    else:
        mlflow.log_metric("same_subject_pass", 0)
        print("Same-subject test failed.")

    return successful, metrics
    
    
def register_model(model, model_name, cs_metrics):
    """
    Registers a given model in the MLFlow Model Registry.

    Args:
        model: The trained model to register.
        model_name (str): Name of the model to register.
        cs_metrics (dict): Cross-subject metrics to log with the model.

    Returns:
        None
    """

    # Log the model and metrics
    mlflow.sklearn.log_model(
        model, 
        artifact_path="model", 
        registered_model_name=model_name
    )

    # Need to interact with model registry to set additional tags
    client = MlflowClient(mlflow.get_tracking_uri())
    model_version = client.get_latest_versions(model_name, stages=['None'])[0].version

    # Initialize a dictionary to accumulate sums for each metric
    metric_sums = {metric: 0 for metric in next(iter(cs_metrics.values())).keys()}
    subject_count = len(cs_metrics)

        # Log cross-subject test metrics as tags
    for subject, metrics in cs_metrics.items():
        for metric, value in metrics.items():
            metric_sums[metric] += value # Gets sums for later usage
            # This stores subject specific results
            # commented out as it seems excessive
            # client.set_model_version_tag(
            #     name=model_name,
            #     version=model_version,
            #     key=f"{subject}_{metric}",
            #     value=value
            # )
    
    # We'll also calculate and log the average of each metric for quick review
    for metric, total in metric_sums.items():
        avg_value = total / subject_count
        client.set_model_version_tag(
            name=model_name,
            version=model_version,
            key=f"avg_{metric}",
            value=avg_value
        )
            
    print(f"Model registered as '{model_name}'.")


def promote_best_model(model_name: str, new_model_version: int):
    """
    Promote the best model based on avg_f1_score to the production stage.

    Args:
        model_name (str): Name of the registered model.
        new_model_version (int): Latest version number of the model being compared.

    Returns:
        None
    """
    client = MlflowClient()

    # Check if a model with alias 'production' exists
    model_versions = client.search_model_versions(f"name='{model_name}'")
    current_prod_version = None
    current_prod_avg_f1 = float("-inf")

    for version in model_versions:
        if "production" in version.aliases:
            current_prod_version = int(version.version)
            current_prod_avg_f1 = float(version.tags.get("avg_f1_score", "-inf"))
            break

    # Retrieve the new model's avg_f1_score
    new_model_avg_f1 = float(client.get_model_version(model_name, new_model_version).tags.get("avg_f1_score", "-inf"))

    if current_prod_version:
        # Compare the new model with the current production model
        if new_model_avg_f1 > current_prod_avg_f1:
            # Set alias 'production' to the new model
            client.set_model_version_alias(name=model_name, version=new_model_version, alias="production")

            # Remove alias 'production' from the old production model
            client.delete_model_version_alias(name=model_name, version=current_prod_version, alias="production")
            print(f"Model version {new_model_version} promoted to 'production'.")
        else:
            print(f"Model version {new_model_version} not promoted.")
    else:
        # No model currently in 'production', set the alias to the new model
        client.set_model_version_alias(name=model_name, version=new_model_version, alias="production")
        print(f"Model version {new_model_version} set as 'production'.")



def central_training_loop(data_handler: dh.DataHandler, thresholds: dict, max_iterations: int = 100):
    """
    Central training loop for iterative training and testing.

    Args:
        data_handler (dh.DataHandler): Data handling object with training and testing sets.
        thresholds (dict): Dictionary with performance thresholds (e.g., accuracy, f1_score).
        max_iterations (int): Maximum number of iterations to avoid infinite loops.
    """

    mlflow.set_experiment("Cross-Subject Model Training")

    # Initial setup
    iteration = 1
    train_subjects = list(data_handler.train_subjects)
    cumulative_data = pd.DataFrame()
    current_metrics = None

    # Ensure there is no run already active
    if mlflow.active_run():
        print(f"Ending active run: {mlflow.active_run().info.run_id}")
        mlflow.end_run()

    while iteration <= max_iterations and train_subjects:
        train_subject = train_subjects.pop(0)  # Select a training subject
        print(f"\nIteration {iteration} - Starting with subject {train_subject}...")
        
        with mlflow.start_run(run_name=f"iteration_{iteration}_trained_on_{train_subject}"):
            mlflow.set_tag("iteration", iteration)
            mlflow.set_tag("train_subject", train_subject)

            trainer = mt.ModelTrainer(LABEL_COLUMN, FEATURE_COLUMNS, RANDOM_STATE)
            model, cumulative_data = train_on_subject(data_handler, cumulative_data, trainer, train_subject)

            # Same-subject testing samples from each training subject 
            ss_samples = {
                subject: data_handler.create_stratified_sample(
                    data=data_handler.train_data,
                    used_indices=data_handler.train_used_indices,
                    sample_size=TEST_SAMPLE_SIZE,
                    subject=subject,
                )
                for subject in data_handler.train_subjects
            }

            # Cross-subject testing samples from each testing subject 
            cs_samples = {
                subject: data_handler.create_stratified_sample(
                    data=data_handler.test_data,
                    used_indices=data_handler.test_used_indices,
                    sample_size=TEST_SAMPLE_SIZE,
                    subject=subject,
                )
                for subject in data_handler.test_subjects
               
            }

            ss_iteration = 1
            ss_successful = False
            current_metrics = None

            with mlflow.start_run(run_name="same_subject_tests", nested=True):
                # Evaluate same-subject performance... 
                with mlflow.start_run(run_name=f"same_subject_test_{train_subject}_iteration_{ss_iteration}", nested=True):
                    ss_successful, current_metrics = same_subject_test(
                        data_handler, trainer, model, ss_iteration, len(cumulative_data), 
                        ss_samples[train_subject], THRESHOLDS
                    )

                # model is trained on same-subject until threshold metrics are met
                while not ss_successful:
                    ss_iteration += 1
                    print(f"Same-subject test failed for {train_subject}. Retraining with additional samples.")

                    with mlflow.start_run(run_name=f"same_subject_test_{train_subject}_iteration_{ss_iteration}", nested=True):
                        # Retrain model with new sample
                        new_model, new_cumulative_data = train_on_subject(
                            data_handler, cumulative_data, trainer, train_subject
                        )

                        ss_successful, latest_metrics = same_subject_test(
                            data_handler, trainer, model, ss_iteration, len(cumulative_data), 
                            ss_samples[train_subject], THRESHOLDS
                        )
                        
                        # Check if new sample has negatively impacted performance
                        if detect_drift (current_metrics, latest_metrics):
                            print("Performance drift detected! Rejecting the latest training sample.")
                            mlflow.log_metric("drift_detected", 1)
                        else: # No drift detected, save our new model and training data
                            print("No drift detected. Accepting the latest training sample.")
                            model = new_model
                            cumulative_data = new_cumulative_data
                            mlflow.log_metric("drift_detected", 0)

            # Evaluate cross-subject performance, if thresholds are not met we must keep training with a new subject
            with mlflow.start_run(run_name="cross_subject_tests", nested=True):
                cs_fail_count, cs_metrics = test_cross_subject(data_handler, trainer, model, thresholds)
                mlflow.log_metric("cross_subject_tests_failed", cs_fail_count)

                # Store our trained model in registry along with cross-subject metrics
                model_name = "harth_rfc"
                model_version = register_model (model, model_name, cs_metrics)

                # After a new model is registered check it's performance against previous versions
                # promote to production if it's the best performing


                if cs_fail_count == 0: # Model successfully trained, exit loop
                    print("Cross-subject tests have been passed!")
                    break
                else: # CS results have not met thresholds, continue to next subject 
                    print(f"Cross-subject test failed for {train_subject}.")

        print("Restarting with a new training subject.")
        iteration += 1

    # Stop condition handling
    if iteration > max_iterations:
        print("Maximum iterations reached. Training loop terminated.")
    elif not train_subjects:
        print("No more training subjects available. Training loop terminated.")


def main ():
    mlflow.set_tracking_uri('http://mlflow:5000')
    mlflow.autolog()
    data_handler = dh.DataHandler(LABEL_COLUMN, FEATURE_COLUMNS, RANDOM_STATE)
    #cross_subject_train_and_test(data_handler)
    central_training_loop(data_handler, THRESHOLDS, 100)


if __name__ == "__main__":
    main()