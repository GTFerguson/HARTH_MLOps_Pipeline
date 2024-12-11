import pandas as pd
import mlflow
import data_handler as dh
import model_trainer as mt

# Constants
LABEL_COLUMN = 'label'
FEATURE_COLUMNS = ['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']
RANDOM_STATE = 42
TRAIN_SAMPLE_SIZE = 500
TEST_SAMPLE_SIZE = 500

THRESHOLDS = {
    'accuracy': 0.9,
    'f1_score': 0.85
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


def test_same_subject(data_handler: dh.DataHandler, trainer: mt.ModelTrainer, model, train_subject: str, thresholds: dict, iteration: int) -> bool:
    """
    Perform same-subject testing and evaluate performance.

    Args:
        data_handler (dh.DataHandler): Data handling object.
        trainer (mt.ModelTrainer): Model trainer object.
        model: Trained model.
        train_subject (str): Subject used for training.
        thresholds (dict): Performance thresholds.
        iteration (int): The iteration of the current test attempt.

    Returns:
        bool: True if performance meets thresholds, False otherwise.
    """

    with mlflow.start_run(run_name=f"same_subject_test_{train_subject}_iteration_{iteration}", nested=True):
        mlflow.set_tag("iteration", iteration)
        print(f"Performing same-subject test on {train_subject}...")
        same_subject_sample = data_handler.create_stratified_sample(
            data=data_handler.train_data,
            used_indices=data_handler.train_used_indices,
            sample_size=TEST_SAMPLE_SIZE,
            subject=train_subject,
        )
        metrics = trainer.test_model(model, same_subject_sample)

        # Success state
        if all(metrics[metric] >= thresholds[metric] for metric in thresholds):
            mlflow.log_metric("same_subject_pass", 1)
            print("Same-subject test passed.")
            return True

        # Fail state
        mlflow.log_metric("same_subject_pass", 0)
        print("Same-subject test failed.")
        return False


def test_cross_subject(data_handler: dh.DataHandler, trainer: mt.ModelTrainer, model, thresholds: dict) -> bool:
    """
    Perform cross-subject testing and evaluate performance.

    Args:
        data_handler (dh.DataHandler): Data handling object.
        trainer (mt.ModelTrainer): Model trainer object.
        model: Trained model.
        thresholds (dict): Performance thresholds.

    Returns:
        bool: True if all cross-subject tests meet thresholds, False otherwise.
    """
    print("Performing cross-subject testing...")
    cross_subject_failures = 0

    for test_subject in data_handler.test_subjects:
        with mlflow.start_run(run_name=f"cross_subject_test_{test_subject}", nested=True):
            mlflow.set_tag("test_subject", test_subject)
            print(f"Testing on {test_subject}...")

            test_sample = data_handler.create_stratified_sample(
                data=data_handler.test_data,
                used_indices=data_handler.test_used_indices,
                sample_size=TEST_SAMPLE_SIZE,
                subject=test_subject,
            )

            metrics = trainer.test_model(model, test_sample)

            if not all(metrics[metric] >= thresholds[metric] for metric in thresholds):
                print(f"Cross-subject test failed for {test_subject}.")
                cross_subject_failures += 1

    # Success state
    if cross_subject_failures == 0:
        print("All cross-subject tests passed.")
        return True

    # Fail state
    print(f"Cross-subject failures: {cross_subject_failures}")
    return False


def train_on_subject (data_handler: dh.DataHandler, cumulative_data: pd.DataFrame, trainer : mt.ModelTrainer, train_subject: str, thresholds: dict, iteration: int):
    """
    Train and evaluate a model on a single training subject.

    Args:
        data_handler (dh.DataHandler): Data handling object.
        cumulative_data (pd.DataFrame): Previously sampled training data.
        trainer (mt.ModelTrainer): Model Training object.
        train_subject (str): Subject to train the model on.
        thresholds (dict): Performance thresholds.
        iteration (int): Current iteration.

    Returns:
        Model: A ML model trained on a sample from the set training subject.
    """

    # Train the model
    train_sample = data_handler.create_stratified_sample(
        data=data_handler.train_data,
        used_indices=data_handler.train_used_indices,
        sample_size=TRAIN_SAMPLE_SIZE,
        subject=train_subject,
    )

    # Combine new sample into cumulative data
    updated_cumulative_data = pd.concat([cumulative_data, train_sample])

    model = trainer.train_model(updated_cumulative_data)

    return model, updated_cumulative_data


def central_training_loop(data_handler: dh.DataHandler, thresholds: dict, max_iterations: int = 100):
    """
    Central training loop for iterative training and testing.

    Args:
        data_handler (dh.DataHandler): Data handling object with training and testing sets.
        thresholds (dict): Dictionary with performance thresholds (e.g., accuracy, f1_score).
        max_iterations (int): Maximum number of iterations to avoid infinite loops.
    """

    mlflow.set_experiment("Cross-Subject Model Training")

    iteration = 1
    train_subjects = list(data_handler.train_subjects)
    cumulative_data = pd.DataFrame()

    while iteration <= max_iterations and train_subjects:
        train_subject = train_subjects.pop(0)  # Select a training subject
        with mlflow.start_run(run_name=f"iteration_{iteration}_trained_on_{train_subject}"):
            mlflow.set_tag("iteration", iteration)
            mlflow.set_tag("train_subject", train_subject)
            print(f"\nIteration {iteration} - Starting...")

            trainer = mt.ModelTrainer(LABEL_COLUMN, FEATURE_COLUMNS, RANDOM_STATE)
            model, cumulative_data = train_on_subject(data_handler, cumulative_data, trainer, train_subject, thresholds, iteration)

            same_sub_iter = 1
            # Evaluate same-subject performance, model is trained on same-subject until threshold metrics are met
            while not test_same_subject(data_handler, trainer, model, train_subject, thresholds, iteration):
                print(f"Same-subject test failed for {train_subject}.")
                model, cumulative_data = train_on_subject(data_handler, cumulative_data, trainer, train_subject, thresholds, same_sub_iter)
                same_sub_iter += 1

            # Evaluate cross-subject performance, if thresholds are not met we must keep training with a new subject
            if not test_cross_subject(data_handler, trainer, model, thresholds):
                print(f"Cross-subject test failed for {train_subject}.")
                mlflow.log_metric("cross_subject_pass", 0)
            else: # Model successfully trained, exit loop
                mlflow.log_metric("cross_subject_pass", 1)
                break

        print("Restarting with a new training subject.")
        iteration += 1

    # Stop condition handling
    if iteration > max_iterations:
        print("Maximum iterations reached. Training loop terminated.")
    elif not train_subjects:
        print("No more training subjects available. Training loop terminated.")


def main ():
    data_handler = dh.DataHandler(LABEL_COLUMN, FEATURE_COLUMNS, RANDOM_STATE)
    #cross_subject_train_and_test(data_handler)
    central_training_loop(data_handler, THRESHOLDS, 100)


if __name__ == "__main__":
    main()