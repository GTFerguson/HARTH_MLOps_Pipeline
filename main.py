import data_handler as dh
import model_trainer as mt
import mlflow

# Constants
LABEL_COLUMN = 'label'
FEATURE_COLUMNS = ['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']
RANDOM_STATE = 42
TRAIN_SAMPLE_SIZE = 200
TEST_SAMPLE_SIZE = 400


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
        trained_model = trainer.train_model(train_sample)

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
            trainer.test_model(trained_model, same_subject_sample)
        
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
            trainer.test_model(trained_model, test_sample)


def main ():
    data_handler = dh.DataHandler(LABEL_COLUMN, FEATURE_COLUMNS, RANDOM_STATE)
    cross_subject_train_and_test(data_handler)




if __name__ == "__main__":
    main()