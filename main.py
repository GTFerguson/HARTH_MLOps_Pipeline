import data_handler as dh
import model_trainer as mt

# Constants
LABEL_COLUMN = 'label'
FEATURE_COLUMNS = ['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']
RANDOM_STATE = 42


def main ():
    data_handler = dh.DataHandler(LABEL_COLUMN, FEATURE_COLUMNS, RANDOM_STATE)
    train_sample = data_handler.create_stratified_sample(
        data_handler.train_data, data_handler.train_used_indices, 1000)
    test_sample = data_handler.create_stratified_sample(
        data_handler.test_data, data_handler.test_used_indices, 1000)

    trainer = mt.ModelTrainer(LABEL_COLUMN, FEATURE_COLUMNS, RANDOM_STATE)
    trained_model = trainer.train_model(train_sample)
    trainer.test_model(trained_model, test_sample)


if __name__ == "__main__":
    main()