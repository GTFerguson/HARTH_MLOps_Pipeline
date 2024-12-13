import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

class DataHandler:
    # Constants
    DATASET_DIR = 'harth/'
    NUM_OF_TEST_SUBJECTS = 10   # Number of subjects set aside to be used ONLY for testing


    # For readability we can use this to convert numeric labels to the corresponding activity name
    LABEL_MAPPING = {
        1: "walking",
        2: "running",
        3: "shuffling",
        4: "stairs (ascending)",
        5: "stairs (descending)",
        6: "standing",
        7: "sitting",
        8: "lying",
        13: "cycling (sit)",
        14: "cycling (stand)",
        130: "cycling (sit, inactive)",
        140: "cycling (stand, inactive)"
    }


    # Class Constructor
    def __init__ (self, label_column : str, feature_columns : list[str], random_state : int):
        self.label_column = label_column
        self.feature_columns = feature_columns
        self.random_state = random_state

        # Initialise empty sets for indices to be added later as they are used for samples
        self.train_used_indices = set()  
        self.test_used_indices = set()  

        self.dataset = self.load_datasets(self.DATASET_DIR)
        self.dataset = self.normalize_features(self.dataset)
        self.split_subjects(self.dataset)


    def load_datasets (self, dataset_dir : str) -> pd.DataFrame:
        # Generate a list of all dataset filepaths
        filepaths = [os.path.join(dataset_dir, file) 
                     for file in os.listdir(dataset_dir) if file.endswith('.csv')]

        df_list = []
        for file in filepaths:
            try:
                temp_df = pd.read_csv(file)
                # Add Subject column to our dataframe, extracting name from filepath
                temp_df['subject'] = os.path.basename(file).split('.')[0]
                df_list.append(temp_df)
            except Exception as e:
                print(f"Error reading {file}:\n{e}")

        # Join all dataframes into one
        data = pd.concat(df_list, ignore_index=True)
        return data


    def normalize_features (self, dataset : pd.DataFrame) -> pd.DataFrame:
        scalar = StandardScaler()
        dataset[self.feature_columns] = scalar.fit_transform(dataset[self.feature_columns])
        return dataset


    # Split the dataset by subjects, setting aside some for testing
    def split_subjects (self, dataset: pd.DataFrame):
        # Retrieve the names of all subjects
        subjects = dataset['subject'].unique()
        
        self.train_subjects, self.test_subjects = train_test_split(
            subjects, test_size=self.NUM_OF_TEST_SUBJECTS, random_state=self.random_state
        )

        print("\nTraining and testing subjects have been selected!\n")
        print(f"Train Subjects: {self.train_subjects}")
        print(f"Test Subjects: {self.test_subjects}\n")

        # Split data into training and test sets by subject
        self.train_data = dataset[dataset['subject'].isin(self.train_subjects)]
        self.test_data = dataset[dataset['subject'].isin(self.test_subjects)]
        
        print("\nTrain Set Class Distribution:\n", 
              self.train_data[self.label_column].value_counts(normalize=True))
        print("\nTest Set Class Distribution:\n", 
              self.test_data[self.label_column].value_counts(normalize=True))


    def create_stratified_sample (self, data : pd.DataFrame, used_indices: set, sample_size : int, subject : str = None) -> pd.DataFrame:
        """
        Create a stratified sample from the given dataframe while avoiding reuse of indices.
        Optionally, filter the data for a specific subject.

        Args:
            data (pd.DataFrame): The dataframe to sample from.
            used_indices (set): A set of indices already used.
            sample_size (int): Total number of rows to sample.
            subject (str, optional): Filter for a specific subject. Defaults to None.

        Returns:
            pd.DataFrame: A stratified sample from the dataframe or a subject-specific stratified sample.
        """
        
        # Nested function to aid with creating a stratified sample
        def sample_group (group):
            # Workout which indices have been unused
            available_indices = group.index.difference(used_indices)
            available_data = group.loc[available_indices]

            # Calculate the proportional sample size for the group
            group_proportion = len(group) / len(data)  # Proportion of the group in the dataset
            target_sample_size = int(group_proportion * sample_size)  # Target sample size for the group

            # Ensure the sample size does not exceed available rows
            final_sample_size = min(target_sample_size, len(available_data))

            # Sample from the available data
            sampled_data = available_data.sample(
                n=final_sample_size,
                random_state=self.random_state
            )

            # Update used_indices with the sampled indices
            used_indices.update(sampled_data.index)
            return sampled_data

        # If a specific subject is provided we filter it from our dataframe
        if subject:
            data = data[data['subject'] == subject]
            if data.empty:
                raise ValueError(f"No data found for subject '{subject}!")

        stratified_sample = data.groupby(self.label_column, group_keys=False).apply(sample_group)
        return stratified_sample