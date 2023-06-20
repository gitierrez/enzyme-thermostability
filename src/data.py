import os
import pandas as pd


def load_dataset(data_dir: str):
    """
    Load the dataset from the given data directory.

    Args:
        data_dir (str): The directory path containing the dataset files.

    Returns:
        pd.DataFrame: The merged dataframe containing the loaded dataset.

    This function reads the training and test data files from the specified data directory and performs the following steps:
    1. Reads the 'train.csv', 'train_updates_20220929.csv', 'test.csv', and 'test_labels.csv' files.
    2. Updates the training dataframe with the information from the train_updates dataset.
    3. Adds labels to the test dataframe by joining it with the test_labels dataset.
    4. Merges both the training and test dataframes, adding a 'split' column to indicate the split.
    5. Resets the index of the merged dataframe.

    Example:
        data_dir = '/path/to/dataset'
        df = load_dataset(data_dir)
    """

    # read data files
    train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'), index_col='seq_id')
    train_updates = pd.read_csv(os.path.join(data_dir, 'train_updates_20220929.csv'), index_col='seq_id')
    test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'), index_col='seq_id')
    test_labels = pd.read_csv(os.path.join(data_dir, 'test_labels.csv'), index_col='seq_id')

    # update training dataframe
    train_df = _update_training_set(train_df, train_updates)

    # add labels to test dataframe
    test_df = test_df.join(test_labels, how='left')

    # merge both dataframes, keeping a column indicating the split
    train_df['split'] = 'train'
    test_df['split'] = 'test'
    df = pd.concat([train_df, test_df]).reset_index(drop=True)

    return df


def _update_training_set(train_df: pd.DataFrame, updates: pd.DataFrame):
    """
    Update the training dataset with new information from the updates dataset.

    Args:
        train_df (pd.DataFrame): The original training dataset.
        updates (pd.DataFrame): The dataset containing updates to be applied to the training dataset.

    Returns:
        pd.DataFrame: The updated training dataset.

    This function identifies records in the updates dataset where all values are NaN and drops those records from the
    training dataset. It then updates the 'pH' and 'tm' columns in the training dataset with the corresponding values
    from the updates dataset where records are not NaN.

    Note:
        - The indexes of the updates dataset must match the indexes of the training dataset for the update to be
          applied correctly.
        - The 'pH' and 'tm' columns in the training dataset will be modified in-place.

    Example:
        train_df = pd.read_csv('train_data.csv')
        updates = pd.read_csv('updates.csv')
        updated_train_df = _update_training_set(train_df, updates)
    """
    nan_records = updates.isnull().all(axis='columns')
    indexes_to_drop = updates[nan_records].index
    train_df = train_df.drop(index=indexes_to_drop)
    indexes_to_swap = updates[~nan_records].index
    train_df.loc[indexes_to_swap, ['pH', 'tm']] = updates.loc[indexes_to_swap, ['pH', 'tm']]
    return train_df
