import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import numpy as np

def handle_missing(data, test_data=None, target=None, strategy='drop_and_impute', fill_value=None, threshold=50, columns=None, test_size=0.2, random_state=None, verbose=False):
    """
    Handle missing data in train and test DataFrames using specified strategy.

    Parameters:
    - data (pd.DataFrame): The input data frame (either the whole dataset or training dataset if test_data is provided).
    - test_data (pd.DataFrame, optional): The testing data frame. If None, data will be split into train and test.
    - target (str): The name of the target column in the training data. If None, will attempt to determine it automatically.
    - strategy (str): The strategy to handle missing data.
        Options:
        - 'auto': Automatically decide strategy based on data type.
        - 'mean': Impute with mean.
        - 'median': Impute with median.
        - 'mode': Impute with mode.
        - 'constant': Fill missing values with a constant value.
        - 'drop': Drop rows or columns based on threshold.
        - 'interpolate': Interpolate missing values.
        - 'flag': Flag missing values with an indicator column.
        - 'predict': Predict missing values using machine learning algorithms (not implemented).
        - 'drop_and_impute': Drop columns with missing values above the threshold and impute the rest.
    - fill_value (any): Value to fill missing values if strategy is 'constant'.
    - threshold (float): Threshold percentage to drop columns if strategy is 'drop'.
    - columns (list): List of columns to apply the strategy. If None, apply to all columns except the target column.
    - test_size (float): The proportion of the dataset to include in the test split if test_data is None.
    - random_state (int, optional): Random seed for reproducibility of the train-test split.
    - verbose (bool): If True, print information about columns dropped and values imputed.

    Returns:
    - pd.DataFrame, pd.DataFrame: DataFrames with missing values handled for train and test sets.
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame")
    
    if test_data is not None and not isinstance(test_data, pd.DataFrame):
        raise ValueError("test_data must be a pandas DataFrame if provided")

    if test_data is None:
        train, test = train_test_split(data, test_size=test_size, random_state=random_state)
    else:
        train, test = data, test_data

    # Automatically detect target column if not provided
    if target is None:
        potential_targets = [col for col in train.columns if col not in test.columns]
        if len(potential_targets) != 1:
            raise ValueError("Could not automatically determine the target column. Please specify the target column explicitly.")
        target = potential_targets[0]

    if columns is None:
        columns = [col for col in train.columns if col != target]

    if strategy in ['drop', 'drop_and_impute']:
        threshold /= 100.0
        missing_percentage = train.isnull().sum() / train.shape[0]
        columns_to_keep = missing_percentage[missing_percentage <= threshold].index
        dropped_columns = missing_percentage[missing_percentage > threshold].index.tolist()
        if target in train.columns and target not in columns_to_keep:
            columns_to_keep = columns_to_keep.insert(0, target)
        train = train.loc[:, columns_to_keep]
        test = test.loc[:, columns_to_keep.drop(target, errors='ignore')]
        columns = [col for col in train.columns if col != target]
        if strategy == 'drop':
            if verbose:
                print(f"Dropped columns: {dropped_columns}")
            return train, test
        if verbose:
            print(f"Dropped columns: {dropped_columns}")

    if strategy in ['auto', 'drop_and_impute']:
        imputers = {}
        for col in columns:
            if train[col].isnull().sum() == 0 and test[col].isnull().sum() == 0:
                continue
            original_train_na = train[col].isnull().sum()
            original_test_na = test[col].isnull().sum()
            if train[col].dtype in ['float64', 'int64']:
                imputer = SimpleImputer(strategy='mean')
            elif train[col].dtype == 'object' or train[col].dtype == 'bool' or train[col].dtype == 'category':
                imputer = SimpleImputer(strategy='most_frequent')
            elif pd.api.types.is_datetime64_any_dtype(train[col]):
                train[col] = train[col].fillna(method='ffill').fillna(method='bfill')
                test[col] = test[col].fillna(method='ffill').fillna(method='bfill')
                if verbose:
                    print(f"Column {col} datetime filled with forward and backward fill in train and test sets.")
                continue
            else:
                raise ValueError(f"Invalid data type for column {col}")
            train[col] = imputer.fit_transform(train[[col]]).ravel()
            test[col] = imputer.transform(test[[col]]).ravel()
            imputers[col] = imputer
            if verbose:
                print(f"Column {col} transformed and/or imputed with {imputer.strategy}. Train missing before: {original_train_na}, after: {train[col].isnull().sum()}")
                print(f"Column {col} imputed with {imputer.strategy}. Test missing before: {original_test_na}, after: {test[col].isnull().sum()}")
    elif strategy in ['mean', 'median', 'most_frequent', 'constant']:
        if strategy == 'constant' and fill_value is None:
            raise ValueError("fill_value must be provided when strategy is 'constant'")
        imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
        for col in columns:
            if train[col].isnull().sum() == 0 and test[col].isnull().sum() == 0:
                continue
            original_train_na = train[col].isnull().sum()
            original_test_na = test[col].isnull().sum()
            train[col] = imputer.fit_transform(train[[col]]).ravel()
            test[col] = imputer.transform(test[[col]]).ravel()
            if verbose:
                print(f"Column {col} transformed and/or imputed with {imputer.strategy}. Train missing before: {original_train_na}, after: {train[col].isnull().sum()}")
                print(f"Column {col} imputed with {imputer.strategy}. Test missing before: {original_test_na}, after: {test[col].isnull().sum()}")
    elif strategy == 'interpolate':
        for col in columns:
            if train[col].isnull().sum() == 0 and test[col].isnull().sum() == 0:
                continue
            original_train_na = train[col].isnull().sum()
            original_test_na = test[col].isnull().sum()
            train[col] = train[col].interpolate()
            test[col] = test[col].interpolate()
            if verbose:
                print(f"Column {col} interpolated. Train missing before: {original_train_na}, after: {train[col].isnull().sum()}")
                print(f"Column {col} interpolated. Test missing before: {original_test_na}, after: {test[col].isnull().sum()}")
    elif strategy == 'flag':
        for col in columns:
            if train[col].isnull().sum() == 0 and test[col].isnull().sum() == 0:
                continue
            train[col + '_missing_flag'] = train[col].isnull().astype(int)
            test[col + '_missing_flag'] = test[col].isnull().astype(int)
            original_train_na = train[col].isnull().sum()
            original_test_na = test[col].isnull().sum()
            train[col] = train[col].fillna(fill_value)
            test[col] = test[col].fillna(fill_value)
            if verbose:
                print(f"Column {col} flagged and filled. Train missing before: {original_train_na}, after: {train[col].isnull().sum()}")
                print(f"Column {col} flagged and filled. Test missing before: {original_test_na}, after: {test[col].isnull().sum()}")
    elif strategy == 'predict':
        raise NotImplementedError("Prediction-based imputation is not implemented yet.")
    else:
        raise ValueError("Invalid strategy. Choose from 'auto', 'mean', 'median', 'most_frequent', 'constant', 'drop', 'interpolate', 'flag', 'predict'")

    return train, test

def report_missing(data, test_data=None):
    """
    Generate a report of missing values in the data.

    Parameters:
    - data (pd.DataFrame): The input data frame (either the whole dataset or training dataset if test_data is provided).
    - test_data (pd.DataFrame, optional): The testing data frame. If None, only the data will be reported.

    Returns:
    - pd.DataFrame: DataFrame with missing values report.
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame")

    if test_data is not None and not isinstance(test_data, pd.DataFrame):
        raise ValueError("test_data must be a pandas DataFrame if provided")

    def missing_report(df, dataset_name="Data"):
        missing_data = df.isnull().sum()
        total_data = df.shape[0]
        missing_percentage = (missing_data / total_data) * 100

        report = pd.DataFrame({
            f"{dataset_name} Missing Values": missing_data,
            f"{dataset_name} Missing Percentage": missing_percentage
        })

        # Sort by the percentage of missing values in descending order
        report = report[report[f"{dataset_name} Missing Values"] > 0].sort_values(by=f"{dataset_name} Missing Percentage", ascending=False)
        return report

    train_report = missing_report(data, dataset_name="Train")

    if test_data is not None:
        test_report = missing_report(test_data, dataset_name="Test")
        combined_report = train_report.join(test_report, how='outer')
        
        # Sort the combined report by the percentage of missing values
        combined_report = combined_report.sort_values(by=[
            'Train Missing Percentage', 'Test Missing Percentage'], ascending=False)
        return combined_report

    return train_report
