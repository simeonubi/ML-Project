from abc import ABC, abstractmethod
import pandas as pd
import logging

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Abstract Base Class for Data Transformation Strategy
class DataTransformationStrategy(ABC):
    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to apply a transformation on the DataFrame.
        
        Parameters:
        df (pd.DataFrame): The input DataFrame.

        Returns:
        pd.DataFrame: The transformed DataFrame.
        """
        pass


class ImputeByGroupMaxStrategy(DataTransformationStrategy):
    def __init__(self, target_column: str, group_column: str):
        """
        Initialize the strategy with target and group column names.

        Parameters:
        target_column (str): The column to impute missing values in.
        group_column (str): The column to group by for calculating max values.
        """
        self.target_column = target_column
        self.group_column = group_column

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Imputing missing values in '{self.target_column}' by max of '{self.group_column}' groups.")
        df[self.target_column] = df[self.target_column].fillna(df.groupby(self.group_column)[self.target_column].transform('max'))
        return df


class ImputeWithDefaultValueStrategy(DataTransformationStrategy):
    def __init__(self, target_column: str, default_value):
        """
        Initialize the strategy with the target column and default fill value.

        Parameters:
        target_column (str): The column to impute missing values in.
        default_value (any): The value to use for imputing missing entries.
        """
        self.target_column = target_column
        self.default_value = default_value

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Imputing missing values in '{self.target_column}' with default value '{self.default_value}'.")
        df[self.target_column] = df[self.target_column].fillna(self.default_value)
        return df


class DeleteRowsWithMissingValuesStrategy(DataTransformationStrategy):
    def __init__(self, target_column: str):
        """
        Initialize the strategy with the target column to check for missing values.

        Parameters:
        target_column (str): The column to check for missing values before deletion.
        """
        self.target_column = target_column

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Deleting rows with missing values in '{self.target_column}'.")
        df = df[df[self.target_column].notna()]
        return df


class StandardizeValuesStrategy(DataTransformationStrategy):
    def __init__(self, target_column: str, replacements: dict):
        """
        Initialize the strategy with the target column and a dictionary of replacements.

        Parameters:
        target_column (str): The column to standardize values in.
        replacements (dict): A dictionary with keys as current values and values as standardized replacements.
        """
        self.target_column = target_column
        self.replacements = replacements

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Standardizing values in '{self.target_column}' with replacements: {self.replacements}.")
        df[self.target_column].replace(self.replacements, inplace=True)
        return df


class DataTransformer:
    def __init__(self, strategy: DataTransformationStrategy):
        """
        Initializes the DataTransformer with a specific transformation strategy.

        Parameters:
        strategy (DataTransformationStrategy): The strategy to be used for transforming the data.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: DataTransformationStrategy):
        """
        Sets a new transformation strategy for the DataTransformer.

        Parameters:
        strategy (DataTransformationStrategy): The new strategy to be used for transforming the data.
        """
        logging.info("Switching data transformation strategy.")
        self._strategy = strategy

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes the transformation using the current strategy.

        Parameters:
        df (pd.DataFrame): The input DataFrame.

        Returns:
        pd.DataFrame: The transformed DataFrame.
        """
        logging.info("Applying data transformation strategy.")
        return self._strategy.transform(df)
    
    
if __name__ == "__main__":

    pass