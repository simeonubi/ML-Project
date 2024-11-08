from abc import ABC, abstractmethod
import pandas as pd

# Abstract class for Data Inspection Strategy
class DataInspectionStrategy(ABC):
    @abstractmethod
    def inspect(self, df: pd.DataFrame):
        """Abstract method for inspecting data."""
        pass


class InfoInspection(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        """Displays info about the DataFrame."""
        print("DataFrame Info:")
        print(df.info())
        print("\n")

class DescribeInspection(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        """Displays descriptive statistics for numeric columns."""
        print("Numeric Columns Description:")
        print(df.describe())
        print("\n")

class DescribeCategoricalInspection(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        """Displays descriptive statistics for categorical columns."""
        print("Categorical Columns Description:")
        print(df.describe(include=['object']))
        print("\n")

class MissingValuesInspection(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        """Displays missing values count for each column."""
        print("Missing Values Count:")
        missing_values = df.isnull().sum()
        print(missing_values[missing_values > 0])
        print("\n")

class ValueCountsInspection(DataInspectionStrategy):
    def __init__(self, columns: list):
        
        self.columns = columns

    def inspect(self, df: pd.DataFrame):
        
        for column in self.columns:
            if column in df.columns:
                print(f"Value count for {column}")
                print(df[column].value_counts())

                print("\n")
            else:
                print(f"Warning: Column {column} not found in DataFrame. \n")


class DataInspector:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.strategies = []

    def add_strategy(self, strategy: DataInspectionStrategy):
        """Adds an inspection strategy to the list."""
        self.strategies.append(strategy)

    def inspect(self):
        """Applies all added inspection strategies to the DataFrame."""
        for strategy in self.strategies:
            strategy.inspect(self.df)


if __name__ == "__main__":
    
    # Create the DataInspector context
    #inspector = DataInspector(df)
    
    # Add various inspection strategies
    # inspector.add_strategy(InfoInspection())
    # inspector.add_strategy(DescribeInspection())
    # inspector.add_strategy(DescribeCategoricalInspection())
    # inspector.add_strategy(MissingValuesInspection())
    
    # Run the inspections
    #inspector.inspect()
    pass