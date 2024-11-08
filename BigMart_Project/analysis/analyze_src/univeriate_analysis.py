
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


class UniveriateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature: str, use_log_transform="original"):
        """
        Perform univariate analysis on specific variables of the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The name of the feature/column to be analyzed.
        use_log_transform (str): Specifies which distribution to plot.
                                 Options are "original", "log", or "both".

        Return:
        None: This method visualizes the distribution of the feature.
        """
        pass

class CategoricalUniveriateAnalysis(UniveriateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature: str, use_log_transform="original"):
        """
        Analyze the distribution of a categorical feature.
        """
        plt.figure(figsize=(10, 6))
        sns.countplot(x=feature, data=df, palette="muted")
        plt.title(f"Count Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Count")
        plt.show()


class NumericalUniveriateAnalysis(UniveriateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature: str, use_log_transform="original"):
        """
        Analyze the distribution of a numerical feature, with options to plot
        the original, log-transformed, or both distributions.

        Parameters:
        use_log_transform (str): Specifies which distribution to plot.
                                 Options are "original", "log", or "both".
        """
        if use_log_transform in ("original", "both"):
            # Plot the original distribution
            plt.figure(figsize=(10, 6))
            sns.displot(df[feature], color='red')
            plt.title(f"Original Distribution of {feature}")
            plt.xlabel(feature)
            plt.ylabel("Frequency")
            plt.show()

        if use_log_transform in ("log", "both"):
            # Handle zeros and negatives for log transformation
            #transformed_feature = df[feature].apply(lambda x: np.log(x) if x > 0 else np.log(1e-6))

            # Plot the log-transformed distribution
            plt.figure(figsize=(10, 6))
            sns.displot(np.log(df[feature]),color='red')
            plt.title(f"Log-Transformed Distribution of {feature}")
            plt.xlabel(f"Log of {feature}")
            plt.ylabel("Frequency")
            plt.show()


class UniveriateAnalyzer:
    def __init__(self, strategy: UniveriateAnalysisStrategy):
        """
        Initializes the UnivariateAnalyzer with a specific analysis strategy.

        Parameters:
        strategy (UniveriateAnalysisStrategy): The strategy to be used for univariate analysis.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: UniveriateAnalysisStrategy):
        """
        Sets a new strategy for the UnivariateAnalyzer.

        Parameters:
        strategy (UniveriateAnalysisStrategy): The new strategy to be used for univariate analysis.
        """
        self._strategy = strategy

    def execute_analysis(self, df: pd.DataFrame, feature: str, use_log_transform="original"):
        """
        Executes the univariate analysis using the current strategy.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The name of the feature/column to be analyzed.
        use_log_transform (str): Specifies which distribution to plot.
                                 Options are "original", "log", or "both".

        Returns:
        None: Executes the strategy's analysis method and visualizes the results.
        """
        self._strategy.analyze(df, feature, use_log_transform)


# Example usage
if __name__ == "__main__":
  
    pass