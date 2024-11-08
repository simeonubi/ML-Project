import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler, LabelEncoder

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Abstract Base Class for Feature Engineering Strategy
# ----------------------------------------------------
# This class defines a common interface for different feature engineering strategies.
# Subclasses must implement the apply_transformation method.
class FeatureEngineeringStrategy(ABC):
    @abstractmethod
    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to apply feature engineering transformation to the DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.

        Returns:
        pd.DataFrame: A dataframe with the applied transformations.
        """
        pass


# Concrete Strategy for Log Transformation
# ----------------------------------------
# This strategy applies a logarithmic transformation to skewed features to normalize the distribution.
class OutletAGECalcation(FeatureEngineeringStrategy):
    def __init__(self, new_feature, old_feature):
        """
        Initializes the Outletageage generation with the specific features to transform.

        Parameters:
        features (list): The list of features to apply the log transformation to.
        """
        self.new_feature = new_feature
        self.old_feature = old_feature

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculaates the age of an outlet in the DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.

        Returns:
        pd.DataFrame: The dataframe with new-transformed features.
        """
        logging.info(f"Generating new feature: {self.new_feature}")
        df_transformed = df.copy()
        df_transformed[self.new_feature] = 2024 - df_transformed[self.old_feature]
        df_transformed = df_transformed.drop(columns=[self.old_feature])
        logging.info("Log transformation completed.")
        return df_transformed


# Concrete Strategy for Standard Scaling
# --------------------------------------
# This strategy applies standard scaling (z-score normalization) to features, centering them around zero with unit variance.
class StandardScaling(FeatureEngineeringStrategy):
    def __init__(self, features):
        """
        Initializes the StandardScaling with the specific features to scale.

        Parameters:
        features (list): The list of features to apply the standard scaling to.
        """
        self.features = features
        self.scaler = StandardScaler()

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies standard scaling to the specified features in the DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.

        Returns:
        pd.DataFrame: The dataframe with scaled features.
        """
        logging.info(f"Applying standard scaling to features: {self.features}")
        df_transformed = df.copy()
        df_transformed[self.features] = self.scaler.fit_transform(df[self.features])
        logging.info("Standard scaling completed.")
        return df_transformed


# Concrete Strategy for Min-Max Scaling
# -------------------------------------
# This strategy applies Min-Max scaling to features, scaling them to a specified range, typically [0, 1].
class MinMaxScaling(FeatureEngineeringStrategy):
    def __init__(self, features, feature_range=(0, 1)):
        """
        Initializes the MinMaxScaling with the specific features to scale and the target range.

        Parameters:
        features (list): The list of features to apply the Min-Max scaling to.
        feature_range (tuple): The target range for scaling, default is (0, 1).
        """
        self.features = features
        self.scaler = MinMaxScaler(feature_range=feature_range)

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies Min-Max scaling to the specified features in the DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.

        Returns:
        pd.DataFrame: The dataframe with Min-Max scaled features.
        """
        logging.info(
            f"Applying Min-Max scaling to features: {self.features} with range {self.scaler.feature_range}"
        )
        df_transformed = df.copy()
        df_transformed[self.features] = self.scaler.fit_transform(df[self.features])
        logging.info("Min-Max scaling completed.")
        return df_transformed


# Concrete Strategy for One-Hot Encoding
# --------------------------------------
# This strategy applies one-hot encoding to categorical features, converting them into binary vectors.
class OneHotEncoding(FeatureEngineeringStrategy):
    def __init__(self, features):
        """
        Initializes the OneHotEncoding with the specific features to encode.

        Parameters:
        features (list): The list of categorical features to apply the one-hot encoding to.
        """
        self.features = features
        self.encoder = OneHotEncoder(sparse=False, drop="first")

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies one-hot encoding to the specified categorical features in the DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.

        Returns:
        pd.DataFrame: The dataframe with one-hot encoded features.
        """
        logging.info(f"Applying one-hot encoding to features: {self.features}")
        df_transformed = df.copy()
        encoded_df = pd.DataFrame(
            self.encoder.fit_transform(df[self.features]),
            columns=self.encoder.get_feature_names_out(self.features),
        )
        df_transformed = df_transformed.drop(columns=self.features).reset_index(drop=True)
        df_transformed = pd.concat([df_transformed, encoded_df], axis=1)
        logging.info("One-hot encoding completed.")
        return df_transformed


class LabelEncodingStrategy(FeatureEngineeringStrategy):
    def __init__(self, features):
        """
        Initializes the LabelEncodingStrategy with the specific features to encode.

        Parameters:
        features (list): The list of categorical features to apply label encoding to.
        """
        self.features = features
        self.label_encoders = {feature: LabelEncoder() for feature in features}

    def apply_transformation(self, df: pd.DataFrame, apply_dummies=True, drop_columns=None) -> pd.DataFrame:
        """
        Applies label encoding to specified categorical features and optionally applies dummy encoding.

        Parameters:
        df (pd.DataFrame): The DataFrame containing features to transform.
        apply_dummies (bool): Whether to apply one-hot encoding after label encoding.
        drop_columns (list): List of columns to drop after encoding.

        Returns:
        pd.DataFrame: The transformed DataFrame.
        """
        logging.info(f"Applying label encoding to features: {self.features}")
        df_transformed = df.copy()
        
        # Apply label encoding to specified columns
        for feature in self.features:
            logging.info(f"Label encoding feature: {feature}")
            df_transformed[feature] = self.label_encoders[feature].fit_transform(df[feature])
        
        # Drop specified columns if needed
        if drop_columns:
            logging.info(f"Dropping columns: {drop_columns}")
            df_transformed.drop(columns=drop_columns, inplace=True)

        # Optionally apply one-hot encoding
        if apply_dummies:
            df_transformed = pd.get_dummies(df_transformed)
            logging.info("One-hot encoding completed on transformed DataFrame.")

        logging.info("Label encoding completed.")
        return df_transformed

# Context Class for Feature Engineering
# -------------------------------------
# This class uses a FeatureEngineeringStrategy to apply transformations to a dataset.
class FeatureEngineer:
    def __init__(self, strategy: FeatureEngineeringStrategy):
        """
        Initializes the FeatureEngineer with a specific feature engineering strategy.

        Parameters:
        strategy (FeatureEngineeringStrategy): The strategy to be used for feature engineering.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: FeatureEngineeringStrategy):
        """
        Sets a new strategy for the FeatureEngineer.

        Parameters:
        strategy (FeatureEngineeringStrategy): The new strategy to be used for feature engineering.
        """
        logging.info("Switching feature engineering strategy.")
        self._strategy = strategy

    def apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes the feature engineering transformation using the current strategy.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.

        Returns:
        pd.DataFrame: The dataframe with applied feature engineering transformations.
        """
        logging.info("Applying feature engineering strategy.")
        return self._strategy.apply_transformation(df)


# Example usage
if __name__ == "__main__":
    # Example dataframe
    # df = pd.read_csv('../extracted-data/your_data_file.csv')

    # Log Transformation Example
    # log_transformer = FeatureEngineer(LogTransformation(features=['SalePrice', 'Gr Liv Area']))
    # df_log_transformed = log_transformer.apply_feature_engineering(df)

    # Standard Scaling Example
    # standard_scaler = FeatureEngineer(StandardScaling(features=['SalePrice', 'Gr Liv Area']))
    # df_standard_scaled = standard_scaler.apply_feature_engineering(df)

    # Min-Max Scaling Example
    # minmax_scaler = FeatureEngineer(MinMaxScaling(features=['SalePrice', 'Gr Liv Area'], feature_range=(0, 1)))
    # df_minmax_scaled = minmax_scaler.apply_feature_engineering(df)

    # One-Hot Encoding Example
    # onehot_encoder = FeatureEngineer(OneHotEncoding(features=['Neighborhood']))
    # df_onehot_encoded = onehot_encoder.apply_feature_engineering(df)

    pass
