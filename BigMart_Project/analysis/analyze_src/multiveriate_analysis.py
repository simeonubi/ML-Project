from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats import chi2_contingency, f_oneway
from sklearn.preprocessing import LabelEncoder

# Abstract Base Class for Analysis Strategies
class AnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, *args, **kwargs):
        pass

# Concrete Strategy for Correlation Heatmap
class CorrelationHeatmapStrategy(AnalysisStrategy):
    def analyze(self, df: pd.DataFrame, features=None):
        if features:
            df = df[features]
        plt.figure(figsize=(10, 10))
        sns.heatmap(df.corr(), annot=True, cmap="BrBG_r", square=True)
        plt.title("Correlation Heatmap")
        plt.show()

# Concrete Strategy for Chi-Squared Test
class ChiSquaredStrategy(AnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature1, feature2):
        cross_tab = pd.crosstab(df[feature1], df[feature2])
        chi2, p, dof, expected = chi2_contingency(cross_tab)
        print("Chi-squared Test Result:")
        print(f"Chi-squared: {chi2}, p-value: {p}, dof: {dof}")
        print("Expected frequencies:\n", expected)
        print("\nCross Tabulation:\n", cross_tab)

# Concrete Strategy for Cramér's V
class CramersVStrategy(AnalysisStrategy):
    def analyze(self, df: pd.DataFrame, features):
        label_encoder = LabelEncoder()
        encoded_df = df[features].apply(label_encoder.fit_transform)

        def cramers_v(x, y):
            cross_tab = pd.crosstab(x, y)
            chi2 = chi2_contingency(cross_tab)[0]
            n = cross_tab.sum().sum()
            r, k = cross_tab.shape
            return np.sqrt(chi2 / (n * (min(r, k) - 1)))
        
        results = pd.DataFrame(index=features, columns=features)
        for col1 in features:
            for col2 in features:
                results.loc[col1, col2] = cramers_v(encoded_df[col1], encoded_df[col2])
        print("Cramér's V Matrix:\n", results)

# Concrete Strategy for ANOVA
class ANOVAStrategy(AnalysisStrategy):
    def analyze(self, df: pd.DataFrame, group_feature, target_feature):
        groups = [df[target_feature][df[group_feature] == group].values for group in df[group_feature].unique()]
        f_stat, p_val = f_oneway(*groups)
        print(f"ANOVA Result: F-statistic = {f_stat}, p-value = {p_val}")

# Context Class for Multivariate Analysis
class MultivariateAnalyzer:
    def __init__(self, strategy: AnalysisStrategy = None):
        self.strategy = strategy

    def set_strategy(self, strategy: AnalysisStrategy):
        self.strategy = strategy

    def execute_analysis(self, df: pd.DataFrame, *args, **kwargs):
        if self.strategy is None:
            raise ValueError("Strategy not set. Please set a strategy before executing analysis.")
        self.strategy.analyze(df, *args, **kwargs)

# Example Usage
if __name__ == "__main__":
    # # Sample DataFrame for demonstration
    # data = pd.DataFrame({
    #     'Outlet_Size': ['Small', 'Medium', 'Medium', 'Large', 'Small'],
    #     'Outlet_Location_Type': ['Urban', 'Suburban', 'Urban', 'Rural', 'Suburban'],
    #     'Outlet_Type': ['Grocery', 'Supermarket', 'Grocery', 'Supermarket', 'Supermarket'],
    #     'Item_Outlet_Sales': [2000, 1500, 2200, 3000, 1600],
    #     'Gr Liv Area': [1500, 1700, 1600, 1800, 1400],
    #     'Overall Qual': [6, 7, 8, 9, 5],
    #     'Total Bsmt SF': [850, 900, 875, 950, 800],
    #     'Year Built': [2001, 2003, 2002, 2005, 1999]
    # })

    # # Create the analyzer with no initial strategy
    # analyzer = MultivariateAnalyzer()

    # # Correlation Heatmap
    # analyzer.set_strategy(CorrelationHeatmapStrategy())
    # analyzer.execute_analysis(data, features=['Gr Liv Area', 'Overall Qual', 'Total Bsmt SF', 'Year Built', 'Item_Outlet_Sales'])

    # # Chi-Squared Test
    # analyzer.set_strategy(ChiSquaredStrategy())
    # analyzer.execute_analysis(data, 'Outlet_Size', 'Outlet_Location_Type')

    # # Cramér's V
    # analyzer.set_strategy(CramersVStrategy())
    # analyzer.execute_analysis(data, features=['Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type'])

    # # ANOVA Test
    # analyzer.set_strategy(ANOVAStrategy())
    # analyzer.execute_analysis(data, 'Outlet_Location_Type', 'Item_Outlet_Sales')
    pass