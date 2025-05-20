import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
from factor_analyzer.factor_analyzer import calculate_kmo
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_polychoric

class DiscreteFactorAnalysisWrapper:
    def __init__(self, data: pd.DataFrame, n_factors: int = 3):
        """
        Initialize the Discrete Factor Analysis Wrapper.
        :param data: pd.DataFrame, the data for factor analysis (discrete or ranked).
        :param n_factors: int, number of factors.
        """
        self.data = data
        self.n_factors = n_factors
        self.correlation_matrix = None
        self.fa = None
        self.loadings_df = None
        self.communalities_df = None
        self.uniquenesses_df = None
        self.factor_scores = None
        self._factors_lbls = [f'Factor {i + 1}' for i in range(self.n_factors)]
        self._prepare_data()

    def _prepare_data(self):
        """
        Prepare data by calculating polychoric correlations.
        """
        # Perform polychoric correlation
        self.correlation_matrix = calculate_polychoric(self.data)

    def fit(self):
        """
        Fit the factor analysis model.
        """
        self.fa = FactorAnalyzer(n_factors=self.n_factors, method='ml', rotation=None)
        self.fa.fit(self.data)
        self._calculate_loadings()

    def _calculate_loadings(self):
        """
        Calculate factor loadings, communalities, and uniquenesses.
        """
        self.loadings_df = pd.DataFrame(self.fa.loadings_, columns=self._factors_lbls, index=self.data.columns)
        self.communalities_df = pd.DataFrame(self.fa.get_communalities(), columns=['Communalities'], index=self.data.columns)
        self.uniquenesses_df = pd.DataFrame(self.fa.get_uniquenesses(), columns=['Uniquenesses'], index=self.data.columns)

    def get_loadings(self) -> pd.DataFrame:
        """
        Get the factor loadings.
        """
        return self.loadings_df

    def get_communalities(self) -> pd.DataFrame:
        """
        Get the communalities.
        """
        return self.communalities_df

    def get_uniquenesses(self) -> pd.DataFrame:
        """
        Get the uniquenesses.
        """
        return self.uniquenesses_df

    def get_factor_scores(self) -> pd.DataFrame:
        """
        Get the factor scores for each observation.
        """
        if self.factor_scores is None:
            self.factor_scores = pd.DataFrame(self.fa.transform(self.data), columns=self._factors_lbls, index=self.data.index)
        return self.factor_scores

    def plot_loadings(self):
        """
        Plot factor loadings.
        """
        plt.figure(figsize=(10, 6))
        sns.heatmap(self.loadings_df, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
        plt.title('Factor Loadings')
        plt.xlabel('Factors')
        plt.ylabel('Variables')
        plt.show()

    def plot_factor_scores(self):
        """
        Plot factor scores for each observation.
        """
        factor_scores = self.get_factor_scores()
        factor_scores.plot(kind='hist', alpha=0.5, bins=20, figsize=(10, 6))
        plt.title('Factor Scores Distribution')
        plt.xlabel('Factor Scores')
        plt.ylabel('Frequency')
        plt.legend(title='Factors')
        plt.show()
