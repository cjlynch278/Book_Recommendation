import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

class Vectorization:
    def __init__(self, config=None):
        """
        Initialize the Vectorization class.
        
        :param config: A dictionary containing configuration options like max_features for TF-IDF.
        """
        self.config = config if config else {}
        self.vectorized_data = None
        self.feature_names = []

    def load_data(self, filepath):
        """
        Load a CSV file into a Pandas DataFrame.
        
        :param filepath: Path to the CSV file.
        """
        self.df = pd.read_csv(filepath)

    def vectorize_text(self, column):
        """
        Vectorize text data using TF-IDF and append to the vectorized dataset.
        
        :param column: Column name in the DataFrame to vectorize.
        """
        tfidf = TfidfVectorizer(max_features=self.config.get("max_features", 100))
        encoded = tfidf.fit_transform(self.df[column].fillna(""))
        self._append_to_vectorized_data(encoded.toarray(), tfidf.get_feature_names_out())

    def vectorize_categorical(self, column):
        """
        One-hot encode categorical data and append to the vectorized dataset.
        
        :param column: Column name in the DataFrame to encode.
        """
        encoder = OneHotEncoder(sparse_output=False)
        encoded = encoder.fit_transform(self.df[[column]].fillna("Unknown"))
        self._append_to_vectorized_data(encoded, encoder.get_feature_names_out([column]))

    def scale_numeric(self, column):
        """
        Scale numeric data and append to the vectorized dataset.
        
        :param column: Column name in the DataFrame to scale.
        """
        scaler = StandardScaler()
        encoded = scaler.fit_transform(self.df[[column]].fillna(0))
        self._append_to_vectorized_data(encoded, [column])

    def _append_to_vectorized_data(self, data, feature_names):
        """
        Helper method to append new features to the vectorized data.
        
        :param data: NumPy array of the new data.
        :param feature_names: List of feature names for the new data.
        """
        if self.vectorized_data is None:
            self.vectorized_data = data
        else:
            self.vectorized_data = np.hstack((self.vectorized_data, data))
        self.feature_names.extend(feature_names)

    def get_vectorized_data(self):
        """
        Retrieve the final vectorized data.
        
        :return: NumPy array of vectorized data.
        """
        return self.vectorized_data

    def get_feature_names(self):
        """
        Retrieve the feature names for the vectorized data.
        
        :return: List of feature names.
        """
        return self.feature_names
