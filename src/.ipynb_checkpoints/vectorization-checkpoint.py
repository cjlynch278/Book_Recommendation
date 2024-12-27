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
    def add_vectorized_data_to_df(self):
        """
        Add the vectorized data as new columns to the original DataFrame.
        """
        if self.vectorized_data is None:
            raise ValueError("The data has not been vectorized yet. Call vectorization methods first.")
        
        if 'vector' not in self.df.columns:
            self.df['vector'] = list(self.vectorized_data)

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

    def get_vector_for_book(self, book_index):
        """
        Retrieve the vectorized data for a specific book by its index.
        
        :param book_index: Index of the book in the original DataFrame.
        :return: Vectorized representation of the book as a NumPy array.
        """
        if self.vectorized_data is None:
            raise ValueError("The data has not been vectorized yet. Call vectorization methods first.")
        return self.vectorized_data[book_index]

    # Euclidean distance
    def get_vector_distance(self, vector1, vector2):
        """
        Calculate the Euclidean distance between two vectors.
    
        :param vector1: First vector as a NumPy array or list.
        :param vector2: Second vector as a NumPy array or list.
        :return: Euclidean distance as a float.
        """
        # Convert vectors to NumPy arrays for computation
        v1 = np.array(vector1)
        v2 = np.array(vector2)
        
        # Calculate the Euclidean distance
        distance = np.sqrt(np.sum((v1 - v2) ** 2))
        return distance

    
    def find_closest_vector(self, target_vector, vector_list):
        """
        Find the closest vector to the target vector from a list of vectors.
        
        :param target_vector: The target vector as a NumPy array or list.
        :param vector_list: List of vectors, each as a NumPy array or list.
        :return: The closest vector and its index in the vector list.
        """
        # Convert target vector to a NumPy array for computation
        target = np.array(target_vector)
        
        # Initialize variables to track the closest vector and minimum distance
        min_distance = float('inf')
        closest_vector = None
        closest_index = -1
        
        # Iterate over the vector list
        for idx, vec in enumerate(vector_list):
            vec = np.array(vec)  # Ensure each vector is a NumPy array
            distance = np.linalg.norm(target - vec)  # Compute Euclidean distance
            
            if distance < min_distance:
                min_distance = distance
                closest_vector = vec
                closest_index = idx
        
        return closest_vector, closest_index
