import sqlite3
import numpy as np
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Misnamed, this is both user and *book* operations

class UserOperations:
    def __init__(self, db_name="user_book_db.sqlite",config=None):
        """
        Initialize the database connection and create tables if they do not exist.
        
        :param db_name: Name of the SQLite database file.
        """
        self.conn = sqlite3.connect('./data/books.db')
        self.config = config if config else {}
        self.vectorized_data = None
        self.feature_names = []
        self.vectorizer = TfidfVectorizer(max_features=20)















    


    #OLD
    def create_tables(self):
        """
        Create the necessary tables in the database.
        """
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    vector TEXT
                )
            """)

            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS user_books (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    book_id INTEGER,
                    FOREIGN KEY (user_id) REFERENCES users(user_id),
                    FOREIGN KEY (book_id) REFERENCES books(book_id),
                    UNIQUE (user_id, book_id)
                )
            """)

            # Books table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS books (
                    id INTEGER PRIMARY KEY,
                    title TEXT NOT NULL,
                    short_title TEXT,
                    title_seo TEXT,
                    genres TEXT,
                    author TEXT,
                    description TEXT,
                    rating REAL,
                    vector TEXT
                )
            """)


    ###################################
    ############ USER Operations ######
    ###################################

    def add_user(self, name, vector=None):
        """
        Add a user to the database.
        
        :param name: Name of the user.
        :param vector: Vector representation of the user (as a list or NumPy array).
        """
        vector_str = json.dumps(vector.tolist() if isinstance(vector, np.ndarray) else vector)
        with self.conn:
            self.conn.execute("""
                INSERT OR IGNORE INTO users (name, vector) VALUES (?, ?)
            """, (name, vector_str))


    def get_user_vector_by_id(self, id):
        """
        Retrieve the vector representation of a user.
        
        :param user_name: Name of the user.
        :return: User vector as a NumPy array or None if not set.
        """
        user = self.conn.execute("SELECT vector FROM users WHERE user_id = ?", (id,)).fetchone()
        return np.array(json.loads(user[0])) if user and user[0] else None

    def get_user_vector_by_user_name(self, user_name):
        """
        Retrieve the vector representation of a user.
        
        :param user_name: Name of the user.
        :return: User vector as a NumPy array or None if not set.
        """
        user = self.conn.execute("SELECT vector FROM users WHERE name = ?", (user_name,)).fetchone()
        return np.array(json.loads(user[0])) if user and user[0] else None

    def get_vector_by_user_id(self, user_id):
        """
        Retrieve the vector representation of a user.
        
        :param user_name: Name of the user.
        :return: User vector as a NumPy array or None if not set.
        """
        user = self.conn.execute("SELECT vector FROM users WHERE user_id = ?", (user_id,)).fetchone()
        return np.array(json.loads(user[0])) if user and user[0] else None


    def get_user_by_id(self, user_id):
        """
        Get a user by its ID and return the result as a DataFrame with all columns.
        
        :param user_id: id of the user.
        :return: DataFrame containing id details.
        """
        # Fetch the user details from the database
        user = self.conn.execute("""
            SELECT user_id, name, vector 
            FROM users WHERE user_id = ?
        """, (user_id,)).fetchall()

        # Convert the result to a DataFrame
        if user:
            columns = ['user_id', 'name', 'vector']
            df = pd.DataFrame(user, columns=columns)
            return df
        else:
            return pd.DataFrame()  # Return an empty DataFrame if no user is found

    def update_user_vector_by_user_name(self, user_name, vector):
        """
        Update the vector representation of a user.
        
        :param user_name: Name of the user.
        :param vector: New vector (as a list or NumPy array).
        """
        vector_str = json.dumps(vector.tolist() if isinstance(vector, np.ndarray) else vector)
        with self.conn:
            self.conn.execute("""
                UPDATE users SET vector = ? WHERE name = ?
            """, (vector_str, user_name))

    def update_user_vector_by_id(self, user_id, vector):
        """
        Update the vector representation of a user.
        
        :param user_name: Name of the user.
        :param vector: New vector (as a list or NumPy array).
        """
        vector_str = json.dumps(vector.tolist() if isinstance(vector, np.ndarray) else vector)
        with self.conn:
            self.conn.execute("""
                UPDATE users SET vector = ? WHERE user_id = ?
            """, (vector_str, user_id))
   
    def overwrite_table(self, table_name, df):
        """
        Overwrite the specified table in the database with the contents of the provided DataFrame.
        
        :param table_name: Name of the table to overwrite.
        :param df: A Pandas DataFrame containing the data to insert.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a Pandas DataFrame.")
        
        try:
            # Begin a transaction
            self.conn.execute("BEGIN TRANSACTION;")
            
            # Clear the table
            self.conn.execute(f"DELETE FROM {table_name};")
            
            # Insert the new data
            df.to_sql(table_name, self.conn, if_exists="append", index=False)
            
            # Commit the transaction
            self.conn.commit()
            print(f"Table '{table_name}' has been successfully overwritten.")
        
        except Exception as e:
            # Rollback in case of error
            self.conn.rollback()
            print(f"Error overwriting table '{table_name}': {e}")

    def close(self):
        """
        Close the database connection.
        """
        self.conn.close()


    ###################################
    ############ BOOK Operations ######
    ###################################

    def get_books_read_by_user_id(self, user_id):
        """
        Get a list of books read by a user, returned as a DataFrame.
        
        :param user_id: id of the user.
        :return: DataFrame with columns 'user_id' and 'book_id'.
        """
        if not user_id:
            return pd.DataFrame()  # Return an empty DataFrame if no user_id is provided
        
        # Corrected query with proper parameter passing
        query = """
            SELECT user_id, book_id 
            FROM user_books 
            WHERE user_id = ?
        """
        # Execute the query and fetch all results
        books = self.conn.execute(query, (user_id,)).fetchall()
        
        # Convert the result into a DataFrame with columns 'user_id' and 'book_id'
        books_df = pd.DataFrame(books, columns=['user_id', 'book_id'])
        
        print(books_df)
        
        all_books = pd.DataFrame()
        for id in books_df['book_id']:
            book = self.get_book_by_id(id)
            all_books = pd.concat([all_books, book], ignore_index=True)

        return all_books

    def add_or_update_vector_column(self, df):
        """
        Add or update the vector column in the books table using SQLite.
    
        :param df: DataFrame with the updated 'vector' column.
        """
        # Ensure the vector is stored as a JSON string
        df['vector'] = df['vector'].apply(lambda x: json.dumps(x.tolist()))
    
        # Add the vector column if it doesn't exist
        if 'vector' not in df.columns:
            self.conn.execute("ALTER TABLE books ADD COLUMN vector TEXT") 
            
        # Update each row with the corresponding vector
        for _, row in df.iterrows():
            self.conn.execute("""
                UPDATE books
                SET vector = ?
                WHERE id = ?
            """, (row['vector'], row['id']))
    
        self.conn.commit()
    def get_all_vectors(self):
        """
        Retrieve all vectors from the books table and return as a NumPy array.
        
        :return: NumPy array of vectors.
        """
        # Execute the query to fetch all vectors
        cursor = self.conn.execute("""
            SELECT vector FROM books
        """)
    
        # Convert the results to a NumPy array
        vectors = [
            np.array(json.loads(row[0]))  # Convert JSON string back to NumPy array
            for row in cursor.fetchall()
        ]
    
        # Return as a 2D NumPy array
        return np.array(vectors)
        
    def overwrite_books_table(self, df):
        """
        Overwrites the books table with the given dataframe. The vector column is converted to JSON format.
    
        :param df: A pandas DataFrame containing the book data.
        """
        # Ensure the table exists
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS books (
                id INTEGER PRIMARY KEY,
                title TEXT,
                short_title TEXT,
                title_seo TEXT,
                genres TEXT,
                author TEXT,
                description TEXT,
                rating REAL,
                vector TEXT
            )
        """)
            
        # Clear the existing records in the table
        self.conn.execute("DELETE FROM books")
            
        # Insert the new data
        for _, row in df.iterrows():
            # Convert the vector column to JSON format if it exists
            vector_json = json.dumps(row['vector'].tolist()) if isinstance(row['vector'], np.ndarray) else row['vector']
                
            self.conn.execute("""
                INSERT INTO books (title, short_title, title_seo, genres, author, description, rating, vector)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                row['title'],
                row['short_title'],
                row['title_seo'],
                row['genres'],
                row['author'],
                row['description'],
                row['rating'],
                vector_json
            ))
            
        # Commit the changes and close the connection
        self.conn.commit()

    def add_book(self, title, author, description):
        """
        Add a book to the database.
        
        :param title: Title of the book.
        :param author: Author of the book.
        :param description: Description of the book.
        """
        with self.conn:
            self.conn.execute("""
                INSERT OR IGNORE INTO books (title, author, description) VALUES (?, ?, ?)
            """, (title, author, description))

    def track_book_reading(self, user_id, book_id):
        """
        Record that a user has read a book.
        
        :param user_name: Name of the user.
        :param book_title: Title of the book.
        """
        # Step 1: Ensure the book exists in the database
        book_exists = self.conn.execute("""
            SELECT 1 FROM books WHERE id = ?
        """, (book_id,)).fetchone()
    
        if not book_exists:
            print("Error: Book does not exist in the database.")
            return
        # Check to see if book has already been read
        read_books_df = self.get_books_read_by_user_id(user_id)
        if book_id in read_books_df['book_id'].values:
            print("Error: this has been read")
            return
        
        self.conn.execute("""
            INSERT OR IGNORE INTO user_books (user_id, book_id) VALUES (?, ?)
        """, (user_id, book_id))
        self.conn.commit()

        self.recalculate_user_vector(user_id, book_id)


    def get_all_books(self):
        # Fetch the book details from the database
        books = self.conn.execute("""
            SELECT id, title, short_title, title_seo, genres, author, description, rating, vector 
            FROM books
        """).fetchall()

        # Convert the result to a DataFrame
        if books:
            columns = ['id', 'title', 'short_title', 'title_seo', 'genres', 'author', 'description', 'rating', 'vector']
            df = pd.DataFrame(books, columns=columns)
            return df
    def get_book_by_id(self, book_id):
        """
        Get a book by its ID and return the result as a DataFrame with all columns.
        
        :param book_id: id of the book.
        :return: DataFrame containing book details.
        """
        # Fetch the book details from the database
        book = self.conn.execute("""
            SELECT id, title, short_title, title_seo, genres, author, description, rating, vector 
            FROM books WHERE id = ?
        """, (book_id,)).fetchall()

        # Convert the result to a DataFrame
        if book:
            columns = ['id', 'title', 'short_title', 'title_seo', 'genres', 'author', 'description', 'rating', 'vector']
            df = pd.DataFrame(book, columns=columns)
            return df
        else:
            return pd.DataFrame()  # Return an empty DataFrame if no book is found


    def get_book_vector_by_id(self, book_id):
        """
        Retrieve the vector representation of a book.
        
        :param book_id: id of the book.
        :return: Id vector as a NumPy array or None if not set.
        """
        book = self.conn.execute("SELECT vector FROM books WHERE id = ?", (book_id,)).fetchone()
        return np.array(json.loads(book[0])) if book and book[0] else None
        



    def load_data(self, filepath):
        """
        Load a CSV file into a Pandas DataFrame.
        
        :param filepath: Path to the CSV file.
        """
        self.df = pd.read_csv(filepath)



    def recalculate_user_vector(self, user_id, book_id):
        """
        Recalculate the user vector when they read a new book.
        
        :param user_id: ID of the user.
        :param book_id: Index of the newly read book in the books DataFrame.
        :return: Updated user vector as a NumPy array.
        """
        # Get the vector for the new book
        print(f"New Book: {self.get_book_by_id(book_id)}")
        new_book_vector = self.get_book_by_id(book_id)['vector']
        print(new_book_vector)
        # Retrieve the current user vector
        user_vector = self.get_vector_by_user_id(user_id)
        
        if user_vector is None:
            # If the user does not have an existing vector, initialize it with the new book's vector
            new_user_vector = new_book_vector
        else:
            # Combine the new book vector with the existing user vector (average or sum)
            user_vector =  self.get_vector_by_user_id(user_id)
            book_vector = self.get_book_vector_by_id(book_id)
            
            new_user_vector = self.get_average_vector(user_vector,book_vector)

        #Commit to db?
        return new_user_vector

    def vectorize_initial_text(self, df):
        """
        Vectorize text data using TF-IDF and append to the vectorized dataset.
        :param dataframe: dataframe with columns to vectorizessa
        
        """


        # Backlog, would it be smarter to simply retrieve the current df?
        temp_df = df
        # Combine text features into a single string for vectorization
        temp_df['combined_text'] = df['title'] + " " + df['short_title'] + " " + df['description']
        
        # Fit and transform the combined text
        vectors = self.vectorizer.fit_transform(temp_df['combined_text'])
        
        # Add vectors to the DataFrame
        df['vector'] = list(vectors.toarray())

        self.user_operations.add_or_update_vector_column(df)
        return df
        
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


    def get_average_vector(self, vector1, vector2):

        vector1_np = np.array(vector1)
        vector2_np = np.array(vector2)
        
        average_vector = (vector1_np + vector2_np) / 2
        
        average_vector = average_vector.tolist()

        return average_vector