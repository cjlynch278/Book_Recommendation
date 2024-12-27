import sqlite3
import numpy as np
import json
import pandas as pd

class UserOperations:
    def __init__(self, db_name="user_book_db.sqlite"):
        """
        Initialize the database connection and create tables if they do not exist.
        
        :param db_name: Name of the SQLite database file.
        """
        self.conn = sqlite3.connect('./data/books.db')


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


    def get_user_vector(self, user_name):
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
    ############ BOOK Operations#######
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
        
        return books_df



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

        self.conn.execute("""
            INSERT OR IGNORE INTO user_books (user_id, book_id) VALUES (?, ?)
        """, (user_id, book_id))
        self.conn.commit()

    def get_book_by_id(self, book_id):

        book = self.conn.execute("""
                select * from books where id = ?
            """, str(book_id)).fetchall()

        return book
                