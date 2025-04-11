import sqlite3
import pandas as pd

class UserBooks:
    def __init__(self, db_path=":memory:"):
        """Initialize the database connection and create the necessary table."""
        self.conn = sqlite3.connect(db_path)
        self.create_table()

    def create_table(self):
        """Create the user_books table if it does not exist."""
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
        self.conn.commit()

    def track_book_reading(self, user_id, book_id):
        """
        Record that a user has read a book.
        
        :param user_id: ID of the user.
        :param book_id: ID of the book.
        """
        # Ensure the book exists
        book_exists = self.conn.execute("""
            SELECT 1 FROM books WHERE id = ?
        """, (book_id,)).fetchone()
    
        if not book_exists:
            print("Error: Book does not exist in the database.")
            return

        # Check if the user has already read this book
        read_books_df = self.get_books_read_by_user_id(user_id)
        if book_id in read_books_df["book_id"].values:
            print("Error: This book has already been read.")
            return
        
        # Insert into user_books table
        self.conn.execute("""
            INSERT OR IGNORE INTO user_books (user_id, book_id) VALUES (?, ?)
        """, (user_id, book_id))
        self.conn.commit()

        self.user.recalculate_user_vector(user_id, book_id)

    def get_books_read_by_user_id(self, user_id):
        """
        Retrieve all books read by a user.

        :param user_id: ID of the user.
        :return: DataFrame with columns ['user_id', 'book_id'].
        """
        query = """
            SELECT user_id, book_id FROM user_books WHERE user_id = ?
        """
        df = pd.read_sql_query(query, self.conn, params=(user_id,))
        return df


    def close_connection(self):
        """Close the database connection."""
        self.conn.close()
