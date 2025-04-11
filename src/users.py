import uuid
import chromadb
from sklearn.feature_extraction.text import TfidfVectorizer


class Users:
    def __init__(self, db_path="./data/chroma_db"):
        """Initialize ChromaDB and persist data on disk."""
        self.vectorizer = TfidfVectorizer(max_features=20)
        self.chroma_client = chromadb.PersistentClient(path=db_path)  # Store DB on disk
        self.users_collection = self.chroma_client.get_or_create_collection(name="users")

    def create_collection(self):
        """Initialize ChromaDB collection with a default user 'Chris Lynch'."""
        default_user = "Chris Lynch"
        default_embedding = self.vectorizer.fit_transform([default_user]).toarray()[0].tolist()

        self.users_collection.add(
            ids=[str(uuid.uuid4())],
            embeddings=[default_embedding],
            metadatas=[{"name": default_user}]
        )

        print("User collection initialized with default user 'Chris Lynch'.")



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

    
    def add_user(self, name):
        """Adds a new user to the ChromaDB collection."""
        user_embedding = self.vectorizer.fit_transform([name]).toarray()[0].tolist()
        user_id = str(uuid.uuid4())

        self.users_collection.add(
            ids=[user_id],
            embeddings=[user_embedding],
            metadatas=[{"name": name}]
        )

        print(f"User '{name}' added with ID: {user_id}")

    def delete_user(self, user_id):
        """Deletes a user from ChromaDB using their ID."""
        self.users_collection.delete(ids=[user_id])
        print(f"User with ID {user_id} deleted.")

    def update_user_vector(self, user_id, new_name):
        """Updates a user's vector representation in ChromaDB."""
        new_embedding = self.vectorizer.fit_transform([new_name]).toarray()[0].tolist()

        # Delete old entry
        self.users_collection.delete(ids=[user_id])

        # Add updated user
        self.users_collection.add(
            ids=[user_id],
            embeddings=[new_embedding],
            metadatas=[{"name": new_name}]
        )

        print(f"User '{new_name}' updated with ID: {user_id}")

    def get_user_by_id(self, user_id):
        """Retrieve a user from ChromaDB by ID."""
        result = self.users_collection.get(ids=[user_id], include=["metadatas", "embeddings"])
        return result if result["ids"] else None

    def get_user_by_name(self, name):
        """Retrieve a user by searching for their name."""
        results = self.users_collection.query(
            query_texts=[name], n_results=1, include=["metadatas", "embeddings"]
        )
        return results if results["ids"] else None
