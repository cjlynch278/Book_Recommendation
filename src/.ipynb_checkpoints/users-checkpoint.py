import uuid
import chromadb
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from chromadb.utils import embedding_functions
from src.books import Books


class Users:
    def __init__(self, db_path="./chroma_db"):
        """Initialize ChromaDB and persist data on disk."""
        self.chroma_client = chromadb.PersistentClient(path=db_path)  # ✅ Persist DB

        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        self.books = Books()
        

        self.users_collection = self.chroma_client.get_or_create_collection(
            name="users",
            embedding_function=self.embedding_function
        )

        
    def delete_users_collection(self):
        """Deletes the entire users collection from ChromaDB if it exists."""
        try:
            # Attempt to get the collection to check if it exists
            self.books_collection = self.chroma_client.get_collection('books')
            if self.users_collection:
                self.chroma_client.delete_collection('users')
                print("✅ 'users' collection deleted.")
            else:
                print("'users' collection does not exist.")
        except Exception as e:
            print(f"Error checking or deleting collection: {e}")
            
    
    def get_books_read_by_user_id(self, user_id: str):
        """Returns all books read by a specific user based on user_id."""
        results = self.books.books_collection.get(
            where={"user_id": user_id},
            include=["metadatas", "documents"]
        )
        return results

    def track_book_reading(self, user_id, book_id):
        """
        Record that a user has read a book using ChromaDB.
        """
        book = self.books.get_book_by_id(book_id)
    
        if not book or not book["ids"]:
            print("Error: Book does not exist in the database.")
            return
    
        # Check if the user already read this book
        read_books = self.get_books_read_by_user_id(user_id)
        if book_id in read_books.get("ids", []):
            print("Error: This book has already been read by the user.")
            return
    
        # Add a new record associating this user with the book
        original_metadata = book["metadatas"][0]
        new_metadata = original_metadata.copy()
        new_metadata["user_id"] = user_id
    
        self.books.books_collection.add(
            ids=[book_id],
            documents=[book["documents"][0]],
            embeddings=[book["embeddings"][0]],
            metadatas=[new_metadata]
        )
    
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
        user_embedding = self.books.get_average_vector()
        user_id = str(uuid.uuid4())

        self.users_collection.add(
            ids=[user_id],
            embeddings=[user_embedding],
            metadatas=[{"name": name}]
        )

        print(f"User '{name}' added with ID: {user_id}")

    def delete_user(self, user_id):
        """Deletes a user from ChromaDB using their ID, but only if they exist."""
        # Check if user exists
        user = self.users_collection.get(ids=[user_id], include=["metadatas"])
        
        if not user["ids"]:  
            print(f"User with ID {user_id} not found. Nothing deleted.")
            return
        
        # If user exists, delete
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


    def recalculate_user_vector(self, user_id, new_book_vector):
        # Fetch existing user vector and number of books read
        user_data = self.users_collection.get(
            where={"user_id": user_id}, 
            include=["embeddings", "metadatas"]
        
        )
        old_user_vector = user_data["embeddings"][0]
        num_books_read = user_data["metadatas"][0].get("num_books_read", 0)
    
        # Weighted average update
        updated_vector = (
            (np.array(old_user_vector) * num_books_read + np.array(new_book_vector))
            / (num_books_read + 1)
        )
    
        # Update user's record
        self.users_collection.update(
            where={"user_id": user_id},
            embeddings=[updated_vector.tolist()],
            metadatas=[{"num_books_read": num_books_read + 1}]
        )

