import uuid
import pandas as pd
import chromadb
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import numpy as np
from chromadb.utils import embedding_functions


class Books:
    def __init__(self, db_path="./chroma_db"):
        """Initialize ChromaDB with persistent storage."""
        self.chroma_client = chromadb.PersistentClient(path=db_path)  # ✅ Persist DB

        # Use ChromaDB's built-in SentenceTransformer embedding function
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        # Check if collection already exists
        existing_collections = [col.name for col in self.chroma_client.list_collections()]
        is_new = "books" not in existing_collections

        self.books_collection = self.chroma_client.get_or_create_collection(
            name="books",
            embedding_function=self.embedding_function
        )

        if is_new:
            print("📚 'books' collection not found. Creating and populating it now...")
            self.create_collection()
        else:
            print("📚 'books' collection found. Skipping creation.")
        
    def delete_books_collection(self):
        """Deletes the entire books collection from ChromaDB if it exists."""
        try:
            # Attempt to get the collection to check if it exists
            self.books_collection = self.chroma_client.get_collection('books')
            if self.books_collection:
                self.chroma_client.delete_collection('books')
                print("✅ 'books' collection deleted.")
            else:
                print("'books' collection does not exist.")
        except Exception as e:
            print(f"Error checking or deleting collection: {e}")
            
    def create_collection(self):
        """Reads books from CSV and stores them in ChromaDB."""
        df = pd.read_csv("./data/books.csv")
        for idx, row in df.iterrows():
            self.add_book(
                title=row["title"],
                author=row["author"],
                genres=row["genres"],
                rating=float(row["rating"]),
                description=row["description"],
            )

        print("Books stored in ChromaDB with default embedding.")

    def add_book(self, title, author, genres, rating, description):
        """Adds a new book to the ChromaDB collection."""
        book_id = str(uuid.uuid4())
            
        document = f"{title} by {author}. Genres: {genres}. Description: {description}"
        
        self.books_collection.add(
            ids=[book_id],
            documents=[document],
            metadatas=[{
                "title": title,
                "author": author,
                "genres": genres,
                "rating": rating,
                "description": description  
            }]
        )

    def get_all_books(self):
        return self.books_collection.get(
            include=["documents", "metadatas", "embeddings"])
        
    def get_book_by_id(self, book_id):
        """Retrieves a book from ChromaDB by its unique ID."""
        result = self.books_collection.get(ids=[str(book_id)], include=["documents", "metadatas", "embeddings"])
        return result if result["ids"] else None

    def get_book_by_name(self, title):
        """Retrieves books by title (exact match)."""
        results = self.books_collection.query(
            query_texts=[title], n_results=5, include=["documents", "metadatas", "embeddings"]
        )
        return results if results["ids"] else None

    def delete_book(self, book_id):
        """Deletes a book from ChromaDB using its ID."""
        self.books_collection.delete(ids=[book_id])
        print(f"Book with ID {book_id} deleted.")

    # This is broken and uses the old vectorizer
    def update_book_vector(self, book_id, new_description):
        """Updates a book's vector representation in ChromaDB."""
        new_embedding = self.vectorizer.fit_transform([new_description]).toarray()[0].tolist()

        # Delete old entry
        self.books_collection.delete(ids=[book_id])

        # Add updated book with new description & vector
        self.books_collection.add(
            ids=[book_id],
            documents=[new_description],  # ✅ Store updated description
            embeddings=[new_embedding],
            metadatas=[{"description": new_description}]
        )

        print(f"Book updated with ID: {book_id}")

    def delete_all_books(self):
        all_ids = self.books_collection.get()['ids']
        if all_ids:
            self.books_collection.delete(all_ids)
        else:
            print("No books to delete.")


    # Get average vector in chroma. This is used for a 'starting' point for each new user.
    def get_average_vector(self):
        all_data = self.books_collection.get(include=["embeddings"])
        
        # Extract all embeddings
        embeddings = all_data["embeddings"]
    
        # Check if embeddings is not empty
        if len(embeddings) > 0:
            average_vector = np.mean(embeddings, axis=0)
            return average_vector
        else:
            return None
    
                   