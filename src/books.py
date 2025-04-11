import uuid
import pandas as pd
import chromadb
from sklearn.feature_extraction.text import TfidfVectorizer


class Books:
    def __init__(self, db_path="./chroma_db"):
        """Initialize ChromaDB with persistent storage."""
        self.vectorizer = TfidfVectorizer(max_features=20)
        self.chroma_client = chromadb.PersistentClient(path=db_path)  # ✅ Persist DB
        self.books_collection = self.chroma_client.get_or_create_collection(name="books")

    def create_collection(self):
        """Reads books from CSV and stores them in ChromaDB."""
        df = pd.read_csv("./data/books.csv")
        book_vectors = self.vectorizer.fit_transform(df["description"].fillna("")).toarray()

        for idx, row in df.iterrows():
            self.add_book(
                title=row["title"],
                author=row["author"],
                genres=row["genres"],
                rating=float(row["rating"]),
                description=row["description"],
                embedding=book_vectors[idx].tolist()
            )

        print("Books stored in ChromaDB with 20-dimensional TF-IDF embeddings.")

    def add_book(self, title, author, genres, rating, description, embedding):
        """Adds a new book to the ChromaDB collection."""
        book_id = str(uuid.uuid4())

        self.books_collection.add(
            ids=[book_id],
            documents=[description],  # ✅ Store description here
            embeddings=[embedding],
            metadatas=[{
                "title": title,
                "author": author,
                "genres": genres,
                "rating": rating
            }]
        )
        print(f"Book '{title}' added with ID: {book_id}")

    def get_book_by_id(self, book_id):
        """Retrieves a book from ChromaDB by its unique ID."""
        result = self.books_collection.get(ids=[book_id], include=["documents", "metadatas", "embeddings"])
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
        self.books_collection.delete(self.books_collection.get()['ids'])
