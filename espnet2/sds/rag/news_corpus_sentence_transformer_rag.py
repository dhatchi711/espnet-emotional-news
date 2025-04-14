import os
import json
import chromadb
import torch
import re
import argparse
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import logging
import shutil


class CustomEmbeddingFunction:
    def __init__(self, model):
        self.model = model

    def __call__(self, input):
        embeddings = self.model.encode(input)
        return embeddings.tolist()


class EmbeddingSearchEngine:
    def __init__(
        self,
        data_base_path,
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ):
        """
        Initialize the search engine with database path and model

        Args:
            data_base_path (str): Path to the ChromaDB database
            model_name (str): Name of the sentence transformer model to use
        """
        self.data_base_path = data_base_path
        self.model_name = model_name
        self.client = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def setup(
        self,
        dataset_dir=None,
        embed_field="title",
        content_field="text",
        force_rebuild=False,
    ):
        """
        Set up the search engine by either loading an existing database
        or embedding new files if dataset_dir is provided
        """
        logging.info("Setting up ChromaDB client...")
        if force_rebuild:
            logging.info("Force rebuilding the database...")
            if os.path.exists(self.data_base_path):
                shutil.rmtree(self.data_base_path)
        self.client = chromadb.PersistentClient(path=self.data_base_path)

        logging.info(f"Using device: {self.device}")
        self.model = SentenceTransformer(self.model_name, device=self.device)

        if dataset_dir:
            logging.info(f"Processing dataset directory: {dataset_dir}")
            self._process_datasets(dataset_dir, embed_field, content_field)

        collections = self.client.list_collections()
        logging.info(f"Available collections: {[c.name for c in collections]}")

    def _clean_and_format_string(self, text):
        """Process a string to only include alphanumeric characters and capitalize words"""
        cleaned = re.sub(r"[^a-zA-Z0-9]", " ", text)
        words = cleaned.split()
        return "".join(word.capitalize() for word in words)

    def _process_datasets(self, dataset_dir, embed_field, content_field):
        """Process datasets in the given directory"""
        for dataset_name in os.listdir(dataset_dir):
            dataset_path = os.path.join(dataset_dir, dataset_name)
            if os.path.isdir(dataset_path):
                collection_name = self._clean_and_format_string(
                    "_".join(dataset_name.split("_")[:-1])
                )
                timestamp = dataset_name.split("_")[-1]
                logging.info(
                    f"Processing dataset: {dataset_name} (Collection: {collection_name})"
                )

                self._process_json_files(
                    collection_name, dataset_path, timestamp, embed_field, content_field
                )

    def _process_json_files(
        self, collection_name, json_dir, timestamp, embed_field, content_field
    ):
        """Process JSON files in a directory and add them to a collection"""

        embedding_function = CustomEmbeddingFunction(self.model)

        collection = self.client.get_or_create_collection(
            name=collection_name, embedding_function=embedding_function
        )
        json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")][
            :2
        ]  # TODO(shikhar): change 10

        for json_file in tqdm(json_files, desc=f"Processing {collection_name}"):
            file_path = os.path.join(json_dir, json_file)

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                    language = data.get('language', 'none')
                    if language != 'english':
                        continue

                    text_to_embed = data.get(embed_field, "")
                    print(text_to_embed)
                    content = data.get(content_field, "")

                    if text_to_embed:
                        collection.add(
                            documents=[text_to_embed],
                            metadatas=[{"content": content}],
                            ids=[f"{timestamp}_{json_file}"],
                        )

            except Exception as e:
                logging.error(f"Error processing {json_file}: {str(e)}")

        logging.info(
            f"Collection '{collection_name}' has {collection.count()} documents"
        )

    def query(self, query_text, collection_name=None, top_k=5, threshold=None):
        """
        Query the database with a text string

        Args:
            query_text (str): Text to search for
            collection_name (str, optional): Name of collection to search in
            top_k (int): Number of top results to return
            threshold (float, optional): Minimum similarity threshold (0-1)

        Returns:
            list: List of dictionaries containing search results
        """
        if not self.client:
            raise ValueError("Search engine not set up. Call setup() first.")

        if collection_name:
            collections = [self.client.get_collection(name=collection_name)]
        else:
            collections = self.client.list_collections()

        all_results = []

        query_embedding = self.model.encode([query_text])[0].tolist()
        for collection in collections:
            logging.info(f"Querying collection: {collection.name}")
            query_result = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "distances", "metadatas"],
            )
            # ids, documents, distances

            if not query_result["ids"] or not query_result["ids"][0]:
                continue

            n_results = len(query_result["ids"][0])
            for i in range(n_results):
                doc_id = query_result["ids"][0][i]
                document = query_result["documents"][0][i]
                distance = query_result["distances"][0][i]
                metadata = query_result["metadatas"][0][i]

                all_results.append(
                    {
                        "collection": collection.name,
                        "id": doc_id,
                        "text": document,
                        "content": metadata.get("content", ""),
                        "distance": distance,
                    }
                )

        all_results.sort(key=lambda x: x["distance"])
        print(len(all_results), "before filtering")

        if threshold is not None:
            all_results = [r for r in all_results if r["distance"] < threshold]
        print(len(all_results), "after filtering", top_k)

        return all_results[:top_k]


def main():
    """Command-line interface for the EmbeddingSearchEngine"""
    parser = argparse.ArgumentParser(
        description="Process JSON files and search with embeddings"
    )
    parser.add_argument("--dataset_dir", type=str, help="Path to the dataset directory")
    parser.add_argument(
        "--data_base_path", type=str, required=True, help="Path to the database"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="Name of the model to use for embeddings",
    )
    parser.add_argument("--query", type=str, help="Optional query to test search")
    parser.add_argument(
        "--collection", type=str, help="Optional collection name to search in"
    )
    parser.add_argument(
        "--top_k", type=int, default=5, help="Number of results to return"
    )
    parser.add_argument(
        "--threshold", type=float, help="Minimum similarity threshold (0-1)"
    )
    parser.add_argument(
        "--force_rebuild", action="store_true", help="Force rebuild the database"
    )
    args = parser.parse_args()

    # Create and setup the search engine
    engine = EmbeddingSearchEngine(
        data_base_path=args.data_base_path, model_name=args.model_name
    )

    # Setup with dataset directory if provided
    engine.setup(dataset_dir=args.dataset_dir, force_rebuild=args.force_rebuild)

    # If a query was provided, test search functionality
    if args.query:
        print(f"\nSearching for: '{args.query}'")
        print("-" * 50)

        results = engine.query(
            query_text=args.query,
            collection_name=args.collection,
            top_k=args.top_k,
            threshold=args.threshold,
        )

        if not results:
            print("No results found.")
        else:
            for i, result in enumerate(results, 1):
                print(f"Result {i}")
                print(f"Collection: {result['collection']}")
                print(f"Document ID: {result['id']}")
                print(f"Text: {result['text']}")
                print(f"Distance: {result['distance']:.4f}")

                # Show preview of content
                content_preview = (
                    result["content"][:200] + "..."
                    if len(result["content"]) > 200
                    else result["content"]
                )
                print(f"Content preview: {content_preview}")
                print("-" * 50)


if __name__ == "__main__":
    main()
