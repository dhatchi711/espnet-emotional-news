import os
import json
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import torch
from json_title_embeddings import CustomEmbeddingFunction

def list_collections(db_path):
    client = chromadb.PersistentClient(path=db_path)
    collections = client.list_collections()
    print("Available collections:")
    for col in collections:
        print(f" - {col.name}")
    return [col.name for col in collections]


class SimpleEmbeddingFunction:
    def __init__(self, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)
        
    def __call__(self, texts):
        return self.model.encode(texts, show_progress_bar=False).tolist()

def query_news(
    dataset_dir,
    db_path,
    query_text,
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    k_per_collection=5,
    final_top_k=5,
    verbose=True
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_fn = CustomEmbeddingFunction(model_name, device=device)
    query_embedding = embedding_fn([query_text])[0]

    client = chromadb.PersistentClient(path=db_path)
    collections = client.list_collections()
    all_results = []

    for col in collections:
        col_name = col.name
        if verbose:
            print(f"Querying collection: {col_name}")
        collection = client.get_collection(name=col_name, embedding_function=embedding_fn)

        try:
            results = collection.query(query_embeddings=[query_embedding], n_results=k_per_collection)
            for doc, meta, id_, dist in zip(
                results['documents'][0], results['metadatas'][0], results['ids'][0], results['distances'][0]
            ):
                all_results.append({
                    "collection": col_name,
                    "title": doc,
                    "metadata": meta,
                    "id": id_,
                    "distance": dist
                })
        except Exception as e:
            if verbose:
                print(f"Error querying {col_name}: {e}")

    # Sort by distance (lower = more similar)
    all_results.sort(key=lambda x: x['distance'])
    if verbose:
        print(f"\nTop {final_top_k} results across all collections:")

    results = []
    for i, result in enumerate(all_results[:final_top_k]):
        file_dir = os.path.join(dataset_dir, result['collection'] + "_" + result['id'].split("_")[0], result['metadata']['json_file'])
        with open(file_dir, 'r') as f:
            data = json.load(f)
            result['content'] = data
        
        if verbose:
            print(f"\n--- Result {i+1} ---")
            print(f"Collection: {result['collection']}")
            print(f"Title: {result['title']}")
            print(f"Metadata: {result['metadata']}")
            print(f"ID: {result['id']}")
            print(f"Distance: {result['distance']:.4f}")
        results.append(result)
    return results

if __name__ == "__main__":
    results = query_news(dataset_dir="/data/user_data/jiaruil5/11692/free-news-datasets/News_Datasets", db_path="/data/user_data/jiaruil5/11692/news_db", query_text="Any news about Lionel Messi?", model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", k_per_collection=5, final_top_k=1, verbose=False)
    print(results)
