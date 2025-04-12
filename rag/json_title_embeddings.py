import os
import json
import chromadb
import argparse
import torch
import re
from chromadb.utils import embedding_functions
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

def clean_and_format_string(text):
    """
    Process a string to only include alphanumeric characters and capitalize words.
    
    Args:
        text (str): Input string to be processed
        
    Returns:
        str: Processed string with only alphanumeric characters and capitalized words
    """
    # Remove all non-alphanumeric characters and replace with space
    cleaned = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    # Remove extra spaces and split into words
    words = cleaned.split()
    # Capitalize each word and join with space
    return ''.join(word.capitalize() for word in words)

class CustomEmbeddingFunction:
    def __init__(self, model_name, device="cuda"):
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        
    def __call__(self, input):
        embeddings = self.model.encode(input, show_progress_bar=True)
        return embeddings.tolist()

def process_json_files(args):
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path=args.data_base_path)
    
    # Initialize sentence transformer model with GPU support
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create custom embedding function that uses GPU
    embedding_function = CustomEmbeddingFunction(args.model_name, device=device)
    
    # Get or create collection
    collection = client.get_or_create_collection(
        name=args.collection_name,
        embedding_function=embedding_function
    )
    
    # Iterate through all JSON files in directory
    json_files = [f for f in os.listdir(args.json_dir) if f.endswith('.json')]
    
    for json_file in tqdm(json_files, desc="Processing JSON files"):
        file_path = os.path.join(args.json_dir, json_file)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Extract title
                title = data.get('title', '')
                
                if title:
                    # Add document to collection
                    collection.add(
                        documents=[title],
                        metadatas=[{
                            'json_file': json_file
                        }],
                        ids=[f"{args.timestamp}_{json_file}"]
                    )
                    
        except Exception as e:
            print(f"Error processing {json_file}: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process JSON files and create embeddings')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--data_base_path', type=str, required=True, help='Path to the data base')
    parser.add_argument('--model_name', type=str, default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", help='Name of the model to use for embeddings')
    args = parser.parse_args()
    
    # Process each dataset directory
    for dataset_name in os.listdir(args.dataset_dir):
        dataset_path = os.path.join(args.dataset_dir, dataset_name)
        if os.path.isdir(dataset_path):
            # Extract collection name (e.g., "Weather_negative" from "Weather_negative_20241201124313")
            collection_name = clean_and_format_string('_'.join(dataset_name.split('_')[:-1]))
            # Extract timestamp (e.g., "20241201124313" from "Weather_negative_20241201124313")
            timestamp = dataset_name.split('_')[-1]
            
            args.collection_name = collection_name
            args.json_dir = dataset_path
            args.timestamp = timestamp
            
            process_json_files(args)
            
            # Print collection statistics
            client = chromadb.PersistentClient(path=args.data_base_path)
            collection = client.get_collection(collection_name)
            print(f"Collection '{collection_name}' has {len(collection.get()['ids'])} documents") 