from datasets import load_dataset
import chromadb
from chromadb.config import Settings
import time

chroma_client = chromadb.Client(Settings(
    persist_directory="./chroma_db"
))

collection = chroma_client.get_or_create_collection(name="sentences")
print("Collection created or retrieved")

ds = load_dataset("williamkgao/bookcorpus100mb")
sentences = ds['train']['text'][:10000]
print("Dataset loaded")

ids = [f"id{i}" for i in range(len(sentences))]

batch_size = 100

print("Starting insertion process...")
start_time = time.time()

for i in range(0, len(sentences), batch_size):
    print(f"Preparing batch {i // batch_size + 1}")
    batch_sentences = sentences[i:i + batch_size]
    batch_ids = ids[i:i + batch_size]
    print(f"Batch {i // batch_size + 1} ready for insertion")
    try:
        collection.upsert(documents=batch_sentences, ids=batch_ids)
    except Exception as e:
        print(f"Error: {e}")
    print(f"Batch {i // batch_size + 1} inserted")

end_time = time.time()
print(f"Insertion completed in {end_time - start_time} seconds")
