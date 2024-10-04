from datasets import load_dataset
import chromadb
import time

chroma_client = chromadb.Client()

collection = chroma_client.get_or_create_collection(name="sentences")
print(collection)

ds = load_dataset("williamkgao/bookcorpus100mb")
sentences = ds['train']['text'][:10000]

ids = [f"id{i}" for i in range(len(sentences))]

start_time = time.time()
for sentence, id in zip(sentences, ids):
    collection.upsert(documents=[sentence], ids=[id])
end_time = time.time()

print(f"Total insertion time: {end_time - start_time} seconds")
