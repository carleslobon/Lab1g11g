import chromadb
import time
import numpy as np
import statistics
from numpy.linalg import norm

# Sentences we chose
sentences_to_process = [
    "usually , he would be tearing around the living room , playing with his toys .",
    "but just one look at a minion sent him practically catatonic .",
    "that had been megan 's plan when she got him dressed earlier .",
    "he 'd seen the movie almost by mistake , considering he was a little young for the pg cartoon , but with older cousins , along with her brothers , mason was often exposed to things that were older .",
    "she liked to think being surrounded by adults and older kids was one reason why he was a such a good talker for his age .",
    "`` are n't you being a good boy ? ''",
    "she said .",
    "mason barely acknowledged her .",
    "instead , his baby blues remained focused on the television .",
    "`` beau ? ''"
]

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

client = chromadb.PersistentClient(path = "./chroma_db")

collection_name = "book_corpus_sentences"
collection = client.get_collection(name=collection_name)

query_times = []

for sentence in sentences_to_process:
    start_time = time.time()
    results = collection.query(
        query_texts=[sentence],
        n_results=3,
    )

    query_embedding = results['distances'][0]
    all_embeddings = results['distances']

    cos_similarities = [cosine_similarity(query_embedding, emb) for emb in all_embeddings]
    
    sorted_similarities = sorted(zip(results['documents'], cos_similarities), key=lambda x: x[1], reverse=True)

    end_time = time.time()
    query_times.append(end_time - start_time)

    print(f"Para la sentencia: '{sentence}'")
    print("Sentencias más similares euclidianas:")
    for similar_doc in results['documents'][:3]:
        print(f" - {similar_doc}")

    print(f"Para la sentencia: '{sentence}'")
    print("Sentencias más similares por coseno:")
    for doc, similarity in sorted_similarities[:3]: 
        print(f" - {doc} (similaridad cosinus: {similarity:.4f})")

print(f"\nEstadísticas de tiempo de collection.query:")
print(f"Tiempo máximo: {max(query_times):.4f} segundos")
print(f"Tiempo mínimo: {min(query_times):.4f} segundos")
print(f"Tiempo promedio: {sum(query_times) / len(query_times):.4f} segundos")
print(f"Desviación estándar: {statistics.stdev(query_times):.4f} segundos")
