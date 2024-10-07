import chromadb
import time
import statistics

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

client = chromadb.Client()

collection_name = "book_corpus_sentences"
collection = client.create_collection(name=collection_name)

with open('chroma/bookcorpus100mb.txt', 'r', encoding='utf-8') as file:
    sentences = file.readlines()

sentences = [sentence.strip() for sentence in sentences][:10000]

add_times = []
query_times = []

for sentence in sentences:
    start_time = time.time()
    collection.add(
        documents=[sentence],
        ids=[str(hash(sentence))]
    )
    end_time = time.time()
    add_times.append(end_time - start_time)

print(f"Estadísticas de tiempo de collection.add:")
print(f"Tiempo máximo: {max(add_times):.4f} segundos")
print(f"Tiempo mínimo: {min(add_times):.4f} segundos")
print(f"Tiempo promedio: {sum(add_times) / len(add_times):.4f} segundos")
print(f"Desviación estándar: {statistics.stdev(add_times):.4f} segundos\n")


for sentence in sentences_to_process:
    start_time = time.time()
    results = collection.query(
        query_texts=[sentence],
        n_results=2
    )
    end_time = time.time()
    query_times.append(end_time - start_time)

    print(f"Para la sentencia: '{sentence}'")
    print("Sentencias más similares:")
    for similar_doc in results['documents']:
        print(f" - {similar_doc}")

print(f"\nEstadísticas de tiempo de collection.query:")
print(f"Tiempo máximo: {max(query_times):.4f} segundos")
print(f"Tiempo mínimo: {min(query_times):.4f} segundos")
print(f"Tiempo promedio: {sum(query_times) / len(query_times):.4f} segundos")
print(f"Desviación estándar: {statistics.stdev(query_times):.4f} segundos")

