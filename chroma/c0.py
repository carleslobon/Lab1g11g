import chromadb

client = chromadb.Client()

collection_name = "book_corpus_sentences"
collection = client.create_collection(name=collection_name)

with open('chroma/bookcorpus100mb.txt', 'r', encoding='utf-8') as file:
    sentences = file.readlines()

sentences = [sentence.strip() for sentence in sentences][:10000]

for sentence in sentences:
    collection.add(
        documents=[sentence],
        ids=[str(hash(sentence))]
    )

print(f"Se han agregado {len(sentences)} frases a la colección '{collection_name}'.")

