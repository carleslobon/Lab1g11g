import chromadb

client = chromadb.PersistentClient(path = "./chroma_db")

collection_name = "book_corpus_sentences"
collection = client.get_or_create_collection(name=collection_name)

with open('chroma/bookcorpus100mb.txt', 'r', encoding='utf-8') as file:
    sentences = file.readlines()

sentences = [sentence.strip() for sentence in sentences][:10000]

for sentence in sentences:
    collection.add(
        documents=[sentence],
        ids=[str(hash(sentence))]
    ) #This sentence automatically embeds all the phrases so we don't need any change in this code to embed the stored phrases

print(f"Se han agregado {len(sentences)} frases a la colección '{collection_name}'.")