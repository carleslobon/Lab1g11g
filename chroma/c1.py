import chromadb

# Inicializando el cliente de ChromaDB
chroma_client = chromadb.Client()

# Accediendo a la colección existente
collection_name = "sentences"
collection = chroma_client.get_collection(name=collection_name)

# Recuperar las primeras 5 frases
num_sentences_to_retrieve = 5
sentence_ids = [f"sentence_{i}" for i in range(num_sentences_to_retrieve)]

# Obtener los documentos de la colección
retrieved_documents = collection.get(documents=sentence_ids)

# Mostrar las frases recuperadas
print("Retrieved documents:")
for doc in retrieved_documents:
    print(doc)
