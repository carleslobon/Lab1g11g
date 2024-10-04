import chromadb

# Inicializa el cliente de ChromaDB
chroma_client = chromadb.Client()

# Obtiene todas las colecciones
collections = chroma_client.list_collections()

# Borra cada colección
for collection in collections:
    print(f"Borrando colección: {collection['name']}")
    chroma_client.delete_collection(name=collection['name'])

print("Todas las colecciones han sido borradas.")
