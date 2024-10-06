from connect import connect
from config import load_config
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # Primera parte de model_output contiene todos los embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Conectando a la base de datos
connection = connect(load_config())
cursor = connection.cursor()

# Frases que hemos elegido
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

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Convertir las 10 frases a embeddings
embeddings = []
for sentence in sentences_to_process:
    encoded_input = tokenizer(sentence, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    embedding = mean_pooling(model_output, encoded_input['attention_mask'])
    embeddings.append((sentence, embedding))

# Computar las 2 frases más similares (excluyendo la frase original) para cada una de ellas usando pgvector
for original_sentence, embedding in embeddings:
    # Convertir el embedding a formato que pgvector puede manejar
    float_embedding = embedding.numpy().flatten().tolist()

    # Ejecutar una consulta SQL para obtener las frases más cercanas usando pgvector, excluyendo la original
    cursor.execute("""
        SELECT sentence, embedding
        FROM sentences_pg
        WHERE sentence <> %s
        ORDER BY embedding <-> %s::vector
        LIMIT 2
    """, (original_sentence, float_embedding))

    closest_sentences = cursor.fetchall()

    print(f"For the sentence: \"{original_sentence}\"")
    for db_sentence, db_embedding in closest_sentences:
        print(f"Closest sentence: \"{db_sentence}\"")
    print()

# Cerrar conexión
cursor.close()
connection.close()
