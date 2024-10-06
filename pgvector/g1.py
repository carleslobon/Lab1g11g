from connect import connect
from config import load_config
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import psycopg2
import time

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # Primera parte de model_output contiene todos los embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Conectando a la base de datos
connection = connect(load_config())
cursor = connection.cursor()

# Leer las frases de la base de datos
cursor.execute("SELECT id, sentence FROM sentences_pg")  # Asegúrate de usar el nombre correcto de la tabla
sentences = cursor.fetchall()
ids = [row[0] for row in sentences]
sentences = [row[1] for row in sentences]

# Inicializar el tokenizer y el modelo
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Calcular embeddings para todas las frases
group_sz = 500  # Tamaño del grupo

# Iniciar el temporizador
start_time = time.time()

for i in range(0, len(sentences), group_sz):
    batch = sentences[i:i + group_sz]
    
    # Tokenizar las frases
    encoded_input = tokenizer(batch, padding=True, truncation=True, return_tensors='pt')

    with torch.no_grad():
        model_output = model(**encoded_input)

    # Obtener los embeddings
    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    embeddings = F.normalize(embeddings, p=2, dim=1)

    # Convertir embeddings a lista
    float_embeddings = embeddings.numpy().tolist()  # Convertir a lista

    # Actualizar la base de datos
    for id, embedding in zip(ids[i:i + group_sz], float_embeddings):
        cursor.execute("UPDATE sentences_pg SET embedding = %s WHERE id = %s", (embedding, id))
    
    print(f"Added embeddings for sentences {i + 1} to {min(i + group_sz, len(sentences))}.")

# Confirmar cambios
connection.commit()

end_time = time.time()
print(f"The total updating time with embeddings is: {end_time - start_time} seconds")

# Cerrar conexión
cursor.close()
connection.close()
